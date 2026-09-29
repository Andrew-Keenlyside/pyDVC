"""Volume bricks read from local files straight into GPU memory (GPUDirect Storage).

Read paths, chosen by ``volumes.gpu_io`` (``$ZVDVC_GPU_IO`` overrides ``auto``):

``host``
    Decode on the host into pinned memory, then one copy to the device
    (:func:`zvdvc.io.volume.to_device`). What every earlier measurement used.
``kvikio``
    kvikio reads the stored bytes into device memory, and zstd chunks are
    decompressed on the GPU by nvCOMP. kvikio goes through cuFile, which uses
    GPUDirect Storage (NVMe to GPU, no host copy) where the system supports it
    and a POSIX bounce buffer otherwise, so this path works, and gives the same
    bytes, on any machine with kvikio installed.
``auto``
    ``kvikio`` when GPUDirect Storage is actually available, ``host`` otherwise.

"Available" is cuFile's own verdict, ``kvikio.cufile_driver.properties.is_gds_available``:
it needs the nvidia-fs driver and a supported filesystem. kvikio's
``defaults.is_compat_mode_preferred()`` is False whenever libcufile loads, even
when cuFile itself then falls back to POSIX reads because nvidia-fs is missing,
so it cannot tell. :func:`status` reports both.

Layouts read on the device; anything else falls back to the host path (the
source records why, :attr:`DeviceReader.why_not`):

* flat files (``.raw``, ``.mhd``, C-ordered ``.npy``), any byte order and header;
* Zarr v3 arrays on a local store, sharded with the index at the end
  (``bytes`` + optional ``crc32c`` index codecs), inner codecs ``bytes`` and
  optionally ``zstd``, the default chunk key encoding: what ``zvdvc convert``
  and ``zvdvc synth`` write.

nvCOMP trusts its input: a malformed zstd chunk can hang its kernel or fail the
CUDA context where the host decoder raises. So each shard index's crc32c is
checked, reads must return every byte asked for, and frames go through
zarr-vectors' checked ``decode_zstd`` when it is available (otherwise each frame
header must declare the chunk's size). Damage inside a zstd block passes all of
these; use the device path on stores you trust.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.config import GpuIo
from zvdvc.geometry.box import Box

GPU_IO_CHOICES: tuple[str, ...] = GpuIo.__args__
ENV = "ZVDVC_GPU_IO"
SLAB_BYTES = 256 << 20          # device staging per read step: bounds the extra memory a brick read needs


IO_THREADS = 16                 # kvikio's thread pool when $KVIKIO_NTHREADS is unset (its default, 1, serialises reads)


@functools.lru_cache(maxsize=1)
def _kvikio() -> Any | None:
    try:
        import kvikio
        import kvikio.defaults
    except Exception:
        return None
    if "KVIKIO_NTHREADS" not in os.environ:
        kvikio.defaults.set("num_threads", min(IO_THREADS, os.cpu_count() or 1))
    return kvikio


def _nvcomp() -> Any | None:
    try:
        from nvidia import nvcomp
    except Exception:
        return None
    return nvcomp


@functools.lru_cache(maxsize=1)
def status() -> dict[str, Any]:
    """What this machine offers: kvikio and nvCOMP versions, and whether cuFile has GPUDirect Storage."""
    kv, nc = _kvikio(), _nvcomp()
    out: dict[str, Any] = {"kvikio": getattr(kv, "__version__", None), "nvcomp": getattr(nc, "__version__", None),
                           "gds_available": False, "compat_mode_preferred": None, "cufile": None}
    if kv is None:
        out["why"] = "kvikio is not installed (conda install -c rapidsai kvikio)"
        return out
    out["compat_mode_preferred"] = bool(kv.defaults.is_compat_mode_preferred())
    try:
        import kvikio.cufile_driver as cd

        out["gds_available"] = bool(cd.properties.is_gds_available)
        out["cufile"] = ".".join(str(v) for v in cd.libcufile_version()) if hasattr(cd, "libcufile_version") else None
    except Exception as exc:                      # no libcufile: kvikio reads through its own POSIX path
        out["why"] = f"cuFile unavailable: {type(exc).__name__}: {exc}"
        return out
    if not out["gds_available"]:
        out["why"] = ("cuFile runs without GPUDirect Storage (no nvidia-fs driver, or an unsupported filesystem); "
                      "`gdscheck -p` shows why")
    return out


def resolve(mode: str = "auto") -> str:
    """The read path to use: ``"kvikio"`` or ``"host"``. ``auto`` defers to ``$ZVDVC_GPU_IO`` first."""
    if mode == "auto":
        mode = os.environ.get(ENV, "auto") or "auto"
    if mode not in GPU_IO_CHOICES:
        raise ValueError(f"gpu_io: {mode!r} is not one of {list(GPU_IO_CHOICES)} (volumes.gpu_io or ${ENV})")
    if mode == "host":
        return "host"
    st = status()
    if mode == "kvikio":
        if st["kvikio"] is None:
            raise ImportError(f"gpu_io 'kvikio' needs kvikio: {st['why']}")
        return "kvikio"
    return "kvikio" if st["gds_available"] else "host"


# --------------------------------------------------------------------------- readers


class DeviceReader:
    """Reads the part of a volume inside a box into a device array (native byte order)."""

    why_not: str | None = None

    def read_into(self, out: Any, valid: Box, at: tuple[int, int, int], stream: Any = None) -> None:
        """Write ``valid`` (volume coordinates) into ``out`` at offset ``at`` (z, y, x)."""
        raise NotImplementedError


def _as_native(raw: Any, dtype: np.dtype) -> Any:
    """A 1-D device byte buffer holding ``dtype`` values (either byte order), as native values."""
    import cupy as cp

    if dtype.isnative or dtype.itemsize == 1:
        return raw.view(dtype.newbyteorder("="))
    swapped = cp.ascontiguousarray(raw.reshape(-1, dtype.itemsize)[:, ::-1]).reshape(-1)
    return swapped.view(dtype.newbyteorder("="))


class _null:
    """A no-op context, standing in for a CUDA stream when none is given."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


class RawDeviceReader(DeviceReader):
    """A flat (z, y, x) C-ordered file: whole rows of each plane are read, then cropped on the device.

    A brick spanning whole planes is one contiguous read; otherwise each plane's rows are one read.
    Reading whole rows costs ``X / (x1 - x0)`` in bytes, which is small for bricks and keeps the
    number of requests at one per plane.
    """

    def __init__(self, path: str | Path, *, header_bytes: int, dtype: np.dtype, shape_zyx: tuple[int, int, int]) -> None:
        self.path, self.header = str(path), int(header_bytes)
        self.dtype, self.shape = np.dtype(dtype), tuple(int(n) for n in shape_zyx)

    def read_into(self, out: Any, valid: Box, at: tuple[int, int, int], stream: Any = None) -> None:
        import cupy as cp

        kvikio = _kvikio()
        Z, Y, X = self.shape
        (z0, y0, x0), (z1, y1, x1) = valid.lo, valid.hi
        isz = self.dtype.itemsize
        plane_rows = (y1 - y0) * X * isz
        whole_planes = y0 == 0 and y1 == Y
        per_slab = max(1, SLAB_BYTES // max(plane_rows, 1))
        with kvikio.CuFile(self.path, "r") as fh:
            for zs in range(z0, z1, per_slab):
                ze = min(z1, zs + per_slab)
                n = ze - zs
                raw = cp.empty(n * plane_rows, dtype=cp.uint8)
                if whole_planes:
                    fh.pread(raw, raw.nbytes, self.header + zs * Y * X * isz).get()
                else:
                    futures = [fh.pread(raw[k * plane_rows:(k + 1) * plane_rows], plane_rows,
                                        self.header + ((zs + k) * Y + y0) * X * isz) for k in range(n)]
                    for f in futures:
                        f.get()
                with stream if stream is not None else _null():
                    part = _as_native(raw, self.dtype).reshape(n, y1 - y0, X)[:, :, x0:x1]
                    dz = at[0] + (zs - z0)
                    out[dz:dz + n, at[1]:at[1] + (y1 - y0), at[2]:at[2] + (x1 - x0)] = part
                del raw


class ShardedZarrReader(DeviceReader):
    """A sharded Zarr v3 array on a local store: the needed inner chunks' stored bytes are read with
    kvikio and decompressed by nvCOMP in batches, then copied into place on the device."""

    MISSING = np.uint64(2**64 - 1)

    def __init__(self, array: Any) -> None:
        from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
        from zarr.storage import LocalStore

        meta = array.metadata
        self.why_not = None
        if getattr(meta, "zarr_format", None) != 3:
            self.why_not = "not a Zarr v3 array"
            return
        if not isinstance(array.store, LocalStore):
            self.why_not = f"store {type(array.store).__name__} is not a local directory"
            return
        codecs = tuple(meta.codecs)
        if len(codecs) != 1 or not isinstance(codecs[0], ShardingCodec):
            self.why_not = "not sharded (one sharding codec expected)"
            return
        sh = codecs[0]
        if str(getattr(sh.index_location, "value", sh.index_location)) != "end":
            self.why_not = "shard index is not at the end"
            return
        inner = tuple(sh.codecs)
        if not inner or not isinstance(inner[0], BytesCodec) or any(not isinstance(c, ZstdCodec) for c in inner[1:]) \
                or len(inner) > 2:
            self.why_not = f"inner codecs {[type(c).__name__ for c in inner]} (bytes [+ zstd] expected)"
            return
        idx = tuple(sh.index_codecs)
        if not idx or not isinstance(idx[0], BytesCodec) or any(not isinstance(c, Crc32cCodec) for c in idx[1:]):
            self.why_not = "shard index codecs are not bytes [+ crc32c]"
            return
        self.zstd = len(inner) == 2
        if self.zstd and _nvcomp() is None:
            self.why_not = "zstd chunks need nvCOMP on the GPU (pip install nvidia-nvcomp-cu12)"
            return
        endian = getattr(inner[0].endian, "value", inner[0].endian)
        self.dtype = np.dtype(array.dtype).newbyteorder("<" if endian == "little" else ">")
        self.inner = tuple(int(n) for n in sh.chunk_shape)
        self.shard = tuple(int(n) for n in meta.chunk_grid.chunk_shape)
        self.per_shard = tuple(s // c for s, c in zip(self.shard, self.inner))
        self.n_inner = int(np.prod(self.per_shard))
        self.index_crc = len(idx) == 2
        self.index_nbytes = self.n_inner * 16 + 4 * self.index_crc
        self.shape = tuple(int(n) for n in array.shape)
        self.fill = array.fill_value
        self.root = Path(str(array.store.root)) / array.path
        self.key = meta.chunk_key_encoding
        self._index: dict[tuple[int, ...], np.ndarray] = {}

    def _shard_path(self, shard: tuple[int, ...]) -> Path:
        return self.root / self.key.encode_chunk_key(shard)

    def _shard_index(self, shard: tuple[int, ...]) -> np.ndarray | None:
        """(n_inner, 2) offsets and sizes, or None when the shard file does not exist (all fill)."""
        if shard not in self._index:
            path = self._shard_path(shard)
            if not path.exists():
                self._index[shard] = None           # type: ignore[assignment]
            else:
                with open(path, "rb") as fh:
                    fh.seek(-self.index_nbytes, os.SEEK_END)
                    raw = fh.read(self.index_nbytes)
                body = raw[: self.n_inner * 16]
                if self.index_crc:
                    import google_crc32c

                    if np.uint32(google_crc32c.value(body)).tobytes() != raw[-4:]:
                        raise ValueError(f"{path}: the shard index fails its crc32c check (a damaged or truncated shard)")
                index = np.frombuffer(body, dtype="<u8").reshape(self.n_inner, 2).copy()
                size = path.stat().st_size - self.index_nbytes
                present = ~((index[:, 0] == self.MISSING) & (index[:, 1] == self.MISSING))
                if present.any() and int((index[present, 0] + index[present, 1]).max()) > size:
                    raise ValueError(f"{path}: the shard index points past the end of its data (a truncated shard)")
                self._index[shard] = index
        return self._index[shard]

    def read_into(self, out: Any, valid: Box, at: tuple[int, int, int], stream: Any = None) -> None:
        import cupy as cp

        kvikio = _kvikio()
        lo, hi = valid.lo, valid.hi
        c_lo = [lo[a] // self.inner[a] for a in range(3)]
        c_hi = [(hi[a] - 1) // self.inner[a] + 1 for a in range(3)]
        chunk_nbytes = int(np.prod(self.inner)) * self.dtype.itemsize
        # (shard path, file offset, stored bytes, destination and source slices) per present chunk
        jobs: list[tuple[str, int, int, tuple[slice, ...], tuple[slice, ...]]] = []
        with stream if stream is not None else _null():
            for cz in range(c_lo[0], c_hi[0]):
                for cy in range(c_lo[1], c_hi[1]):
                    for cx in range(c_lo[2], c_hi[2]):
                        c = (cz, cy, cx)
                        c0 = [c[a] * self.inner[a] for a in range(3)]
                        s0 = [max(lo[a], c0[a]) for a in range(3)]
                        s1 = [min(hi[a], c0[a] + self.inner[a]) for a in range(3)]
                        dst = tuple(slice(at[a] + s0[a] - lo[a], at[a] + s1[a] - lo[a]) for a in range(3))
                        src = tuple(slice(s0[a] - c0[a], s1[a] - c0[a]) for a in range(3))
                        shard = tuple(c[a] // self.per_shard[a] for a in range(3))
                        within = tuple(c[a] % self.per_shard[a] for a in range(3))
                        index = self._shard_index(shard)
                        entry = None if index is None else index[int(np.ravel_multi_index(within, self.per_shard))]
                        if entry is None or (entry[0] == self.MISSING and entry[1] == self.MISSING):
                            out[dst] = self.fill              # a chunk that was never written reads as the fill value
                            continue
                        jobs.append((str(self._shard_path(shard)), int(entry[0]), int(entry[1]), dst, src))
            # decode in groups whose decoded size stays within SLAB_BYTES
            group = max(1, SLAB_BYTES // chunk_nbytes)
            for g in range(0, len(jobs), group):
                self._read_group(jobs[g:g + group], out, chunk_nbytes, stream, cp, kvikio)

    def _read_group(self, jobs: list, out: Any, chunk_nbytes: int, stream: Any, cp: Any, kvikio: Any) -> None:
        total = sum(j[2] for j in jobs)
        stored = cp.empty(total, dtype=cp.uint8)
        views, pos, handles, futures = [], 0, {}, []
        try:
            for path, offset, nbytes, _, _ in jobs:
                fh = handles.get(path)
                if fh is None:
                    fh = handles[path] = kvikio.CuFile(path, "r")
                view = stored[pos:pos + nbytes]
                futures.append(fh.pread(view, nbytes, offset))
                views.append(view)
                pos += nbytes
            for f, (path, _, nbytes, _, _) in zip(futures, jobs):
                if f.get() != nbytes:
                    raise ValueError(f"{path}: short read of a chunk ({nbytes} bytes expected): a truncated shard?")
        finally:
            for fh in handles.values():
                fh.close()
        if self.zstd:
            decoded = cp.empty((len(jobs), chunk_nbytes), dtype=cp.uint8)
            outs = [decoded[i] for i in range(len(jobs))]
            _decode_zstd(views, outs, chunk_nbytes, stream, [j[0] for j in jobs])
            chunks = outs
        else:
            chunks = views
        for (_, _, nbytes, dst, src), raw in zip(jobs, chunks):
            if raw.nbytes != chunk_nbytes:
                raise ValueError(f"a chunk decoded to {raw.nbytes} bytes, not {chunk_nbytes}: not the layout expected")
            values = _as_native(raw, self.dtype).reshape(self.inner)
            out[dst] = values[src]


_ZSTD_MAGIC = 0xFD2FB528


def zstd_content_size(header: bytes) -> int | None:
    """The decompressed size a zstd frame header declares, or None when it declares none.

    Raises ``ValueError`` when ``header`` does not start a zstd frame.
    """
    if len(header) < 5 or int.from_bytes(header[:4], "little") != _ZSTD_MAGIC:
        raise ValueError("not a zstd frame")
    fhd = header[4]
    fcs_flag, single, dict_flag = fhd >> 6, (fhd >> 5) & 1, fhd & 3
    pos = 5 + (0 if single else 1) + (0, 1, 2, 4)[dict_flag]
    n = (1 if single else 0, 2, 4, 8)[fcs_flag]
    if n == 0:
        return None
    if len(header) < pos + n:
        raise ValueError("truncated zstd frame header")
    value = int.from_bytes(header[pos:pos + n], "little")
    return value + 256 if n == 2 else value


def _decode_zstd(frames: list[Any], outs: list[Any], nbytes: int, stream: Any, where: list[str]) -> None:
    """Decompress zstd ``frames`` (device) into ``outs`` (device, ``nbytes`` each), refusing bad input.

    nvCOMP trusts its input: a truncated or malformed frame can hang its kernel or kill the CUDA
    context. zarr-vectors' ``decode_zstd`` checks every frame's structure on the device first and
    reports per-frame errors; it is used when present (``"decode-zstd" in zarr_vectors.FEATURES``).
    Otherwise each frame header must declare exactly ``nbytes`` before nvCOMP sees it; damage inside
    a block is caught by neither.
    """
    import cupy as cp

    try:
        import zarr_vectors as zv

        checked = "decode-zstd" in getattr(zv, "FEATURES", ())
    except Exception:
        checked = False
    if checked:
        from zarr_vectors.gpu import decode_zstd

        _, errors = decode_zstd(frames, nbytes, out=outs, stream=stream)
        if errors:
            i, why = next(iter(sorted(errors.items())))
            raise ValueError(f"{where[i]}: {len(errors)} chunk(s) failed to decode, e.g. {why}")
        return
    heads = cp.asnumpy(cp.concatenate([f[:18] for f in frames])) if frames else np.empty(0, np.uint8)
    pos = 0
    for i, f in enumerate(frames):
        k = min(18, int(f.size))
        try:
            size = zstd_content_size(heads[pos:pos + k].tobytes())
        except ValueError as exc:
            raise ValueError(f"{where[i]}: a chunk is not a valid zstd frame ({exc})") from None
        if size is not None and size != nbytes:
            raise ValueError(f"{where[i]}: a chunk's zstd frame declares {size} bytes, not {nbytes}")
        pos += k
    nvcomp = _nvcomp()
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW,
                         cuda_stream=(stream.ptr if stream is not None else cp.cuda.get_current_stream().ptr))
    codec.decode(nvcomp.as_arrays(frames), out=outs)
