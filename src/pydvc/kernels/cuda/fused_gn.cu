// pyDVC fused Gauss-Newton kernels.
//
// Compiled by NVRTC through cupy.RawModule (pydvc/kernels/fused.py), one
// specialisation per (DOF, objective, interpolation, brick type), or on the
// host by pydvc/kernels/cuda/emulate.py, which prepends a CUDA shim so the
// tests exercise this exact source without a GPU. Hence: no #includes, and
// every kernel must be correct for any blockDim that is a multiple of 32.
//
// Kernels
//   sample_values  one thread per (point, sample): values and an inside flag
//   gn_sums        one block per active point: the one-pass normal-equation
//                  sums (layout in pydvc.kernels.objective.sum_layout)
//   gn_solve       one thread per active point: exact normal equations from
//                  the sums, Jacobi-scaled Cholesky, step, stopping tests
//
// Coordinates: a point's brick-local centre is split into an integer voxel
// (cint) and a fraction (cfrac); sample positions are formed as
// cfrac + t + F d in float32 and only then added to cint, so precision does
// not degrade with the coordinate's magnitude.

namespace pydvc {

enum Objective { SAD = 0, SSD = 1, ZSSD = 2, NSSD = 3, ZNSSD = 4 };
enum Interp { NEAREST = 0, TRILINEAR = 1, TRICUBIC = 2 };
enum Status { ST_GOOD = 0, ST_RANGE_FAIL = -1, ST_SINGULAR = -5 };

template <int DOF>
struct Layout {
    static constexpr int NJJ = DOF * (DOF + 1) / 2;
    static constexpr int J = NJJ;
    static constexpr int VJ = NJJ + DOF;
    static constexpr int QJ = NJJ + 2 * DOF;
    static constexpr int S = NJJ + 3 * DOF;      // v, vv, vq, q, qq, n, absdiff
    static constexpr int NSUM = S + 7;
    static constexpr int NA = DOF > 3 ? DOF - 3 : 1;
};

// geom = [nx, ny, nz, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]: brick shape and the
// half-open valid region, both brick-local.
struct Geom {
    int n[3];
    int lo[3];
    int hi[3];
};

__device__ __forceinline__ Geom load_geom(const int* __restrict__ g) {
    Geom r;
    for (int a = 0; a < 3; ++a) {
        r.n[a] = g[a];
        r.lo[a] = g[3 + a];
        r.hi[a] = g[6 + a];
    }
    return r;
}

template <typename T>
__device__ __forceinline__ float ld(const T* __restrict__ p, long long i) {
    return (float)__ldg(p + i);
}

__device__ __forceinline__ void catmull_rom(float t, float* w, float* dw) {
    const float t2 = t * t, t3 = t2 * t;
    w[0] = 0.5f * (-t3 + 2.0f * t2 - t);
    w[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
    w[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
    w[3] = 0.5f * (t3 - t2);
    dw[0] = 0.5f * (-3.0f * t2 + 4.0f * t - 1.0f);
    dw[1] = 0.5f * (9.0f * t2 - 10.0f * t);
    dw[2] = 0.5f * (-9.0f * t2 + 8.0f * t + 1.0f);
    dw[3] = 0.5f * (3.0f * t2 - 2.0f * t);
}

// Interpolate at voxel i (floor of the position, per axis) plus fraction f.
// Returns false, leaving the outputs untouched, if the stencil leaves the
// valid region.
template <int INTERP, bool GRAD, typename T>
__device__ __forceinline__ bool interpolate(const T* __restrict__ brick, const Geom& g, const int* i, const float* f,
                                            float& val, float& gx, float& gy, float& gz) {
    const int before = INTERP == TRICUBIC ? 1 : 0;
    const int after = INTERP == TRICUBIC ? 2 : (INTERP == TRILINEAR ? 1 : 0);
    int c[3];
    for (int a = 0; a < 3; ++a) {
        c[a] = (INTERP == NEAREST && f[a] >= 0.5f) ? i[a] + 1 : i[a];
        const int lo = g.lo[a] > 0 ? g.lo[a] : 0;
        const int hi = g.hi[a] < g.n[a] ? g.hi[a] : g.n[a];
        if (c[a] - before < lo || c[a] + after > hi - 1) return false;
    }
    const long long nx = g.n[0], ny = g.n[1];
    if (INTERP == NEAREST) {
        val = ld(brick, ((long long)c[2] * ny + c[1]) * nx + c[0]);
        gx = gy = gz = 0.0f;
        return true;
    }
    if (INTERP == TRILINEAR) {
        const float wx[2] = {1.0f - f[0], f[0]}, wy[2] = {1.0f - f[1], f[1]}, wz[2] = {1.0f - f[2], f[2]};
        const float dw[2] = {-1.0f, 1.0f};
        float v = 0.0f, ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int zz = 0; zz < 2; ++zz) {
            for (int yy = 0; yy < 2; ++yy) {
                const long long row = ((long long)(c[2] + zz) * ny + (c[1] + yy)) * nx + c[0];
                const float t0 = ld(brick, row), t1 = ld(brick, row + 1);
                const float sx = wx[0] * t0 + wx[1] * t1;
                v += wz[zz] * wy[yy] * sx;
                if (GRAD) {
                    ax += wz[zz] * wy[yy] * (t1 - t0);
                    ay += wz[zz] * dw[yy] * sx;
                    az += dw[zz] * wy[yy] * sx;
                }
            }
        }
        val = v;
        gx = ax;
        gy = ay;
        gz = az;
        return true;
    }
    float wx[4], dwx[4], wy[4], dwy[4], wz[4], dwz[4];
    catmull_rom(f[0], wx, dwx);
    catmull_rom(f[1], wy, dwy);
    catmull_rom(f[2], wz, dwz);
    const long long base = ((long long)(c[2] - 1) * ny + (c[1] - 1)) * nx + (c[0] - 1);
    float v = 0.0f, ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int zz = 0; zz < 4; ++zz) {
        float vz = 0.0f, vzdx = 0.0f, vzdy = 0.0f;
        for (int yy = 0; yy < 4; ++yy) {
            const long long row = base + ((long long)zz * ny + yy) * nx;
            const float t0 = ld(brick, row), t1 = ld(brick, row + 1), t2 = ld(brick, row + 2), t3 = ld(brick, row + 3);
            const float sx = wx[0] * t0 + wx[1] * t1 + wx[2] * t2 + wx[3] * t3;
            vz += wy[yy] * sx;
            if (GRAD) {
                vzdx += wy[yy] * (dwx[0] * t0 + dwx[1] * t1 + dwx[2] * t2 + dwx[3] * t3);
                vzdy += dwy[yy] * sx;
            }
        }
        v += wz[zz] * vz;
        if (GRAD) {
            ax += wz[zz] * vzdx;
            ay += wz[zz] * vzdy;
            az += dwz[zz] * vz;
        }
    }
    val = v;
    gx = ax;
    gy = ay;
    gz = az;
    return true;
}

__device__ __forceinline__ void mat3(const float* a, const float* b, float* c) {
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
            c[3 * r + k] = a[3 * r] * b[k] + a[3 * r + 1] * b[3 + k] + a[3 * r + 2] * b[6 + k];
}

// F = (I + E) R and the matrices A_k with dx'/dp_k = A_k d for k >= 3, in
// CCPi's parameterisation (pydvc.geometry.warp): R = Rx(psi) Ry(the) Rz(phi).
template <int DOF>
__device__ void warp_matrices(const float* __restrict__ p, float* F, float* A) {
    for (int k = 0; k < 9; ++k) F[k] = (k % 4 == 0) ? 1.0f : 0.0f;
    if (DOF == 3) return;
    const float cf = cosf(p[3]), sf = sinf(p[3]), ct = cosf(p[4]), st = sinf(p[4]), cs = cosf(p[5]), ss = sinf(p[5]);
    const float rz[9] = {cf, sf, 0.0f, -sf, cf, 0.0f, 0.0f, 0.0f, 1.0f};
    const float drz[9] = {-sf, cf, 0.0f, -cf, -sf, 0.0f, 0.0f, 0.0f, 0.0f};
    const float ry[9] = {ct, 0.0f, -st, 0.0f, 1.0f, 0.0f, st, 0.0f, ct};
    const float dry[9] = {-st, 0.0f, -ct, 0.0f, 0.0f, 0.0f, ct, 0.0f, -st};
    const float rx[9] = {1.0f, 0.0f, 0.0f, 0.0f, cs, ss, 0.0f, -ss, cs};
    const float drx[9] = {0.0f, 0.0f, 0.0f, 0.0f, -ss, cs, 0.0f, -cs, -ss};
    float t[9], R[9], dR[3][9];
    mat3(ry, rz, t);
    mat3(rx, t, R);
    mat3(ry, drz, t);
    mat3(rx, t, dR[0]);                      // d/dphi
    mat3(dry, rz, t);
    mat3(rx, t, dR[1]);                      // d/dthe
    mat3(ry, rz, t);
    mat3(drx, t, dR[2]);                     // d/dpsi
    if (DOF == 6) {
        for (int k = 0; k < 9; ++k) F[k] = R[k];
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 9; ++k) A[9 * j + k] = dR[j][k];
        return;
    }
    const float exx = p[6], eyy = p[7], ezz = p[8], exy = p[9], eyz = p[10], exz = p[11];
    const float IE[9] = {1.0f + exx, exy, exz, exy, 1.0f + eyy, eyz, exz, eyz, 1.0f + ezz};
    mat3(IE, R, F);
    for (int j = 0; j < 3; ++j) mat3(IE, dR[j], A + 9 * j);
    // strain derivatives: dE_k R, where dE_k selects rows of R
    const int rows_a[6] = {0, 1, 2, 0, 1, 0}, rows_b[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 0; k < 6; ++k) {
        float* Ak = A + 9 * (3 + k);
        for (int m = 0; m < 9; ++m) Ak[m] = 0.0f;
        const int a = rows_a[k], b = rows_b[k];
        for (int m = 0; m < 3; ++m) {
            Ak[3 * a + m] += R[3 * b + m];
            if (a != b) Ak[3 * b + m] += R[3 * a + m];
        }
    }
}

// --------------------------------------------------------------------- sample_values

template <int DOF, int INTERP, typename T>
__global__ void sample_values(const T* __restrict__ brick, const int* __restrict__ geom, const int* __restrict__ cint,
                              const float* __restrict__ cfrac, const float* __restrict__ params,
                              const float* __restrict__ offsets, int B, int M, float* __restrict__ values,
                              unsigned char* __restrict__ inside) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * M) return;
    const int b = (int)(idx / M), m = (int)(idx % M);
    const Geom g = load_geom(geom);
    float F[9], A[9 * Layout<DOF>::NA];
    warp_matrices<DOF>(params + (long long)b * DOF, F, A);
    const float* p = params + (long long)b * DOF;
    const float dx = offsets[3 * m], dy = offsets[3 * m + 1], dz = offsets[3 * m + 2];
    int i[3];
    float f[3];
    for (int a = 0; a < 3; ++a) {
        const float s = cfrac[3 * b + a] + p[a] + F[3 * a] * dx + F[3 * a + 1] * dy + F[3 * a + 2] * dz;
        const float fl = floorf(s);
        i[a] = cint[3 * b + a] + (int)fl;
        f[a] = s - fl;
    }
    float v = 0.0f, gx, gy, gz;
    const bool ok = interpolate<INTERP, false, T>(brick, g, i, f, v, gx, gy, gz);
    values[idx] = ok ? v : 0.0f;
    inside[idx] = ok ? 1 : 0;
}

// --------------------------------------------------------------------- gn_sums

// Reduce acc across the block into res (shared); valid after the final barrier.
template <int N>
__device__ __forceinline__ void block_reduce(float* acc, float* res) {
    __shared__ float red[8][N];                   // blockDim <= 256
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = (blockDim.x + 31) >> 5;
    for (int k = 0; k < N; ++k) {
        float v = acc[k];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        float s = 0.0f;
        for (int w = 0; w < nwarp; ++w) s += red[w][k];
        res[k] = s;
    }
    __syncthreads();
}

template <int DOF, int OBJ, int INTERP, typename T>
__global__ void gn_sums(const T* __restrict__ brick, const int* __restrict__ geom, const int* __restrict__ cint,
                        const float* __restrict__ cfrac, const float* __restrict__ params,
                        const int* __restrict__ active, int n_active, const float* __restrict__ q,
                        const float* __restrict__ shift, const float* __restrict__ offsets, int M,
                        float* __restrict__ sums, int* __restrict__ outside) {
    typedef Layout<DOF> L;
    const int a = blockIdx.x;
    if (a >= n_active) return;
    const int pt = active[a];
    __shared__ float mats[9 + 9 * L::NA];
    __shared__ float res[L::NSUM + 1];
    if (threadIdx.x == 0) warp_matrices<DOF>(params + (long long)pt * DOF, mats, mats + 9);
    __syncthreads();
    const Geom g = load_geom(geom);
    const float* p = params + (long long)pt * DOF;
    const float* F = mats;
    const float* A = mats + 9;
    const float s0 = shift[pt];
    float acc[L::NSUM + 1];
    for (int k = 0; k <= L::NSUM; ++k) acc[k] = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        const float dx = offsets[3 * m], dy = offsets[3 * m + 1], dz = offsets[3 * m + 2];
        int i[3];
        float f[3];
        for (int ax = 0; ax < 3; ++ax) {
            const float s = cfrac[3 * pt + ax] + p[ax] + F[3 * ax] * dx + F[3 * ax + 1] * dy + F[3 * ax + 2] * dz;
            const float fl = floorf(s);
            i[ax] = cint[3 * pt + ax] + (int)fl;
            f[ax] = s - fl;
        }
        float val, gx, gy, gz;
        if (!interpolate<INTERP, true, T>(brick, g, i, f, val, gx, gy, gz)) {
            acc[L::NSUM] += 1.0f;
            continue;
        }
        float J[DOF];
        J[0] = gx;
        J[1] = gy;
        J[2] = gz;
        for (int k = 0; k < DOF - 3; ++k) {
            const float* Ak = A + 9 * k;
            J[3 + k] = gx * (Ak[0] * dx + Ak[1] * dy + Ak[2] * dz) + gy * (Ak[3] * dx + Ak[4] * dy + Ak[5] * dz) +
                       gz * (Ak[6] * dx + Ak[7] * dy + Ak[8] * dz);
        }
        const float v = val - s0;
        const float qq = q[(long long)pt * M + m];
        int t = 0;
        for (int r = 0; r < DOF; ++r)
            for (int c = r; c < DOF; ++c) acc[t++] += J[r] * J[c];
        for (int r = 0; r < DOF; ++r) {
            acc[L::J + r] += J[r];
            acc[L::VJ + r] += v * J[r];
            acc[L::QJ + r] += qq * J[r];
        }
        acc[L::S + 0] += v;
        acc[L::S + 1] += v * v;
        acc[L::S + 2] += v * qq;
        acc[L::S + 3] += qq;
        acc[L::S + 4] += qq * qq;
        acc[L::S + 5] += 1.0f;
        acc[L::S + 6] += fabsf(v - qq);
    }
    block_reduce<L::NSUM + 1>(acc, res);
    for (int k = threadIdx.x; k < L::NSUM; k += blockDim.x) sums[(long long)a * L::NSUM + k] = res[k];
    if (threadIdx.x == 0) outside[a] = (int)res[L::NSUM];
}

// --------------------------------------------------------------------- gn_solve

template <int DOF, int OBJ>
__device__ void normal_from_sums(const float* S, float s0, float* H, float* b, float& obj, float& grad_energy,
                                 float& sum_g2) {
    typedef Layout<DOF> L;
    int t = 0;
    for (int r = 0; r < DOF; ++r)
        for (int c = r; c < DOF; ++c) {
            H[r * DOF + c] = S[t];
            H[c * DOF + r] = S[t];
            ++t;
        }
    const float* SJ = S + L::J;
    const float* SvJ = S + L::VJ;
    const float* SqJ = S + L::QJ;
    const float Sv = S[L::S], Svv = S[L::S + 1], Svq = S[L::S + 2], Sq = S[L::S + 3], Sqq = S[L::S + 4];
    const float n = S[L::S + 5], Sabs = S[L::S + 6];
    const float nn = n > 1.0f ? n : 1.0f;
    if (OBJ == SAD || OBJ == SSD) {
        for (int r = 0; r < DOF; ++r) b[r] = SvJ[r] - SqJ[r];
        obj = OBJ == SAD ? Sabs : Svv - 2.0f * Svq + Sqq;
    } else if (OBJ == ZSSD) {
        const float vbar = Sv / nn;
        for (int r = 0; r < DOF; ++r) {
            const float mr = SJ[r] / nn;
            for (int c = 0; c < DOF; ++c) H[r * DOF + c] -= n * mr * (SJ[c] / nn);
            b[r] = SvJ[r] - vbar * SJ[r] - SqJ[r] + mr * Sq;
        }
        obj = (Svv - n * vbar * vbar) - 2.0f * (Svq - vbar * Sq) + Sqq;
    } else {
        const float vbar = OBJ == NSSD ? 0.0f : Sv / nn;
        const float norm2 = OBJ == NSSD ? Svv : Svv - n * vbar * vbar;
        const float norm = sqrtf(norm2 > 0.0f ? norm2 : 0.0f);
        const float inv = norm > 0.0f ? 1.0f / norm : 1.0f;
        float a[DOF];
        for (int r = 0; r < DOF; ++r) a[r] = (SvJ[r] - vbar * SJ[r]) * inv;
        if (OBJ == ZNSSD)
            for (int r = 0; r < DOF; ++r)
                for (int c = 0; c < DOF; ++c) H[r * DOF + c] -= n * (SJ[r] / nn) * (SJ[c] / nn);
        const float ghat_q = (Svq - vbar * Sq) * inv;
        for (int r = 0; r < DOF; ++r) {
            for (int c = 0; c < DOF; ++c) H[r * DOF + c] = (H[r * DOF + c] - a[r] * a[c]) * inv * inv;
            const float jr = a[r] - SqJ[r] + (OBJ == ZNSSD ? (SJ[r] / nn) * Sq : 0.0f);
            b[r] = (jr - a[r] * (1.0f - ghat_q)) * inv;
        }
        obj = (OBJ == NSSD ? 0.5f : 0.25f) * (1.0f - 2.0f * ghat_q + Sqq);
    }
    grad_energy = S[0] + S[DOF] + S[2 * DOF - 1];
    sum_g2 = Svv + 2.0f * s0 * Sv + n * s0 * s0;
}

template <int DOF, int OBJ>
__global__ void gn_solve(const float* __restrict__ sums, const int* __restrict__ outside,
                         const int* __restrict__ active, int n_active, const float* __restrict__ shift,
                         const float* __restrict__ seeds, float* __restrict__ params, float* __restrict__ prev_obj,
                         float* __restrict__ objmin, unsigned char* __restrict__ n_iter,
                         signed char* __restrict__ status, int* __restrict__ done, float obj_tol, float disp_tol,
                         float disp_max, float pivot_tol, float featureless) {
    typedef Layout<DOF> L;
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_active) return;
    const int pt = active[a];
    if (outside[a] > 0) {
        status[pt] = ST_RANGE_FAIL;
        done[a] = 1;
        return;
    }
    float H[DOF * DOF], b[DOF], obj, ge, g2;
    normal_from_sums<DOF, OBJ>(sums + (long long)a * L::NSUM, shift[pt], H, b, obj, ge, g2);
    objmin[pt] = obj;
    bool singular = !(ge > featureless * g2);
    // Jacobi scaling, then Cholesky of the unit-diagonal system
    float sc[DOF];
    for (int r = 0; r < DOF; ++r) {
        const float d = H[r * DOF + r];
        singular = singular || !(d > 0.0f);
        sc[r] = d > 0.0f ? 1.0f / sqrtf(d) : 1.0f;
    }
    float Lm[DOF * DOF], y[DOF], x[DOF];
    for (int k = 0; k < DOF * DOF; ++k) Lm[k] = 0.0f;
    for (int j = 0; j < DOF && !singular; ++j) {
        float s = H[j * DOF + j] * sc[j] * sc[j];
        for (int k = 0; k < j; ++k) s -= Lm[j * DOF + k] * Lm[j * DOF + k];
        if (!(s > pivot_tol)) {
            singular = true;
            break;
        }
        const float d = sqrtf(s);
        Lm[j * DOF + j] = d;
        for (int i = j + 1; i < DOF; ++i) {
            float t = H[i * DOF + j] * sc[i] * sc[j];
            for (int k = 0; k < j; ++k) t -= Lm[i * DOF + k] * Lm[j * DOF + k];
            Lm[i * DOF + j] = t / d;
        }
    }
    if (singular) {
        status[pt] = ST_SINGULAR;
        done[a] = 1;
        return;
    }
    for (int i = 0; i < DOF; ++i) {
        float t = -b[i] * sc[i];
        for (int k = 0; k < i; ++k) t -= Lm[i * DOF + k] * y[k];
        y[i] = t / Lm[i * DOF + i];
    }
    for (int i = DOF - 1; i >= 0; --i) {
        float t = y[i];
        for (int k = i + 1; k < DOF; ++k) t -= Lm[k * DOF + i] * x[k];
        x[i] = t / Lm[i * DOF + i];
    }
    float* p = params + (long long)pt * DOF;
    for (int r = 0; r < DOF; ++r) p[r] += x[r] * sc[r];
    n_iter[pt] += 1;
    const float dt0 = x[0] * sc[0], dt1 = x[1] * sc[1], dt2 = x[2] * sc[2];
    const bool conv = fabsf(obj - prev_obj[pt]) < obj_tol || sqrtf(dt0 * dt0 + dt1 * dt1 + dt2 * dt2) < disp_tol;
    prev_obj[pt] = obj;
    float far = 0.0f;
    for (int r = 0; r < 3; ++r) far = fmaxf(far, fabsf(p[r] - seeds[3 * pt + r]));
    const bool out_of_range = far > disp_max;
    if (out_of_range) status[pt] = ST_RANGE_FAIL;
    done[a] = (conv || out_of_range) ? 1 : 0;
}

}  // namespace pydvc
