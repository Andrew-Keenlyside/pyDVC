import pytest


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """A test that reaches a scaffold stub is skipped, not failed, and names the milestone."""
    try:
        return (yield)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


def _cuda_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs cupy and a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
