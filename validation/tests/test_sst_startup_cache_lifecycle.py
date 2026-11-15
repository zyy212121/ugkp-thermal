pass
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
BACKENDS = (
    "common/thermal/GpuResidentStrict.cu",
    "precision/fp32/common/thermal/GpuResidentStrict.cu",
    "applications/gasUGKP/private_backend/GpuResidentStrict.cu",
    "precision/fp32/applications/gasUGKP/private_backend/GpuResidentStrict.cu",
)

def test_workplan_publication_preserves_device_owned_state():
    source = (ROOT / "common/GpuWorkPlanPublish.cuh").read_text()
    assert "s->fixedWorkBlockTuned=1" in source
    assert "syncDeviceState(" not in source
    assert "cudaMemcpy" not in source

@pytest.mark.parametrize("backend", BACKENDS)
def test_sst_gpu_cache_is_mirrored_after_existing_completion(backend):
    source = (ROOT / backend).read_text()
    body = source.split('extern "C" int ugkwpGpuResidentStrictConfigureSst', 1)[1]
    body = body.split('\nextern "C"', 1)[0]
    kernel = body.index("initialiseSstConservativeStateKernel<<<")
    complete = body.index("cudaDeviceSynchronize()", kernel)
    copy = body.index("cudaMemcpyDeviceToHost", complete)
    assert kernel < complete < copy
    assert body[kernel:copy].count("cudaDeviceSynchronize()") == 1
    cache = body[complete:copy]
    assert "offsetof(DeviceState, sstJayatillekeP)" in cache
    assert "offsetof(DeviceState, sstThermalYPlus)" in cache
    assert "sizeof(s->sstThermalYPlus)" in cache
    assert "reinterpret_cast<unsigned char*>(s) + first" in cache
    assert "reinterpret_cast<const unsigned char*>(s->deviceState) + first" in cache
