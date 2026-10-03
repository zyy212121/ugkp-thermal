from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def test_thermal_worker_routing_and_physics_are_preserved():
    for branch, folder in [("FSH", "private_backend"), ("CHT", "gpu")]:
        source = (ROOT / "applications" / branch / folder / "GpuResidentStrict.cu").read_text()
        for stage in ["Pool", "Moment"]:
            assert '#include "../../../gpu/thermal/CsrSegmented' + stage + 'Workers.cuh"' in source
            assert '#include "../../../gpu/CsrSegmented' + stage + 'Workers.cuh"' not in source
        assert "particleTemperatureFromSpecificEnthalpyDevice" in source
        assert "cub::DeviceScan::ExclusiveSum" in source
def test_thermal_workers_share_physics_and_queue_without_solver_policy():
    for stage in ("Pool", "Moment"):
        adapter = (ROOT / "gpu/thermal" / ("CsrSegmented" + stage + "Workers.cuh")).read_text()
        assert '#include "GpuSegmented' + stage + 'Workers.cuh"' in adapter
        code = (ROOT / 'common' / ('GpuSegmented' + stage + 'Workers.cuh')).read_text()
        assert '#include "CsrPersistentQueue.cuh"' in code
        assert "runCsrPersistentQueue" in code
        assert code.count('struct Csr' + stage + 'Operation') == 1
        assert 'directThermalPoolDispatch' not in code
    for branch, folder in [("FSH", "private_backend"), ("CHT", "gpu")]:
        source = (ROOT / "applications" / branch / folder / "GpuResidentStrict.cu").read_text()
        assert "directThermalPoolDispatch" not in source

