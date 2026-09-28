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
    pool = (ROOT / "gpu/thermal/CsrSegmentedPoolWorkers.cuh").read_text()
    assert pool.count("void executeCsrSegmentedPoolTask") == 1
    assert pool.count("executeCsrSegmentedPoolTask<PoissonMode>") == 1
    for stage in ("Pool", "Moment"):
        code = (ROOT / "gpu/thermal" / ("CsrSegmented" + stage + "Workers.cuh")).read_text()
        if stage == 'Moment':
            assert '#include "GpuSegmentedMomentWorkers.cuh"' in code
            code = (ROOT / 'common/GpuSegmentedMomentWorkers.cuh').read_text()
        assert '#include "CsrPersistentQueue.cuh"' in code
        assert "runCsrPersistentQueue" in code
    for branch, folder in [("FSH", "private_backend"), ("CHT", "gpu")]:
        source = (ROOT / "applications" / branch / folder / "GpuResidentStrict.cu").read_text()
        assert "directThermalPoolDispatch" not in source

