pass
from validation.tools.paths import SOURCE_ROOT
import pytest


@pytest.mark.parametrize("block", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("workers", [1, 3, 4, 5, 11, 12, 13, 513])
def test_tile_ownership_is_exact(block, workers):
    per_block = block // 8
    grid = (workers + per_block - 1) // per_block
    visits = [0] * workers
    for cta in range(grid):
        for thread in range(block):
            worker = per_block * cta + thread // 8
            if worker < workers:
                visits[worker] += 1
    assert visits == [8] * workers


def test_production_tile_and_geometry_use_family_widths():
    source = (SOURCE_ROOT / "common/operators/evaluateGasBoundaryLayerKernel.cuh").read_text()
    assert "__launch_bounds__(32) void evaluateGasBoundaryLayerTile8Kernel" not in source
    assert "const int workersPerBlock=blockDim.x/8;" in source
    assert "const int block=ugkp::gpu::workPlan().faceBlockThreads;" in source
    assert "(workers+workersPerBlock-1)/workersPerBlock,block,0,stream" in source
    for prefix in ("", "precision/fp32/"):
        geometry = (SOURCE_ROOT / prefix / "common/thermal/ParticleGeometryAdapter.cuh").read_text()
        assert "refreshParticleTrackingGeometryKernel<<<(s->nCells+s->fixedCellBlockThreads-1)/s->fixedCellBlockThreads,s->fixedCellBlockThreads" in geometry
