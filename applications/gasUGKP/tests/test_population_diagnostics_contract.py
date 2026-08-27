from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "private_backend" / "GpuResidentStrict.cu").read_text()


def test_schema_three_records_population_clustering_and_launch_geometry():
    assert 'std::fprintf(file, "3,")' in CUDA
    for column in (
        "block_exponent",
        "block_threads",
        "sm_count",
        "particle_blocks_per_sm",
        "light_blocks_per_sm",
        "heavy_blocks_per_sm",
        "heavy_reduction_enabled",
        "heavy_threshold",
        "heavy_tile_particles",
        "pretransport_particle_count",
        "base_particle_count",
        "injected_particle_count",
        "removed_particle_count",
        "injection_fraction",
        "source_residual_mass",
        "occupancy_i2",
        "occupancy_imax",
        "heavy_cell_count",
        "heavy_particle_count",
        "heavy_task_count_estimate",
    ):
        assert column in CUDA


def test_injection_count_has_no_per_particle_diagnostic_atomic():
    assert "sourceInjectedCount[j] = created" in CUDA
    assert "atomicAdd(s.sourceInjectedCount" not in CUDA
    assert "++created" in CUDA


def test_pretransport_count_reuses_an_existing_cell_kernel():
    assert "diagnosticPreTransportParticleCount" in CUDA
    assert "*s.diagnosticPreTransportParticleCount" in CUDA
    assert "clearPoissonThermalPoolKernel" in CUDA


def test_occupancy_metrics_are_postprocessed_from_existing_counts():
    assert "sample.occupancyI2" in CUDA
    assert "sample.occupancyImax" in CUDA
    assert "sample.heavyCellCount" in CUDA
    assert "sample.heavyParticleCount" in CUDA
    assert "sample.heavyTaskCountEstimate" in CUDA


def test_skipped_stages_are_reported_as_exact_zero_not_event_overhead():
    assert "stageOccurrenceExecuted" in CUDA
    assert "!developmentProbe.stageOccurrenceExecuted[stage][occurrence]" in CUDA
    assert "UGKP_DEV_PROBE_LEAVE_IF(ProbeInjection, runInjection)" in CUDA
    assert "UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPre, runBinPre)" in CUDA
    assert "UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPost, runBinPost)" in CUDA
