from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "private_backend" / "GpuResidentStrict.cu").read_text()


def test_probe_accumulates_repeated_stage_intervals():
    assert "ProbeMaxOccurrences = 2" in CUDA
    assert "totalStartEvent" in CUDA
    assert "totalStopEvent" in CUDA
    assert "stageStartEvents[ProbeStageCount][ProbeMaxOccurrences]" in CUDA
    assert "stageStopEvents[ProbeStageCount][ProbeMaxOccurrences]" in CUDA
    assert "stageOccurrenceCount[ProbeStageCount]" in CUDA
    assert "sample.stageMs[stage] += elapsedMs" in CUDA
    assert "events[static_cast<int>(stage) + 1]" not in CUDA


def test_pressure_pre_is_intentionally_measured_twice():
    assert CUDA.count("UGKP_DEV_PROBE_ENTER(ProbePressurePre)") >= 2
    assert CUDA.count("UGKP_DEV_PROBE_LEAVE(ProbePressurePre)") >= 2
