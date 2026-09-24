from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
CASE_NAME = "MSS7_twoPhase_dense_transientContact"


def test_transient_contact_case_is_runnable_with_full_area_radiation() -> None:
    thermal = ROOT / "examples" / "thermal"
    particle_properties = (
        thermal / CASE_NAME / "constant" / "particleProperties"
    ).read_text(encoding="utf-8")
    allrun = (thermal / "Allrun").read_text(encoding="utf-8")
    prepare = (thermal / "mss7pre" / "prepare_run.sh").read_text(
        encoding="utf-8"
    )

    assert CASE_NAME in allrun
    assert CASE_NAME in prepare
    assert re.search(
        r"gpuResidentStuckWallPatches\s*\(\s*\)\s*;", particle_properties
    )
    stuck_model = particle_properties.split("gpuResidentStuckModel", 1)[1]
    assert re.search(r"\bheatTransfer\s+false\s*;", stuck_model)


def test_empty_nonthermal_stuck_configuration_disables_gpu_stuck_model() -> None:
    header = (ROOT / "applications" / "CHT" / "gpu" / "GpuResidentStrict.H").read_text(
        encoding="utf-8"
    )

    assert "if (candidatePatches.empty() && !heatTransfer)" in header
    disabled_branch = header.split(
        "if (candidatePatches.empty() && !heatTransfer)", 1
    )[1].split("ugkwpGpuResidentStrictConfigureParticleStuckModel", 1)[0]
    assert "particleWallHeatTransferEnabled_ = false;" in disabled_branch
    assert "particleStuckCandidateFaceMask_.transfer(candidateFaceMask);" in disabled_branch
    assert "return;" in disabled_branch
