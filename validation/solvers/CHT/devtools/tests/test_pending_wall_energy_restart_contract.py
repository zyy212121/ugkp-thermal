from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/CHT')
COUPLER = ROOT / "thermal" / "GpuSolidThermalCoupler.C"
STATE = ROOT / "thermal" / "GpuThermalExchangeState.C"
HOST = ROOT / "gpu" / "GpuResidentStrict.H"
CUDA = ROOT / "gpu" / "GpuWallEnergy64.cu"


def test_pending_ledgers_are_touched_only_by_startup_and_write_persistence():
    coupler = COUPLER.read_text(encoding="utf-8")
    assert "restorePendingWallEnergyLedgers(restartState)" in coupler
    persist = coupler[coupler.index("label GpuSolidThermalCoupler::persistAtWriteTime") :]
    assert "pendingWallEnergySnapshot(completed)" in persist
    assert "writePendingWallEnergyTemporary(pendingWallEnergy)" in persist
    advance = coupler[coupler.index("GpuThermalCouplingResult GpuSolidThermalCoupler::exchangeIfDue") : coupler.index("label GpuSolidThermalCoupler::persistAtWriteTime")]
    assert "writePendingWallEnergyTemporary" not in advance


def test_checkpoint_uses_actual_state_and_pending_ledger_without_hash_admission():
    coupler = COUPLER.read_text(encoding="utf-8")
    state = STATE.read_text(encoding="utf-8")
    assert "completed.formatVersion = 4;" in coupler
    assert "writePendingWallEnergyTemporary" in coupler
    assert "readRestartState(runTime.timePath())" in state
    assert "pending wall-energy ledger count does not match manifest" not in state
    assert "ThermalRestartPreflight::fileSha1" not in coupler
    assert "restart is missing its required pending wall-energy ledger" in coupler


def test_gpu_transfer_is_compact_and_limited_to_coupled_patch_ranges():
    host = HOST.read_text(encoding="utf-8")
    cuda = CUDA.read_text(encoding="utf-8")
    assert "peekPendingWallEnergyLedgers" in host
    assert "restorePendingWallEnergyLedgers" in host
    assert "gasWallEnergyPatchSizes_" in host
    assert "ugkwpGpuResidentStrictPeekWallEnergyLedgerRange" in cuda
    assert "ugkwpGpuResidentStrictUploadWallEnergyLedgerRange" in cuda
    assert "s->gasWallEnergy + firstFace" in cuda
    assert "s->particleWallDepositedEnergy + firstFace" in cuda
    assert "s->particleWallReflectedEnergy + firstFace" in cuda


def test_restart_uses_actual_configured_face_addresses():
    coupler = COUPLER.read_text(encoding="utf-8")
    assert "pending.faceIds.size() == configuredFaces.size()" in coupler
    assert "pending.faceIds.begin()" in coupler
    assert "configuredFaces.begin()" in coupler
