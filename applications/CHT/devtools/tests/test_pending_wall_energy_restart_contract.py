from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COUPLER = ROOT / "thermal" / "GpuSolidThermalCoupler.C"
STATE = ROOT / "thermal" / "GpuThermalExchangeState.C"
HOST = ROOT / "gpu" / "GpuResidentStrict.H"
CUDA = ROOT / "gpu" / "GpuResidentStrict.cu"


def test_pending_ledgers_are_touched_only_by_startup_and_write_persistence():
    coupler = COUPLER.read_text(encoding="utf-8")
    assert "restorePendingWallEnergyLedgers(restartState)" in coupler
    persist = coupler[coupler.index("label GpuSolidThermalCoupler::persistAtWriteTime") :]
    assert "pendingWallEnergySnapshot(completed)" in persist
    assert "writePendingWallEnergyTemporary(pendingWallEnergy)" in persist
    advance = coupler[coupler.index("GpuThermalCouplingResult GpuSolidThermalCoupler::exchangeIfDue") : coupler.index("label GpuSolidThermalCoupler::persistAtWriteTime")]
    assert "writePendingWallEnergyTemporary" not in advance


def test_checkpoint_format_requires_a_hashed_pending_ledger_artifact():
    coupler = COUPLER.read_text(encoding="utf-8")
    state = STATE.read_text(encoding="utf-8")
    assert "completed.formatVersion = 4;" in coupler
    assert '"pendingWallEnergyLedgers"' in coupler
    assert "manifest.state.formatVersion >= 4" in state
    assert "pending wall-energy ledger count does not match manifest" in state
    assert "legacy lagged CHT checkpoint has no pending wall-energy ledger" in state


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


def test_restart_requires_exact_coupled_face_mapping():
    coupler = COUPLER.read_text(encoding="utf-8")
    assert "pending.faceIds.size() != configuredFaces.size()" in coupler
    assert "pending.faceIds.begin()" in coupler
    assert "configuredFaces.begin()" in coupler
