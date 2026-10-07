from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/compute_source_hash.py"
SPEC = importlib.util.spec_from_file_location("compute_source_hash", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_covers_effective_common_and_solver_build_inputs() -> None:
    paths = {entry[0] for entry in MODULE.source_manifest(ROOT)}
    required = {
        "Allwmake",
        "tools/compute_source_hash.py",
        "common/GpuSchedulingConfiguration.H",
        "common/GpuToolB1.cuh",
        "common/gasNumerics/RiemannGasFlux.cuh",
        "common/gasNumerics/GpuPackingProjectionAlgebra.cuh",
        "common/gasNumerics/GpuParticlePhysicsAlgebra.cuh",
        "common/gpu/GpuCouplingMath.H",
        "common/wall/GpuColdWallSolidification.H",
        "applications/gasUGKP/private_backend/build_private_backend.sh",
        "applications/FSH/private_backend/build_private_backend.sh",
        "applications/CHT/tools/build_cuda_solver.sh",
    }
    assert required <= paths
    assert all("build_logs" not in path for path in paths)
    assert all("/tests/" not in path for path in paths)
    assert all("/results/" not in path for path in paths)


def test_hash_is_deterministic_and_records_links() -> None:
    first = MODULE.source_hash(ROOT)
    second = MODULE.source_hash(ROOT)
    assert first == second
    assert len(first) == 64
    entries = {entry[0]: entry for entry in MODULE.source_manifest(ROOT)}
    riemann_link = entries[
        "applications/gasUGKP/private_backend/RiemannGasFlux.cuh"
    ]
    assert riemann_link[1] == "link"
    assert riemann_link[2] == "../../../common/gasNumerics/RiemannGasFlux.cuh"


def test_hash_changes_with_common_content_and_link_target(tmp_path: Path) -> None:
    explicit = (
        "Allwmake",
        "tools/compute_source_hash.py",
        "applications/gasUGKP/private_backend/build_private_backend.sh",
        "applications/FSH/private_backend/build_private_backend.sh",
        "applications/CHT/tools/build_cuda_solver.sh",
    )
    for relative_text in explicit:
        path = tmp_path / relative_text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_text, encoding="utf-8")
    canonical = tmp_path / "common/gasNumerics/Test.H"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("alpha", encoding="utf-8")
    second = tmp_path / "common/gasNumerics/Test2.H"
    second.write_text("alpha", encoding="utf-8")
    link = tmp_path / "applications/gasUGKP/private_backend/Test.H"
    link.symlink_to("../../../common/gasNumerics/Test.H")

    initial = MODULE.source_hash(tmp_path)
    canonical.write_text("beta", encoding="utf-8")
    changed_content = MODULE.source_hash(tmp_path)
    assert changed_content != initial

    canonical.write_text("alpha", encoding="utf-8")
    link.unlink()
    link.symlink_to("../../../common/gasNumerics/Test2.H")
    changed_target = MODULE.source_hash(tmp_path)
    assert changed_target != initial


def test_hash_includes_gas_state_inc_and_changes_with_abi_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "common/gasTransport/GasStateFields.inc"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("UGKWP_GAS_STATE_FIELD(int, nCells)\n", encoding="utf-8")
    relative = manifest.relative_to(tmp_path).as_posix()
    entries = {entry[0]: entry for entry in MODULE.source_manifest(tmp_path)}
    assert relative in entries
    initial = MODULE.source_hash(tmp_path)
    manifest.write_text("UGKWP_GAS_STATE_FIELD(long, nCells)\n", encoding="utf-8")
    assert MODULE.source_hash(tmp_path) != initial


def test_hash_excludes_test_only_inc_files(tmp_path: Path) -> None:
    fixture = tmp_path / "common/tests/Fixture.inc"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("not a production ABI", encoding="utf-8")
    assert MODULE.source_manifest(tmp_path) == ()
