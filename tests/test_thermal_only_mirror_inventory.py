"""Thermal-only helpers must never become required inputs in the fluid repo."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOS = ("gpu-riemann-gkp-main", "ugkp-thermal")
THERMAL_ONLY = (
    "common/GpuThermalLaunchOccupancy.cuh",
    "common/operators/appendSelectedStuckParticleIndex.cuh",
    "common/operators/atomicAddParticleWallEnergyByFace.cuh",
    "common/wall/GpuColdWall1DAdvance.inl",
    "common/wall/GpuColdWall1DInputs.inl",
    "common/wall/GpuColdWall1DGasOnly.cuh",
)
SHARED = "common/CsrPersistentQueue.cuh"


def manifest():
    return json.loads((ROOT / "tools/managed_mirrors.json").read_text())


@pytest.mark.parametrize("relative", THERMAL_ONLY)
def test_thermal_helper_has_only_thermal_inventory_ownership(relative):
    data = manifest()
    assert relative not in {entry["path"] for entry in data["mirrors"]}
    assert relative in data["local_only"]["ugkp-thermal"]
    assert relative not in data["local_only"]["gpu-riemann-gkp-main"]


@pytest.fixture
def pair(tmp_path):
    # Retain the real classification of the six helpers and a genuinely shared
    # control, without copying repositories or hiding failure behind unrelated
    # historical version skew between their other common implementations.
    data = manifest()
    selected = set(THERMAL_ONLY) | {SHARED}
    data["mirrors"] = [entry for entry in data["mirrors"]
                       if entry["path"] in selected]
    data["local_only"] = {
        repo: [path for path in data["local_only"][repo] if path in selected]
        for repo in REPOS
    }
    data["inventory_roots"] = ["common"]
    for repo in REPOS:
        root = tmp_path / repo
        (root / "tools").mkdir(parents=True)
        (root / "common").mkdir()
        (root / "tools/managed_mirrors.json").write_text(json.dumps(data))
        shutil.copy2(ROOT / "tools/managed_mirrors.py",
                     root / "tools/managed_mirrors.py")
        (root / SHARED).write_bytes(b"shared upstream input\n")
    for relative in THERMAL_ONLY:
        target = tmp_path / "ugkp-thermal" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def snapshot(pair):
    return {path.relative_to(pair).as_posix(): path.read_bytes()
            for path in pair.rglob("*") if path.is_file()}


def run_tool(pair, repo, *args):
    env = dict(os.environ, UGKP_MANAGED_MIRROR_ROOT=str(pair),
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-B", str(pair / repo / "tools/managed_mirrors.py"),
         *args], cwd=pair / repo, env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("repo", REPOS)
def test_paired_check_accepts_thermal_helpers_absent_from_fluid(pair, repo):
    before = snapshot(pair)
    checked = run_tool(pair, repo)
    assert snapshot(pair) == before, "check-only invocation changed inputs"
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "paired comparison" in checked.stdout


@pytest.mark.parametrize("repo", REPOS)
def test_paired_check_still_rejects_shared_drift_without_copying(pair, repo):
    (pair / "gpu-riemann-gkp-main" / SHARED).write_bytes(b"shared drift\n")
    before = snapshot(pair)
    checked = run_tool(pair, repo)
    assert snapshot(pair) == before, "failed read-only check changed inputs"
    assert checked.returncode == 1, checked.stdout + checked.stderr
    assert "mirror drift: " + SHARED in checked.stdout
    assert not any(relative in checked.stdout for relative in THERMAL_ONLY)


@pytest.mark.parametrize("repo", REPOS)
def test_explicit_sync_repairs_shared_control_but_never_copies_thermal_helpers(
    pair, repo
):
    shared = pair / "gpu-riemann-gkp-main" / SHARED
    shared.write_bytes(b"shared drift\n")
    before = snapshot(pair)
    synced = run_tool(pair, repo, "--sync")
    assert synced.returncode == 0, synced.stdout + synced.stderr
    assert shared.read_bytes() == b"shared upstream input\n"
    assert all(not (pair / "gpu-riemann-gkp-main" / path).exists()
               for path in THERMAL_ONLY)
    expected = dict(before)
    expected["gpu-riemann-gkp-main/" + SHARED] = b"shared upstream input\n"
    assert snapshot(pair) == expected, "sync changed more than the shared input"
