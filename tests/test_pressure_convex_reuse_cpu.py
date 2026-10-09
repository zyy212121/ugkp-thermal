"""Execute the production limiter; count real helper calls in test-only copies.

Counters establish source-level work removal, not GPU instruction counts/timing.
The frozen reference protects FP32/FP64 behavior independently of new wrappers.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT / "tests/fixtures/pressure"


@pytest.fixture(scope="module", params=["float", "double"])
def executable(request, tmp_path_factory):
    compiler=shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required")
    directory=tmp_path_factory.mktemp("convex_reuse_"+request.param)
    # Keep instrumentation out of production. Actual arithmetic/control flow
    # and the actual pressureLocalConvexScale call site remain unchanged.
    algebra=(ROOT / "common/GpuPressureConvexAlgebra.cuh").read_text()
    algebra, roots=re.subn(r"(PRESSURE_CONVEX_HD (?:float|double) root\([^\n]+\{)",r"\1 ++rootCalls;",algebra)
    algebra, fractions=re.subn(r"(PRESSURE_CONVEX_HD (?:float|double) fraction\([^\n]+\{)",r"\1 ++fractionCalls;",algebra)
    algebra, initials=re.subn(r"(InitialState<Real> initialState\s*\([^)]*\)\s*\{)",r"\1 ++initialCalls;",algebra)
    assert (roots,fractions,initials)==(2,2,1), "test instrumentation no longer matches production helper signatures"
    (directory / "GpuPressureConvexAlgebra.cuh").write_text(algebra)
    for name in ("GpuPressureKickAccumulation.cuh", "GpuPressureUnsortedAlgebra.cuh"):
        shutil.copyfile(ROOT / "common" / name,directory / name)
    binaries={}
    for kind,include in [("instrumented",directory),("normal",ROOT / "common")]:
        binary=directory / kind
        build=subprocess.run([compiler,"-std=c++17","-O2","-Wall","-Wextra","-pedantic",
            "-DTEST_REAL="+request.param,"-I",str(include),"-I",str(FIXTURES),
            str(FIXTURES / "convex_reuse_cpu.cpp"),"-o",str(binary)],capture_output=True,text=True)
        assert build.returncode==0,build.stdout+build.stderr
        binaries[kind]=binary
    return binaries


@pytest.mark.parametrize("mode",["counts","counts_fast_paths"])
def test_cell_and_face_invariants_are_reused(executable,mode):
    run=subprocess.run([str(executable["instrumented"]),mode],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


@pytest.mark.parametrize("mode",["equivalence","cells"])
@pytest.mark.parametrize("kind",["instrumented","normal"])
def test_prepared_limiter_matches_frozen_reference(executable,mode,kind):
    run=subprocess.run([str(executable[kind]),mode],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
