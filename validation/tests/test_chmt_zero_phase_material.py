"""Run the actual native CPU material owner; no CUDA backend or emulation.

Requires a sourced Foundation OpenFOAM environment. Bounds sanitizer observes
whether the production driver consumes the unselected condensed-material index.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

ROOT = SOURCE_ROOT
APP = ROOT / "applications/CHMT"


def checked_run(command, directory, log, **kwargs):
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True, **kwargs)
    log.write_text(shlex.join(map(str, command)) + "\n" + result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def native_phase_probe(tmp_path_factory):
    if not (os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh")):
        pytest.skip("source Foundation OpenFOAM for the native CPU material regression")
    build = tmp_path_factory.mktemp("chmt_zero_phase_material")
    foam = Path(os.environ["WM_PROJECT_DIR"])
    lib = Path(os.environ["FOAM_LIBBIN"])
    mpi = lib / os.environ.get("FOAM_MPI", "dummy")
    executable = build / "phase_probe"
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-g",
               "-ffp-contract=off", "-fsanitize=bounds", "-fno-sanitize-recover=bounds",
               "-DNoRepository", "-DWM_DP", "-DWM_LABEL_SIZE=32", "-DUGKWP_GAS_SPECIES=2",
               "-I" + str(APP), "-I" + str(ROOT / "common")]
    for include in ("OpenFOAM", "finiteVolume", "meshTools", "OSspecific/POSIX"):
        command.append("-I" + str(foam / "src" / include / "lnInclude"))
    command.append(str((VALIDATION_ROOT / 'tests/fixtures/chmt_zero_phase_material/probe.C')))
    command.extend(str(APP / source) for source in (
        "materials/CpuMaterialDriver.C", "materials/MaterialTransport.C",
        "film/CpuFilmDriver.C", "ablation/CpuSurfaceInterface.C", "mesh/Geometry.C"))
    command.extend(["-L" + str(lib), "-Wl,-rpath," + str(lib),
                    "-lfiniteVolume", "-lmeshTools", "-lOpenFOAM",
                    "-L" + str(mpi), "-Wl,-rpath," + str(mpi), "-lPstream",
                    "-o", str(executable)])
    checked_run(command, build, build / "build.log", timeout=300)
    case = build / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    header = "FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n"
    files = {
        "controlDict": "application phase_probe; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .01; writeControl timeStep; writeInterval 1; runTimeModifiable false;",
        "fvSchemes": "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear orthogonal; } interpolationSchemes { default linear; } snGradSchemes { default orthogonal; }",
        "fvSolution": "solvers {}",
        "blockMeshDict": "convertToMeters 1; vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1)); blocks (hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)); edges (); boundary (left { type wall; faces ((0 4 7 3)); } right { type wall; faces ((1 2 6 5)); } sides { type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }); mergePatchPairs ();",
    }
    for name, contents in files.items():
        (case / "system" / name).write_text(header % name + contents + "\n")
    checked_run(["blockMesh", "-case", str(case)], build, build / "blockMesh.log", timeout=30)
    return executable, case


@pytest.mark.parametrize("mode", ["unselected", "selected", "melting"])
def test_native_cpu_zero_phase_material(native_phase_probe, mode):
    executable, case = native_phase_probe
    output = checked_run([str(executable), "-case", str(case), "-phaseCase", mode],
                         executable.parent, executable.parent / (mode + ".log"), timeout=30)
    assert "PASS native CPU phase " + mode in output
