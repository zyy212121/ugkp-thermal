"""Actual OF10 solves and physical outputs survive removal of report-only work.

The baseline and current material owners compile with their own dependencies.
Solve counters exist only in the test translation units.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import os
import shlex
import shutil
import subprocess
import tarfile

import pytest

ROOT = SOURCE_ROOT
APP = ROOT / "applications/CHMT"
BASE = "0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668"
CASES = ["cpu_material", "conduction_terminal", "owner_terminal",
         "transport_terminal", "sweep_terminal", "geometry_terminal",
         "surface_terminal", "reacting_wall_film", "owner_endpoint", "sources_terminal"]


def run(command, log, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    log.write_text(shlex.join(map(str, command)) + "\n" + result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def native_material_build(tmp_path_factory):
    if os.environ.get("WM_PROJECT_VERSION") != "10" or not shutil.which("blockMesh"):
        pytest.skip("source Foundation OpenFOAM 10 for actual material solve parity")
    build = tmp_path_factory.mktemp("material_report_removal")
    foam = Path(os.environ["WM_PROJECT_DIR"])
    lib = Path(os.environ["FOAM_LIBBIN"])
    mpi = lib / os.environ.get("FOAM_MPI", "dummy")
    flags = [os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-ffp-contract=off",
             "-DNoRepository", "-DWM_DP", "-DWM_LABEL_SIZE=32", "-DF2_TERMINAL=1"]
    includes = ["-I" + str(APP), "-I" + str(ROOT / "common")]
    for name in ("OpenFOAM", "finiteVolume", "meshTools", "OSspecific/POSIX"):
        includes.append("-I" + str(foam / "src" / name / "lnInclude"))
    links = ["-L" + str(lib), "-Wl,-rpath," + str(lib), "-lfiniteVolume",
             "-lmeshTools", "-lOpenFOAM", "-L" + str(mpi), "-Wl,-rpath," + str(mpi), "-lPstream"]
    baseline = build / "baseline"
    baseline.mkdir()
    archive = subprocess.check_output(
        ["git", "archive", BASE, "applications/CHMT", "common"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(baseline, filter="data")
    baseline_app = baseline / "applications/CHMT"
    saved = os.environ.get("CHMT_MATERIAL_BASELINE_ROOT")
    if saved:
        for source in ("materials/CpuMaterialDriver.C", "materials/CpuMaterialDriver.H"):
            (baseline_app / source).write_bytes(
                (Path(saved) / "applications/CHMT" / source).read_bytes())
    objects = {}
    for label, owner in (("baseline", baseline_app), ("current", APP)):
        objects[label] = []
        for source in ("materials/MaterialTransport.C", "film/CpuFilmDriver.C",
                       "ablation/CpuSurfaceInterface.C", "mesh/Geometry.C", "restart/Checkpoint.C"):
            obj = build / (label + "-" + Path(source).stem + ".o")
            run(flags + ["-I" + str(owner)] + includes
                + ["-c", str(owner / source), "-o", str(obj)],
                obj.with_suffix(".build.log"), timeout=300)
            objects[label].append(str(obj))
    case = build / "case"
    (case / "constant").mkdir(parents=True)
    shutil.copytree((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/of_case/system'), case / "system")
    run(["blockMesh", "-case", str(case)], build / "blockMesh.log", timeout=30)
    return build, flags, includes, links, objects, baseline_app, case


@pytest.mark.parametrize("name", CASES)
def test_native_material_physical_and_solve_parity(native_material_build, name):
    build, flags, includes, links, objects, baseline_app, case = native_material_build
    source = ((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_cpu_material_of.C') if name == "cpu_material"
              else (VALIDATION_ROOT / 'solvers/CHMT/tests') / ("test_" + name + "_of.C"))
    observations = []
    for label, overlay in (("baseline", baseline_app), ("current", APP)):
        executable = build / (name + "-" + label)
        run(flags + ["-I" + str(overlay)] + includes + [str(source)] + objects[label]
            + links + ["-o", str(executable)], executable.with_suffix(".build.log"), timeout=300)
        output = run([str(executable), "-case", str(case)],
                     executable.with_suffix(".run.log"), timeout=60)
        snapshot = "\n".join(line for line in output.splitlines() if line.startswith("CHMT_PHYSICAL "))
        assert snapshot, "native fixture did not emit its physical observation"
        executable.with_suffix(".physical.txt").write_text(snapshot + "\n")
        observations.append(snapshot)
    assert observations[0] == observations[1], "actual solves or FP64 physical/packet/time outputs changed"


def test_material_and_evolution_reports_omit_removed_diagnostics(tmp_path):
    source = tmp_path / "reports.cpp"
    source.write_text(r'''
#include "materials/CpuMaterialDriver.H"
#include "io/MultirateEvolution.H"
#include <type_traits>
#include <utility>
#define FIELD_CHECK(member) \
 template<class T,class=void>struct has_##member:std::false_type{}; \
 template<class T>struct has_##member<T,std::void_t<decltype(std::declval<T>().member)>>:std::true_type{};
FIELD_CHECK(linearSolves)
FIELD_CHECK(nonlinearIterations)
FIELD_CHECK(minimumTemperature)
FIELD_CHECK(energyResidual)
FIELD_CHECK(numericalMassRoundoff)
FIELD_CHECK(budgetDelta)
FIELD_CHECK(failedLocalDt)
FIELD_CHECK(boundaryDriveSamples)
FIELD_CHECK(materialRhsEvaluations)
FIELD_CHECK(reactionSteps)
FIELD_CHECK(filmSteps)
FIELD_CHECK(solidTargetSweeps)
static_assert(!has_linearSolves<chmt::CpuMaterialReport>::value,"material solve report remains");
static_assert(!has_nonlinearIterations<chmt::CpuMaterialReport>::value,"iteration report remains");
static_assert(!has_minimumTemperature<chmt::CpuMaterialReport>::value,"temperature report remains");
static_assert(!has_energyResidual<chmt::CpuMaterialReport>::value,"energy report remains");
static_assert(!has_linearSolves<chmt::WindowEvolutionReport>::value,"evolution solve report remains");
static_assert(!has_energyResidual<chmt::WindowEvolutionReport>::value,"evolution audit report remains");
static_assert(!has_numericalMassRoundoff<chmt::CpuMaterialReport>::value,"material roundoff report remains");
static_assert(!has_budgetDelta<chmt::CpuMaterialReport>::value,"material budget report remains");
static_assert(!has_numericalMassRoundoff<chmt::WindowEvolutionReport>::value,"evolution roundoff report remains");
static_assert(!has_failedLocalDt<chmt::CpuMaterialReport>::value,"failed duration report remains");
static_assert(!has_boundaryDriveSamples<chmt::CpuMaterialReport>::value,"drive sample report remains");
static_assert(!has_materialRhsEvaluations<chmt::CpuMaterialReport>::value,"RHS report remains");
static_assert(!has_reactionSteps<chmt::CpuMaterialReport>::value,"reaction report remains");
static_assert(!has_filmSteps<chmt::CpuMaterialReport>::value,"film report remains");
static_assert(!has_solidTargetSweeps<chmt::CpuMaterialReport>::value,"target sweep report remains");
int main(){}
''')
    run([os.environ.get("CXX", "g++"), "-std=c++17", "-I" + str(APP),
         "-I" + str(ROOT / "common"), str(source), "-o", str(tmp_path / "reports")],
        tmp_path / "reports.build.log", timeout=60)
