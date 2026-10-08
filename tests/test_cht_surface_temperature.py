"""Host OpenFOAM tests of the actual solid candidate, without a GPU or time loop.

Run with a Foundation OpenFOAM environment sourced (wmake and blockMesh on PATH).
The fixtures use the production candidate and properties, with test-only runtime
observers and explicitly requested fault injection in the generated translation unit.
"""
from pathlib import Path
import shutil
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
THERMAL = ROOT / "applications/CHT/thermal"


def header(kind, obj):
    return f"FoamFile {{ version 2.0; format ascii; class {kind}; object {obj}; }}\n"


@pytest.fixture(scope="module")
def candidate_executable(tmp_path_factory):
    if not (shutil.which("wmake") and shutil.which("blockMesh")):
        pytest.skip("source a Foundation OpenFOAM environment for the production candidate regression")
    build = tmp_path_factory.mktemp("cht_surface_build")
    (build / "Make").mkdir()
    shutil.copy(ROOT / "tests/fixtures/cht_surface/candidate.cpp", build / "candidate.C")
    production = (THERMAL / "GpuSolidThermalCoupler.C").read_text()
    accumulation = production.split("scalarField combined(gasPreview[pairI]);", 1)[1].split("solidEnergy[pairI] =", 1)[0]
    (build / "CompletedIntervalEnergy.inc").write_text("scalarField combined(gasPreview[pairI]);" + accumulation)

    # Runtime observers live only in the generated test translation unit.
    # They count real field materializations in the production solve, without
    # changing its values, iteration ordering, or OpenFOAM dependencies.
    observed = production.replace(
        "scalarField surface(patch);",
        "scalarField surface(patch); ++chtSurfaceMaterializations;",
    )
    observed = re.sub(
        r"(?m)^(\s*)(previousSurface\[coupledI\] =\s*\n)",
        r"\1++chtSurfaceSnapshots;\n\1\2",
        observed,
    )
    # Corrupt the completed candidate, not the input, to exercise the actual
    # convergence validation on a late face after the OpenFOAM boundary update.
    observed = observed.replace(
        "        finalSurfaceResidual = scalar(0);",
        """        if (chtSurfaceFault)
        {
            fixedGradientFvPatchScalarField& faultPatch =
                refCast<fixedGradientFvPatchScalarField>
                (candidate->boundaryFieldRef()[coupledPatchIds_[0]]);
            const label faultFace = faultPatch.size()-1;
            const scalar nan = std::numeric_limits<scalar>::quiet_NaN();
            if (chtSurfaceFault == 1) faultPatch[faultFace] += 1;
            if (chtSurfaceFault == 2) faultPatch[faultFace] = nan;
            if (chtSurfaceFault == 3) faultPatch[faultFace] = 0;
            if (chtSurfaceFault == 4) faultPatch.gradient()[faultFace] = nan;
            if (chtSurfaceFault == 5)
                candidate->primitiveFieldRef()[faultPatch.patch().faceCells()[faultFace]] = nan;
            if (chtSurfaceFault == 6 || chtSurfaceFault == 7)
                const_cast<scalarField&>(faultPatch.patch().deltaCoeffs())[faultFace] =
                    chtSurfaceFault == 6 ? scalar(0) : nan;
        }
        finalSurfaceResidual = scalar(0);""",
    )
    (build / "GpuSolidThermalCoupler.C").write_text(
        "int chtSurfaceMaterializations = 0;\nint chtSurfaceSnapshots = 0;\nint chtSurfaceFault = 0;\n"
        + observed
    )
    for name in ("GpuSolidThermalProperties", "GpuThermalExchangeState"):
        (build / (name + ".C")).write_text(f'#include "{THERMAL / (name + ".C")}"\n')
    (build / "Make/files").write_text("candidate.C\nGpuSolidThermalCoupler.C\nGpuSolidThermalProperties.C\nGpuThermalExchangeState.C\n\nEXE = " + str(build / "candidate") + "\n")
    (build / "Make/options").write_text(f"EXE_INC = -I{THERMAL} -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools\n")
    result = subprocess.run(["wmake"], cwd=build, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return build / "candidate"


def make_case(case, q, variable=False, balanced=False, width=1, transverse_cells=1):
    for sub in ("0", "constant", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)
    (case / "system/controlDict").write_text(header("dictionary", "controlDict") + "application candidate; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 0.1; writeControl timeStep; writeInterval 1; writePrecision 17; runTimeModifiable false;\n")
    (case / "system/fvSchemes").write_text(header("dictionary", "fvSchemes") + "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear orthogonal; } interpolationSchemes { default linear; } snGradSchemes { default orthogonal; }\n")
    (case / "system/fvSolution").write_text(header("dictionary", "fvSolution") + "solvers { TsolidCandidate { solver PCG; preconditioner DIC; tolerance 1e-14; relTol 0; } }\n")
    (case / "system/blockMeshDict").write_text(header("dictionary", "blockMeshDict") + f"convertToMeters 1; vertices ((0 0 0) ({width} 0 0) ({width} 1 0) (0 1 0) (0 0 1) ({width} 0 1) ({width} 1 1) (0 1 1)); blocks (hex (0 1 2 3 4 5 6 7) (1 {transverse_cells} 1) simpleGrading (1 1 1)); edges (); boundary (left {{ type wall; faces ((0 4 7 3)); }} right {{ type wall; faces ((1 2 6 5)); }} sides {{ type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}); mergePatchPairs ();\n")
    right = "fixedGradient; gradient uniform 0" if balanced else "zeroGradient"
    (case / "0/T").write_text(header("volScalarField", "T") + "dimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField { left { type fixedGradient; gradient uniform 0; } right { type " + right + "; } sides { type zeroGradient; } }\n")
    kappa = "type table; outOfBounds error; values ((100 2) (1000 20));" if variable else "type constant; value 6;"
    (case / "constant/testProperties").write_text(header("dictionary", "testProperties") + f"heatFlux {q}; variable {'true' if variable else 'false'}; balanced {'true' if balanced else 'false'}; properties {{ rho {{ type constant; value 1000; }} Cp {{ type constant; value 1; }} kappa {{ {kappa} }} emissivity {{ type constant; value 0.8; }} }}\n")
    result = subprocess.run(["blockMesh", "-case", str(case)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("q,variable,balanced,width", [
    (600, False, False, 1),
    (-600, False, False, 1),
    (1000, True, False, 1),
    (-500, True, False, 1),
    (600, False, True, 1),
    (600, False, True, 0.25),
])
def test_production_surface_closure(candidate_executable, tmp_path, q, variable, balanced, width):
    make_case(tmp_path, q, variable, balanced, width)
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS surface and ledger closure" in result.stdout


def test_gas_contact_and_radiation_use_surface_but_solid_solid_uses_owner():
    source = (THERMAL / "GpuSolidThermalCoupler.C").read_text()
    # Gas/contact data flow is exercised by the native mapping regression below.
    radiation = source.split("List<scalarField> fluidWallEmissivity() const", 1)[1].split("std::pair<fileName, fileName> writeManifestTemporary", 1)[0]
    assert "coupledPatchSurfaceTemperature(Tsolid(), solidPatchI)" in radiation
    solid_solid = source.split("GpuThermalCouplingResult GpuSolidThermalCoupler::exchangeIfDue", 1)[1].split("const label auxiliaryPatchI =", 1)[1].split("commitPhaseStarted = true;", 1)[0]
    assert solid_solid.count("coupledPatchOwnerTemperature") == 4
    assert "coupledPatchSurfaceTemperature" not in solid_solid


def test_only_particle_and_only_radiation_energy(candidate_executable, tmp_path):
    # Execute the production energy-aggregation block before the real candidate.
    for channel in ("particle", "radiation", "mixed"):
        case = tmp_path / channel
        make_case(case, 750, variable=True)
        with (case / "constant/testProperties").open("a") as stream:
            stream.write(f"energyChannel {channel};\n")
        result = subprocess.run([str(candidate_executable), "-case", str(case)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_existing_aitken_includes_surface_convergence(candidate_executable, tmp_path):
    make_case(tmp_path, 1000, variable=True)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write("aitken true;\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def mapping_executable(candidate_executable):
    build = candidate_executable.parent
    source = (THERMAL / "GpuSolidThermalCoupler.C").read_text()
    mapping = source.split("    void mapSolidWallTemperatureToFluid", 1)[1].split("    List<scalarField> fluidWallEmissivity() const", 1)[0]
    (build / "WallMapping.inc").write_text("    void mapSolidWallTemperatureToFluid" + mapping)
    snapshot = source.split("    List<scalarField> previousSurface(coupledPatchIds_.size());", 1)[1].split("    mesh_.schemes().setFluxRequired", 1)[0]
    (build / "SurfaceSnapshot.inc").write_text("    List<scalarField> previousSurface(coupledPatchIds_.size());" + snapshot)
    membership = source.split("bool isCoupledPatch(", 1)[1].split("class ScalarSolverLogSilencer", 1)[0]
    (build / "PatchMembership.inc").write_text("bool isCoupledPatch(" + membership)
    properties = source.split("        forAll(mesh_.boundary(), patchI)", 1)[1].split("        forAll(coupledPatchIds_, coupledI)", 1)[0]
    properties = "        forAll(mesh_.boundary(), patchI)" + properties
    gather = "candidate->boundaryField()[patchI].patchInternalField()"
    assert properties.count(gather) == 1
    (build / "BoundaryProperties.inc").write_text(properties.replace(gather, f"(++ownerGathers, {gather})"))
    shutil.copy(ROOT / "tests/fixtures/cht_surface/mapping.cpp", build / "mapping.C")
    (build / "Make/files").write_text("mapping.C\nGpuSolidThermalCoupler.C\nGpuSolidThermalProperties.C\nGpuThermalExchangeState.C\n\nEXE = " + str(build / "mapping") + "\n")
    result = subprocess.run(["wmake"], cwd=build, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return build / "mapping"


@pytest.mark.parametrize("mode", ["mapping", "snapshot", "properties"])
def test_surface_state_is_read_only_when_consumed(mapping_executable, tmp_path, mode):
    make_case(tmp_path, 600, variable=True)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write(f"testMode {mode};\n")
    result = subprocess.run([str(mapping_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {mode}" in result.stdout


def test_unconverged_surface_is_rejected(candidate_executable, tmp_path):
    make_case(tmp_path, 1000, variable=True)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write("maximumIterations 2;\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "solid candidate nonlinear solve did not converge" in result.stderr
    assert "surfaceFluxResidual=" in result.stderr


def test_solid_solid_owner_conductivity_is_retained(candidate_executable, tmp_path):
    make_case(tmp_path, 1000, variable=True)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write("ownerConductivity true;\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_startup_rejects_missing_gradient(candidate_executable, tmp_path):
    make_case(tmp_path, 600)
    path = tmp_path / "0/T"
    path.write_text(path.read_text().replace("gradient uniform 0;", ""))
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "gradient" in result.stderr and "missing" in result.stderr


def test_explicit_completed_interval_ledger_and_restart_guards_are_retained():
    source = (THERMAL / "GpuSolidThermalCoupler.C").read_text()
    exchange = source.split("GpuThermalCouplingResult GpuSolidThermalCoupler::exchangeIfDue", 1)[1]
    order = ["peekGasWallEnergy()", "scalarField combined(gasPreview[pairI]);",
             "combined[faceI] +=", "mapIntegratedFaceEnergyToSolid(pairI, combined)",
             "solveTemporarySolidCandidate", "commitPhaseStarted = true;",
             "downloadAndResetGasWallEnergy", "requireBitwiseGasWallPreviewMatch",
             "publishSolidCandidate", "mapSolidWallTemperatureToFluid(Tgas)"]
    positions = [exchange.index(item) for item in order]
    assert positions == sorted(positions)
    accumulation = exchange.split("scalarField combined(gasPreview[pairI]);", 1)[1].split("autoPtr<GpuSolidThermalCandidate>", 1)[0]
    assert "radiation().solidWallRadiationEnergyJ" in accumulation
    assert "combined[faceI] += particleContactPreview" in accumulation
    assert "particleContactPreview += particleReflectedPreview;" in exchange
    assert "reconstructed wall temperature checksum differs from restart state" in source
    assert "newlyUploadedWallTemperatureSha1" in source
    assert "ThermalRestartPreflight::fileSha1(timePath/solidRelative)" in source


def test_startup_reconstructs_nonzero_gradient_ignoring_stale_value(candidate_executable, tmp_path):
    make_case(tmp_path, 600)
    path = tmp_path / "0/T"
    path.write_text(path.read_text().replace("gradient uniform 0;", "gradient uniform 120; value uniform 999;", 1))
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write("initialSurface 360;\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("aitken", [False, True])
def test_convergence_does_not_materialize_surface_fields_each_iteration(
    candidate_executable, tmp_path, aitken
):
    # Restoring the old gather/copy/scan path makes this native runtime test fail.
    # Variable conductivity needs several nonlinear iterations, so copying an
    # unchanged snapshot at each iteration cannot accidentally satisfy the count.
    make_case(tmp_path, 1000, variable=True)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write(f"checkSurfaceWork true; aitken {str(aitken).lower()};\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS surface work materializations=0 snapshots=1" in result.stdout


@pytest.mark.parametrize("q", [500, -500])
@pytest.mark.parametrize("aitken", [False, True])
@pytest.mark.parametrize("ownership", ["gas", "mixed", "owner"])
def test_multiface_variable_conductivity_preserves_patch_ownership(
    candidate_executable, tmp_path, q, aitken, ownership
):
    make_case(tmp_path, q, variable=True, balanced=True, transverse_cells=3)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write(f"aitken {str(aitken).lower()}; checkSurfaceWork true;\n")
        if ownership == "mixed":
            stream.write("mixedOwnerConductivity true;\n")
        elif ownership == "owner":
            stream.write("ownerConductivity true;\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {"gas": 2, "mixed": 1, "owner": 0}[ownership]
    assert f"PASS surface work materializations=0 snapshots={expected}" in result.stdout
    assert result.stdout.count("FACE ") == 6


@pytest.mark.parametrize("fault", range(1, 8))
def test_convergence_rejects_invalid_completed_surface_on_late_face(
    candidate_executable, tmp_path, fault
):
    # Omitting the convergence validator accepts a stale, otherwise valid last
    # face; dropping individual guards loses their specific rejection path.
    make_case(tmp_path, 600, transverse_cells=3)
    with (tmp_path / "constant/testProperties").open("a") as stream:
        stream.write(f"surfaceFault {fault};\n")
    result = subprocess.run([str(candidate_executable), "-case", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    reason = "stale coupled solid surface" if fault == 1 else "invalid accepted coupled solid surface state"
    assert reason in result.stderr
