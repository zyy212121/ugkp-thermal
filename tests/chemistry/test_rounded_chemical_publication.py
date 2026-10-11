"""Audit the densities the real adapter publishes, including conversion roundoff.

These deterministic host probes cover FP32/FP64 arithmetic, not native CUDA runs.
"""
from pathlib import Path
import subprocess

import pytest
from test_gas_chemistry_adapter import SOURCE as ADAPTER_SOURCE

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ADAPTER_SOURCE.split("int main(){")[0] + r'''
#include <string>

Real configure(Fixture& f, const std::string& scenario) {
    f.rho[0] = Real(1);
    f.mx[0] = f.my[0] = f.mz[0] = Real(0);
    Real volume;
    if (scenario == "closure" || scenario == "input") {
        f.partial[0] = sizeof(Real) == 4 ? Real(0x1.99999ap-4) : Real(0x1.999999999999ap-5);
        f.partial[2] = sizeof(Real) == 4 ? Real(0x1.cccbccp-1) : Real(0x1.e6666666665e6p-1);
        volume = sizeof(Real) == 4 ? Real(0x1.333334p-2) : Real(0x1.6666666666666p-1);
        if (scenario == "input")
            f.partial[2] -= Real(64)*std::numeric_limits<Real>::epsilon();
    } else if (scenario == "zero" || scenario == "absent") {
        f.partial[0] = Real(0);
        f.partial[2] = Real(1) - Real(64)*std::numeric_limits<Real>::epsilon();
        volume = Real(.7);
    } else if (scenario == "mass") {
        f.partial[0] = sizeof(Real) == 4 ? Real(.45) : Real(.4);
        f.partial[2] = Real(1) - f.partial[0];
        volume = sizeof(Real) == 4 ? Real(.7) : Real(.3);
    } else if (scenario == "element") {
        f.partial[0] = sizeof(Real) == 4 ? Real(.35) : Real(.65);
        f.partial[2] = Real(1) - f.partial[0];
        volume = sizeof(Real) == 4 ? Real(.7) : Real(.3);
    } else {
        f.partial[0] = sizeof(Real) == 4 ? Real(.5) : Real(.65);
        f.partial[2] = Real(1) - f.partial[0];
        volume = sizeof(Real) == 4 ? Real(3.7) : Real(.3);
    }
    Real partial[2] = {f.partial[0], f.partial[2]};
    f.energy[0] = mixtureEnergy(partial, Real(700), f.state.gasSpecies.thermo);
    return volume;
}

ClosedReactorInput<Real,2> inputFor(const Fixture& f, Real volume) {
    ClosedReactorInput<Real,2> input;
    input.volume = volume;
    input.totalMass = f.rho[0]*volume;
    input.internalEnergy = f.energy[0]*volume;
    input.speciesMass[0] = f.partial[0]*volume;
    input.speciesMass[1] = f.partial[2]*volume;
    return input;
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    Fixture f;
    const Real volume = configure(f, scenario);
    auto& g = f.state.gasSpecies;
    auto& controls = g.chemistryControls;
    if (scenario == "mass" || scenario == "element") {
        controls.relativeConservationTolerance = Real(0);
        controls.absoluteMassTolerance = scenario == "mass" ? Real(0) : Real(1e-5);
        controls.absoluteElementTolerance = scenario == "element"
            ? (sizeof(Real) == 4 ? Real(1e-6) : Real(1e-15)) : Real(.01);
    }
    if (scenario == "energy") {
        controls.thermo.relativeEnergyTolerance = Real(0);
        controls.thermo.absoluteEnergyTolerance = sizeof(Real) == 4 ? Real(.1875) : Real(1e-11);
    }
    const Real interval = scenario == "zero" ? Real(0) : Real(1e-6);
    const auto input = inputFor(f, volume);
    ClosedReactorResult<Real,2> result;
    ChemistryAudit<Real,2> reactorAudit;
    if (scenario != "input")
        assert(advanceClosedReactor(input, interval, g.thermo, g.mechanism,
                                   controls, result, reactorAudit));

    Real savedPartial[4], savedBulk[12];
    std::memcpy(savedPartial, f.partial, sizeof savedPartial);
    Real bulk[12] = {f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],
                    f.mz[0],f.mz[1],f.energy[0],f.energy[1],f.temperature[0],f.temperature[1]};
    std::memcpy(savedBulk, bulk, sizeof savedBulk);
    unsigned char savedAudit[sizeof f.audit];
    std::memcpy(savedAudit, f.audit, sizeof savedAudit);
    const auto status = advanceGasChemistryCell(f.state, 0, double(interval), volume);
    Real after[12] = {f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],
                     f.mz[0],f.mz[1],f.energy[0],f.energy[1],f.temperature[0],f.temperature[1]};
    assert(std::memcmp(savedBulk, after, sizeof savedBulk) == 0);
    assert(f.partial[1] == savedPartial[1] && f.partial[3] == savedPartial[3]);
    if (scenario == "input" || scenario == "mass" || scenario == "element" || scenario == "energy") {
        // The direct reactor passed every strict check above. Rejection here must
        // audit the extra mass/element/energy defect of the rounded publication.
        assert(!status);
        assert(status.code == ChemistryCode::ConservationFailure);
        assert(status.attemptedSteps == (scenario == "input" ? 0 : 1));
        assert(std::memcmp(savedPartial, f.partial, sizeof savedPartial) == 0);
        assert(std::memcmp(savedAudit, f.audit, sizeof savedAudit) == 0);
        assert(f.cellStatus[0] == int(GasTransportCode::ChemistryFailure));
        return 0;
    }

    assert(status);
    const Real tolerance = g.densityClosureTolerance*f.rho[0];
    assert(std::abs(f.partial[0] + f.partial[2] - f.rho[0]) <= tolerance);
    if (scenario == "closure" || scenario == "absent" || scenario == "zero") {
        assert(std::abs(savedPartial[0] + savedPartial[2] - f.rho[0]) <= tolerance);
        // Golden counterexamples: only the nonbinary-volume conversion takes
        // this already-admissible density beyond the unchanged closure gate.
        assert(std::abs(result.speciesMass[0]/volume + result.speciesMass[1]/volume
                        - f.rho[0]) > tolerance);
        if (scenario == "zero")
            assert(std::memcmp(savedPartial, f.partial, sizeof savedPartial) == 0);
        else {
            assert(f.partial[0] == result.speciesMass[0]/volume);
            assert(f.partial[2] != result.speciesMass[1]/volume);
        }
        if (scenario == "absent" || scenario == "zero") assert(f.partial[0] == Real(0));
    }

    const Real published[2] = {f.partial[0]*volume, f.partial[2]*volume};
    const auto& audit = f.audit[0];
    assert(audit.massResidual == published[0] + published[1] - input.totalMass);
    for (int s = 0; s < 2; ++s)
        assert(audit.speciesMassChange[s] == published[s] - input.speciesMass[s]);
    Real beforeElement = Real(0), afterElement = Real(0);
    for (int s = 0; s < 2; ++s) {
        const Real atoms = g.thermo.elementComposition[s]/g.thermo.species[s].molarMass;
        beforeElement += input.speciesMass[s]*atoms;
        afterElement += published[s]*atoms;
    }
    assert(audit.maximumElementResidual == std::abs(afterElement - beforeElement));
    assert(audit.energyResidual == mixtureEnergy(published, result.temperature, g.thermo)
                                   - input.internalEnergy);
    assert(audit.integratedTime == interval);
    assert(audit.acceptedSteps == reactorAudit.acceptedSteps);
    assert(audit.maximumLocalError == reactorAudit.maximumLocalError);
    assert(audit.rejectedSteps == reactorAudit.rejectedSteps);
    assert(audit.nonlinearIterations == reactorAudit.nonlinearIterations);
    assert(std::memcmp(savedAudit + sizeof audit, &f.audit[1], sizeof audit) == 0);
    assert(f.cellStatus[0] == 0 && f.chemistryStatus[0]);
}
'''


@pytest.fixture(scope="module", params=["double", "float"])
def runner(tmp_path_factory, request):
    tmp = tmp_path_factory.mktemp("rounded-chemistry-" + request.param)
    src = tmp / "publication.cpp"
    src.write_text(SOURCE.replace("TEST_REAL", request.param))
    exe = tmp / "publication"
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-I", str(ROOT), str(src), "-o", str(exe)], capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    return exe


@pytest.mark.parametrize("scenario", ["closure", "zero", "absent", "audit", "input",
                                      "mass", "element", "energy"])
def test_rounded_chemical_publication(runner, scenario):
    run = subprocess.run([str(runner), scenario], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
