"""Audit the densities the real adapter publishes, including conversion roundoff.

These deterministic host probes cover FP64 arithmetic, not native CUDA runs.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

import pytest
from test_gas_chemistry_adapter import SOURCE as ADAPTER_SOURCE

ROOT = SOURCE_ROOT
SOURCE = ADAPTER_SOURCE.split("int main(){")[0] + r'''
#include <string>

Real configure(Fixture& f, const std::string& scenario) {
    f.rho[0] = Real(1);
    f.mx[0] = f.my[0] = f.mz[0] = Real(0);
    Real volume;
    if (scenario == "closure" || scenario == "input") {
        f.partial[0] = Real(0x1.999999999999ap-5);
        f.partial[2] = Real(0x1.e6666666665e6p-1);
        volume = Real(0x1.6666666666666p-1);
        if (scenario == "input")
            f.partial[2] -= Real(64)*std::numeric_limits<Real>::epsilon();
    } else if (scenario == "zero" || scenario == "absent") {
        f.partial[0] = Real(0);
        f.partial[2] = Real(1) - Real(64)*std::numeric_limits<Real>::epsilon();
        volume = Real(.7);
    } else if (scenario == "mass") {
        f.partial[0] = Real(.4);
        f.partial[2] = Real(1) - f.partial[0];
        volume = Real(.3);
    } else if (scenario == "element") {
        f.partial[0] = Real(.65);
        f.partial[2] = Real(1) - f.partial[0];
        volume = Real(.3);
    } else {
        f.partial[0] = Real(.65);
        f.partial[2] = Real(1) - f.partial[0];
        volume = Real(.3);
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
            ? (Real(1e-15)) : Real(.01);
    }
    if (scenario == "energy") {
        controls.thermo.relativeEnergyTolerance = Real(0);
        controls.thermo.absoluteEnergyTolerance = Real(1e-11);
    }
    const Real interval = scenario == "zero" ? Real(0) : Real(1e-6);
    const auto input = inputFor(f, volume);
    ClosedReactorResult<Real,2> result;

    assert(advanceClosedReactor(input, interval, g.thermo, g.mechanism,
                                   controls, result));

    Real savedPartial[4], savedBulk[12];
    std::memcpy(savedPartial, f.partial, sizeof savedPartial);
    Real bulk[12] = {f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],
                    f.mz[0],f.mz[1],f.energy[0],f.energy[1],f.temperature[0],f.temperature[1]};
    std::memcpy(savedBulk, bulk, sizeof savedBulk);
    const auto status = advanceGasChemistryCell(f.state, 0, double(interval), volume);
    Real after[12] = {f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],
                     f.mz[0],f.mz[1],f.energy[0],f.energy[1],f.temperature[0],f.temperature[1]};
    assert(std::memcmp(savedBulk, after, sizeof savedBulk) == 0);
    assert(f.partial[1] == savedPartial[1] && f.partial[3] == savedPartial[3]);
    // Former input and rounded-conservation refusals now publish. Their exact
    // residuals remain checked below against the masses actually written.

    assert(status);
    const Real tolerance = g.densityClosureTolerance*f.rho[0];
    if (scenario != "input")assert(std::abs(f.partial[0] + f.partial[2] - f.rho[0]) <= tolerance);
    else {assert(f.partial[0]==result.speciesMass[0]/volume);assert(f.partial[2]==result.speciesMass[1]/volume);}
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
    // Independently account actual stored mass and elements in wider precision.
    // The direct reactor comparison above checks conversion/repair, while these
    // checks detect lost species mass or an unintended energy source.
    long double deltaMass=0,deltaElement=0;
    for(int i=0;i<2;++i){
        const long double delta=static_cast<long double>(published[i])-input.speciesMass[i];
        deltaMass+=delta;
        deltaElement+=delta*g.thermo.elementComposition[i]/g.thermo.species[i].molarMass;
        assert(std::isfinite(published[i]));
    }
    assert(std::abs(deltaMass)<1e-12L);
    assert(std::abs(deltaElement)<1e-10L);
    assert(std::isfinite(result.temperature));
    long double publishedEnergy=0;
    for(int i=0;i<2;++i)publishedEnergy+=static_cast<long double>(published[i])*speciesE(i,result.temperature,g.thermo);
    assert(std::isfinite(publishedEnergy));
    assert(std::abs(publishedEnergy-input.internalEnergy)<1e-7L);
    assert(f.cellStatus[0] == 0 && f.chemistryStatus[0]);
}
'''


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rounded-chemistry-" + 'double')
    src = tmp / "publication.cpp"
    src.write_text(SOURCE.replace("TEST_REAL", 'double'))
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
