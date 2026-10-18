"""Independent SDIRK2 method and temporal-convergence oracle.

These tests do not claim the removed adaptive solver's requested error tolerance
is achieved by one step. The external uniform partitions are fixed in advance.
"""
import math
from pathlib import Path
import subprocess
import pytest
from validation.tests.chemistry.test_gas_rates import SOURCE as RATE_SOURCE
from validation.tools.paths import SOURCE_ROOT

MODEL = RATE_SOURCE.split("int main(){", 1)[0]
SOURCE = MODEL + r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <iomanip>
#include <iostream>
int main() {
  Model model;
  for (int steps : {1,2,4,8,16,32}) {
    ClosedReactorInput<double,2> state;
    state.speciesMass[0]=.028; state.totalMass=.028; state.volume=.001;
    state.internalEnergy=mixtureEnergy(state.speciesMass,700.,model.t);
    const double originalEnergy=state.internalEnergy;
    ChemistryControls<double> controls;
    controls.relativeTolerance=1e-7;
    controls.absoluteMassFractionTolerance=1e-13;
    controls.absoluteTemperatureTolerance=1e-6;
    for (int i=0;i<steps;++i) {
      ClosedReactorResult<double,2> result;
      bool evaluated=false;
      auto status=advanceClosedReactor(state,.25/steps,model.t,model.m,
                                       controls,result,&evaluated);
      assert(status && evaluated && status.attemptedSteps==1);
      assert(result.speciesMass[0]>=0 && result.speciesMass[1]>=0);
      assert(std::abs(result.speciesMass[0]+result.speciesMass[1]-.028)<1e-14);
      assert(std::abs(result.temperature-700.)<1e-9);
      assert(std::abs(mixtureEnergy(result.speciesMass,result.temperature,model.t)
                      -originalEnergy)<1e-9);
      for (int s=0;s<2;++s) state.speciesMass[s]=result.speciesMass[s];
    }
    std::cout << steps << " " << std::setprecision(17)
              << state.speciesMass[0] << "\n";
  }
}
'''

@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    folder=tmp_path_factory.mktemp("review-sdirk-oracle")
    source=folder/"oracle.cpp"
    source.write_text(SOURCE)
    binary=folder/"oracle"
    subprocess.run(["g++","-std=c++17","-O2","-ffp-contract=off",
                    "-Wall","-Wextra","-Werror","-I"+str(SOURCE_ROOT),
                    str(source),"-o",str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    return {int(n):float(mass) for n,mass in
            (line.split() for line in result.stdout.splitlines())}

def test_fixed_single_pass_matches_independent_sdirk_stability_function(measured):
    # For y'=-2y, the two-stage stiffly accurate SDIRK2 Butcher tableau gives
    # R(z)=(1+(1-2*gamma)*z)/(1-gamma*z)^2, gamma=1-1/sqrt(2).
    gamma=1-1/math.sqrt(2)
    for steps,mass in measured.items():
        z=-.5/steps
        stability=(1+(1-2*gamma)*z)/(1-gamma*z)**2
        assert abs(mass-.028*stability**steps)<2e-15

def test_uniform_external_partition_has_second_order_global_error(measured):
    exact=.028*math.exp(-.5)
    errors={n:abs(mass-exact) for n,mass in measured.items()}
    # Assert the actual one-step error explicitly, instead of relaxing the
    # historical adaptive-accuracy assertions.
    assert math.isclose(errors[1],9.148102899797905e-5,rel_tol=1e-10)
    assert errors[1]>2e-5
    for n in (4,8,16):
        ratio=errors[n]/errors[2*n]
        assert 3.9<ratio<4.2, (n,ratio,errors)
    print("fixed_partition_steps,measured_mass,exact_mass,absolute_error")
    for n in sorted(measured):
        print(f"{n},{measured[n]:.17g},{exact:.17g},{errors[n]:.17g}")
