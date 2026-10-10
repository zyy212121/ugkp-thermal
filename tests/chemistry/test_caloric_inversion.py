"""Generic common inversion shares the gas solver without a gas dependency."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_generic_caloric_inversion(tmp_path):
    source = tmp_path / "caloric.cpp"
    source.write_text(r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
#include <initializer_list>
struct Condensed {
    ugkwp::ThermoStatus operator()(double T,double& U,double& C) const {
        U=-2e6+800*T+.2*T*T; C=800+.4*T;
        return ugkwp::ThermoStatus::Success;
    }
};
struct Invalid {
    ugkwp::ThermoStatus operator()(double,double& U,double& C) const {
        U=0;C=-1;return ugkwp::ThermoStatus::Success;
    }
};
int main() {
    ugkwp::ThermoInversionControls<double> controls;
    for(double target:{200.,777.,2000.}) {
        double U,C;Condensed{}(target,U,C);double T=400;
        assert(ugkwp::invertCaloricEnergy(Condensed{},U,200.,2000.,controls,T)==ugkwp::ThermoStatus::Success);
        assert(std::abs(T-target)<1e-7);
    }
    double T=456;
    assert(ugkwp::invertCaloricEnergy(Condensed{},1e12,200.,2000.,controls,T)==ugkwp::ThermoStatus::EnergyOutOfRange);
    assert(T==456);
    assert(ugkwp::invertCaloricEnergy(Invalid{},0.,200.,2000.,controls,T)==ugkwp::ThermoStatus::NonPositiveHeatCapacity);
    assert(T==456);
    controls.maximumIterations=1;
    assert(ugkwp::invertCaloricEnergy(Condensed{},-1e6,200.,2000.,controls,T)==ugkwp::ThermoStatus::IterationLimit);
    assert(T==456);
    controls.maximumIterations=100;controls.minTemperature=800;
    assert(ugkwp::invertCaloricEnergy(Condensed{},-1.5e6,200.,2000.,controls,T)==ugkwp::ThermoStatus::EnergyOutOfRange);
    assert(T==456);
}
''')
    binary = tmp_path / "caloric"
    subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
