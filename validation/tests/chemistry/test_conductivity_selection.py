import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_nonfinite_direct_conductivity_is_not_absent(tmp_path):
    source = tmp_path / 'conductivity.cpp'
    source.write_text(r'''
#include "common/gasTransport/GasStateView.H"
#include <cassert>
#include <limits>
struct State { double gasThermalConductivity; };
struct ConfiguredState { double gasThermalConductivity; bool gasThermalConductivityConfigured; };
int main(){
 assert(ugkwp::gasHasDirectConductivity(ConfiguredState{-1,true}));
 assert(!ugkwp::gasHasDirectConductivity(ConfiguredState{-1,false}));
 assert(!ugkwp::gasHasDirectConductivity(State{-1}));
 assert(ugkwp::gasHasDirectConductivity(State{0}));
 assert(ugkwp::gasHasDirectConductivity(State{std::numeric_limits<double>::quiet_NaN()}));
 assert(ugkwp::gasHasDirectConductivity(State{-std::numeric_limits<double>::infinity()}));
}
''')
    binary = tmp_path / 'conductivity'
    subprocess.run(['g++','-std=c++17','-I',str(SOURCE_ROOT),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
