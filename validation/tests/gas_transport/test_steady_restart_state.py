"""Actual standalone steady clock persistence, independent of CUDA availability."""
from test_mixture_state import compile_probe, ROOT


def test_steady_clock_roundtrip_does_not_use_output_iteration_label(tmp_path):
    header=ROOT/'applications/gasUGKPSteady/SteadyRestartState.H'
    checkpoint=tmp_path/'steadyState.dat'
    compile_probe(tmp_path,f'''
#include "{header}"
int main(){{
 SteadyRestartState state{{.125,2.375,1900,2e-5}},loaded;
 if(!writeSteadyRestartState("{checkpoint}",state))return 1;
 if(!readSteadyRestartState("{checkpoint}",loaded))return 2;
 if(loaded.boundaryTime!=.125 || loaded.samplingTime!=2.375 || loaded.sampleCount!=1900 || loaded.samplingDeltaT!=2e-5)return 3;
 // Output folder1900 is a sample index, never the boundary clock.
 loaded.samplingTime+=.001; ++loaded.sampleCount;
 if(!writeSteadyRestartState("{checkpoint}",loaded))return 4;
 SteadyRestartState resumed;
 if(!readSteadyRestartState("{checkpoint}",resumed)||resumed.boundaryTime!=.125
    ||resumed.samplingDeltaT!=2e-5 ||resumed.sampleCount!=1901||resumed.samplingTime!=2.376)return 5;
 SteadyRestartState fresh;
 if(readSteadyRestartState("{checkpoint}.absent",fresh)||fresh.boundaryTime!=0)return 6;
}}
''')
