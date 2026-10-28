import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/tests/gas_transport'))
from test_mixture_state import compile_probe
from test_steady_gas_residual import kernel_fixture

def test_pseudo_step_uses_local_temperature_and_pressure(tmp_path):
 fixture=kernel_fixture().replace('double* gasPseudoDt=nullptr;','ugkwp::MolecularViscosityView<double> gasViscosity; double* gasPseudoDt=nullptr;')
 compile_probe(tmp_path,fixture+r"""
int main(){DeviceState s;initialise(s);double steps[2]={};Real cp[2]={1000,1000},gasR[2]={300,300};s.gasPseudoDt=steps;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.heatCapacity=cp;s.gasSpecies.gasConstant=gasR;s.gasSpecies.diffusivity=nullptr;
ugkwp::PowerLawDiffusivity<double> laws[2]={{2,300,100000,2,-1},{1,300,100000,2,-1}};s.gasViscosity.diffusion=laws;s.gasViscosity.diffusionCount=2;
for(int f=0;f<3;++f){s.riemannBoundaryPFix[f]=0;s.riemannBoundaryTFix[f]=0;}
for(double T:{150.,300.,600.})for(double pressure:{50000.,100000.,200000.}){
 for(int c=0;c<2;++c){s.Tgas[c]=T;s.p[c]=pressure;s.gasFluxPositivityScale[c]=2;}
 for(int c=0;c<2;++c){threadIdx.x=c;prepareSteadyGasTimeStepKernel(&s,.5,1.);double D=2*(T/300)*(T/300)*100000/pressure;
 near(steps[c],.5/(2+2*(2*D)*(1+2)),"local diffusion pseudo timestep");}
}
}
""")
