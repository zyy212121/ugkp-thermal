"""Compile the production steady reduction/time-step kernels on the host."""
from test_mixture_state import compile_probe, ROOT
from test_legacy_ale_transport import fixture


def kernel_fixture():
    return fixture()+r'''
#include "gasTransport/GasBuildConfig.H"
struct DeviceState:State {
 double* gasPseudoDt=nullptr;
 double* gasSteadyResidual=nullptr;
 Real gravityX=0,gravityY=0,gravityZ=0,gasPr=.7;
};
#include "STEADY_KERNEL_HEADER"
'''.replace('STEADY_KERNEL_HEADER',str(ROOT/'applications/gasUGKPSteady/private_backend/SteadyGasKernels.cuh'))


def test_raw_residual_uses_volume_not_pseudo_time_and_includes_gravity(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double steps[2]={1e-12,1e3},r[18]={};s.gasPseudoDt=steps;s.gasSteadyResidual=r;
s.V[0]=2;s.V[1]=5;s.gasPhiRho[0]=3;s.gasPhiRhoUx[0]=7;s.gasPhiRhoE[0]=11;
for(int c=0;c<2;++c){threadIdx.x=c;steadyGasSpatialResidualKernel(&s);near(r[c],3/s.V[c],"mass residual volume");near(r[2+c],7/s.V[c],"momentum residual volume");near(r[8+c],11/s.V[c],"energy residual volume");}
near(r[0]*s.V[0],r[1]*s.V[1],"interior equal/opposite flux magnitudes");
double saved[18];std::memcpy(saved,r,sizeof(r));steps[0]=7;steps[1]=2;
for(int c=0;c<2;++c){threadIdx.x=c;steadyGasSpatialResidualKernel(&s);}
ck(std::memcmp(saved,r,sizeof(r))==0,"raw residual changed with pseudo time");
s.gravityX=.2;s.gravityY=-.3;s.gravityZ=.4;
for(int c=0;c<2;++c){s.rhoUx[c]=3;s.rhoUy[c]=4;s.rhoUz[c]=5;threadIdx.x=c;steadyGasSpatialResidualKernel(&s);double sign=c==0?-1:1;
near(r[2+c],fabs(sign*7/s.V[c]+s.rho[c]*.2),"momentum includes gravity");near(r[4+c],fabs(s.rho[c]*-.3),"y gravity");near(r[8+c],fabs(sign*11/s.V[c]+3*.2-4*.3+5*.4),"energy includes work");}
}
''')


def test_sst_residual_omits_constrained_wall_omega_only(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double r[18]={};s.gasSteadyResidual=r;s.sstConfigured=1;s.sstWallTreatment=0;
s.sstPhiRhoK[0]=.4;s.sstPhiRhoOmega[0]=.5;s.riemannBoundaryKind[1]=2;
for(int c=0;c<2;++c){threadIdx.x=c;steadyGasSpatialResidualKernel(&s);double k=0,o=0;ck(sstSourcesForCell(s,c,0.,k,o),"SST source");double sign=c==0?-1:1;
near(r[14+c],fabs(sign*.4/s.V[c]+k),"k physical residual");near(r[16+c],c==0?0:fabs(.5/s.V[c]+o),"only constrained omega omitted");}
}
''')


def test_pseudo_step_responds_to_cell_rate_and_cap(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double steps[2]={};s.gasPseudoDt=steps;s.gasFluxPositivityScale[0]=2;s.gasFluxPositivityScale[1]=8;
for(int c=0;c<2;++c){threadIdx.x=c;prepareSteadyGasTimeStepKernel(&s,.5,.1);near(steps[c],c==0?.1:.0625,"local convective rate/cap");}
s.sstConfigured=1;
for(int c=0;c<2;++c){threadIdx.x=c;prepareSteadyGasTimeStepKernel(&s,.5,1.);near(steps[c],.5/(s.gasFluxPositivityScale[c]+s.omega[c]),"SST source dt rate");}
}
''')


def test_pseudo_step_includes_largest_species_diffusivity(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double steps[2]={};Real cp[2]={1000,1000},gasR[2]={300,300},D[2]={.1,2};s.gasPseudoDt=steps;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.heatCapacity=cp;s.gasSpecies.gasConstant=gasR;s.gasSpecies.diffusivity=D;
for(int c=0;c<2;++c){s.gasFluxPositivityScale[c]=2;threadIdx.x=c;prepareSteadyGasTimeStepKernel(&s,.5,1.);near(steps[c],.5/(2+2*(2*2)*(1+2)),"largest species diffusion with correction");}
}
''')


def test_pseudo_step_neighbor_cell_uses_face_owner_eddy_viscosity(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double steps[2]={};s.gasPseudoDt=steps;s.nut[0]=3;s.nut[1]=1;
// For the neighbour row, the interior face still sees max(owner,neighbour).
threadIdx.x=1;prepareSteadyGasTimeStepKernel(&s,.5,1.);
const double thermalRatio=s.gasCp/((s.gasCp-s.Rgas)*s.turbulentPrandtl);
const double transport=fmax(4./3.,thermalRatio);
near(steps[1],.5/(2*transport*(3+2*1)),"neighbour omitted owner nut");
}
''')


def test_species_residual_is_raw_conservative_flux_over_volume(tmp_path):
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double r[18]={};Real flux[6]={.1,0,0,-.1,0,0};s.gasSteadyResidual=r;s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.flux=flux;s.V[0]=2;s.V[1]=5;
for(int c=0;c<2;++c){threadIdx.x=c;steadyGasSpatialResidualKernel(&s);near(r[10+c],.1/s.V[c],"species 0 raw residual");near(r[12+c],.1/s.V[c],"species 1 raw residual");}
}
''')
