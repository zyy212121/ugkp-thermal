"""Finite-bath root: gas-table bounds, affine derivative and nonzero exchange."""
import pytest
from test_mixture_state import compile_probe
from test_steady_mixture_particle_coupling import paired_fixture

@pytest.mark.parametrize("slope,particle,dt",[(-.22,10000,2e-5),(-.22,10000,100),(0,10000,2e-5),(-.22,2500,2e-5)])
def test_heat_equation_and_budget_independently(tmp_path,slope,particle,dt):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);
auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
for(int k=0;k<2;++k){coeff[3*k]=1000;coeff[3*k+1]=SLOPE;coeff[3*k+2]=0;}
s.gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;
s.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;
s.count=1;s.pStatus[1]=s.pStatus[2]=0;s.cellParticleOffset[1]=s.cellParticleOffset[2]=1;
s.pm[0]=1e-6;s.pT[0]=PARTICLE;s.TpMax=10000;s.pux[0]=s.puy[0]=s.puz[0]=0;
for(int c=0;c<2;++c){Real rhos[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
s.rhoUx[c]=s.rhoUy[c]=s.rhoUz[c]=0;
s.rhoE[c]=ugkwp::mixtureEnergy(rhos,Real(1000),s.gasSpecies.thermo);
s.Tgas[c]=1000;threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
const double oldE=s.rhoE[0],volume=s.epsGPrev[0]*s.V[0],oldT=s.pT[0];
const double a=-expm1(-6*2*.06/(2e6*.001*.001)*DT),capacity=s.pm[0]*800;
const long double cv=1000.L+SLOPE*1000.L-(long double)ugkwp::universalGasConstant<double>()/.028L;
const long double linear=cv+(long double)capacity*a/volume;
const long double rhs=(long double)capacity*a/volume*(oldT-1000.L);
const long double delta=2*rhs/(linear+sqrtl(linear*linear+2*SLOPE*rhs));
const long double expectedGas=1000.L+delta,expectedParcel=oldT+a*(expectedGas-oldT);
threadIdx.x=0;mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,DT,false);
ck(s.gasSpecies.cellStatus[0]==0,"heat root status");
closeValue(s.Tgas[0],expectedGas,2e-9,"gas root");closeValue(s.pT[0],expectedParcel,1e-8,"parcel cooling");
const long double gasHeat=(long double)volume*((long double)s.rhoE[0]-oldE);
const long double parcelHeat=(long double)capacity*((long double)s.pT[0]-oldT);
const long double tol=16*std::numeric_limits<double>::epsilon()*fabs(volume*oldE);
closeValue(gasHeat+parcelHeat,0,tol,"two-phase energy budget");
closeValue((long double)s.pT[0]-oldT,(long double)a*(s.Tgas[0]-oldT),1e-8,"exchange equation");
const long double actualDelta=(long double)s.Tgas[0]-1000.L;
closeValue(volume*(cv*actualDelta+.5L*SLOPE*actualDelta*actualDelta)+capacity*a*(s.Tgas[0]-oldT),0,tol,"caloric equation");
ck(parcelHeat<0 && gasHeat>0,"conserved zero exchange is not a solution");
ck(s.gasSpecies.rho[0]==.9 && s.gasSpecies.rho[2]==.1,"heat changed species inventories");
}
'''.replace('SLOPE',repr(slope)).replace('PARTICLE',repr(particle)).replace('DT',repr(dt)))


def test_affine_derivative_bounds_and_unconverged_status(tmp_path):
    compile_probe(tmp_path,paired_fixture()+r'''
int main(){PairedState s;setup(s);auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
for(int k=0;k<2;++k){coeff[3*k]=1000;coeff[3*k+1]=0;coeff[3*k+2]=0;}
Real rho[2]={.9,.1};auto controls=s.gasSpecies.thermoControls;
controls.maximumIterations=2;controls.relativeEnergyTolerance=1e-14;controls.relativeTemperatureTolerance=1e-14;
const double oldE=ugkwp::mixtureEnergy(rho,1000.,s.gasSpecies.thermo),A=1000,H=1e5;
double T=1000;auto status=ugkwp::invertMixtureHeatExchange(rho,oldE,A,H,s.gasSpecies.thermo,controls,T);
ck(status==ugkwp::ThermoStatus::Success,"one Newton correction must include linear capacity A");
closeValue(T,1000+H/(1000-ugkwp::universalGasConstant<double>()/.028+A),1e-10,"affine analytic root");
controls.maximumIterations=1;T=1000;
status=ugkwp::invertMixtureHeatExchange(rho,oldE,A,H,s.gasSpecies.thermo,controls,T);
ck(status==ugkwp::ThermoStatus::IterationLimit && T==1000,"unresolved root silently succeeded");
controls.maximumIterations=100;controls.maxTemperature=1001;
status=ugkwp::invertMixtureHeatExchange(rho,oldE,A,H,s.gasSpecies.thermo,controls,T);
ck(status!=ugkwp::ThermoStatus::Success && T==1000,"gas root escaped tightened domain");
ugkwp::MixtureHeatExchangeEvaluator<double,2,double> evaluator{{rho,s.gasSpecies.thermo},oldE,A,1000,100,3000};
double E,C;ck(evaluator(10000,E,C)==ugkwp::ThermoStatus::TemperatureOutOfRange,"gas calorics evaluated at parcel temperature");
}
''')
