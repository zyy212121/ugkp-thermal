"""Fixed impermeable mixture SST walls exercise the production flux and timestep."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def wall_fixture():
    return fixture().replace('gasSpecies;};', 'gasSpecies; Real gasThermalConductivity=-1;};') + r'''
void wall(State& s){
 initialise(s);s.turbulenceModel=3;s.sstConfigured=1;s.sstWallTreatment=1;
 s.gasMu=.001;s.turbulentPrandtl=.9;s.sstCoefficients=ugkwp::defaultSstCoefficients();
 s.sstKMin=1e-12;s.sstOmegaMin=1e-8;
 s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
 auto* meta=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);
 auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
 meta[1].molarMass=.04;coeff[0]=1000;coeff[1]=.2;coeff[3]=1500;coeff[4]=.2;
 for(int c=0;c<2;++c){Real masses[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
 s.rhoE[c]=ugkwp::mixtureEnergy(masses,Real(700),s.gasSpecies.thermo);
 threadIdx.x=c;recoverGasPrimitivesKernel(&s);s.sstWallDistance[c]=.5;s.k[c]=.1;s.omega[c]=10;
 initialiseSstConservativeStateKernel(&s);}
 s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=500;
 s.riemannBoundaryUFix[1]=1;s.riemannBoundaryUy[1]=2;s.Uy[0]=12;
 s.gasCp=7;s.Rgas=700; // deliberate scalar bootstrap contamination
 // Constant-Pr P/y+ came from initialiseSstConservativeStateKernel.
}
void near(Real a,Real b,const char* message){ck(std::abs(a-b)<Real(3e-5)*std::max(Real(1),std::abs(b)),message);}
'''


@pytest.mark.parametrize('direct_conductivity',[False,True])
@pytest.mark.parametrize('kinetic',[0.,1e-12,.1])
def test_wall_thermo_flux_and_diffusion_use_same_mixture_properties(tmp_path,direct_conductivity,kinetic):
    compile_probe(tmp_path,wall_fixture()+f'\nint main(){{State s;wall(s);s.k[0]=Real({kinetic});s.gasThermalConductivity=Real({.31 if direct_conductivity else -1});'+r'''
 if(ugkwp::gasHasDirectConductivity(s)){s.sstJayatillekeP=123;s.sstThermalYPlus=456;} // direct-conductivity must not reuse constant-Pr cache
 const Real cp=Real(.9)*1100+Real(.1)*1600;
 const Real R=ugkwp::universalGasConstant<Real>()*(Real(.9)/Real(.028)+Real(.1)/Real(.04));
 const Real rho=s.p[0]/(R*500);
 const Real molecular=ugkwp::gasHasDirectConductivity(s)?s.gasThermalConductivity:s.gasMu*cp/s.gasPrClamped;
 const Real Pr=s.gasMu*cp/molecular,ratio=Pr/s.turbulentPrandtl;
 const auto expected=ugkpwall::sstJayatillekeThermalTransport(rho,cp,s.gasMu,Pr,s.turbulentPrandtl,
 s.sstWallCmu,s.sstWallKappa,s.sstWallE,ugkpwall::jayatillekeSmoothP(ratio),
 ugkpwall::jayatillekeThermalYPlus(ratio,s.sstWallKappa,s.sstWallE),s.k[0],Real(.5),Real(10),Real(2),Real(-400));
 Real mu=0,conductivity=0,q=0;int active=0;
 gasFaceSubgridTransportProperties(s,1,0,-1,2,rho,mu,conductivity,q,active);
 ck(active==1,"mixture thermal wall not active");near(q,expected.heatFlux,"wall heat uses legacy or owner thermodynamics");
 const Real baseline=ugkwp::gasHasDirectConductivity(s)?s.gasThermalConductivity:s.gasMu*s.gasSpecies.heatCapacity[0]/s.gasPrClamped;
 near(baseline+conductivity,expected.conductivity,"flux/stability conductivity inconsistent with wall closure");
 s.cellPlaneCount[0]=1;s.cellPlaneStart[0]=1;
 // Direct physical wall flux carries heat and work once; species/mass remain zero.
 Real mass,mx,my,mz,energy;
 ck(computeRiemannGasFaceFluxDevice<true>(s,1,mass,mx,my,mz,energy),"mixture wall flux rejected");
 ck(s.gasSpecies.faceStatus[1]==0,"wall face status rejected");
 near(mass,0,"wall leaks bulk mass");near(s.gasSpecies.flux[1],0,"wall leaks species A");near(s.gasSpecies.flux[4],0,"wall leaks species B");
 near(energy-2*my,expected.heatFlux,"heat and tangential wall work inconsistent");
}
''')


def test_existing_wall_capability_dispatch_keeps_unavailable_particle_model(tmp_path):
    compile_probe(tmp_path,r'''
int main(){ugkwp::GasCapabilityRequest r;r.mode=ugkwp::GasMode::MixtureFrozen;r.turbulenceModel=3;r.sstWallTreatment=1;
if(!ugkwp::validateGasCapabilities(r))return 1;
r.movingGeometry=true;if(!ugkwp::validateGasCapabilities(r))return 2;r.movingGeometry=false;
r.mode=ugkwp::GasMode::MixtureChemistry;if(!ugkwp::validateGasCapabilities(r))return 3;
r.mode=ugkwp::GasMode::MixtureFrozen;r.particleCoupling=true;if(ugkwp::validateGasCapabilities(r))return 4;}
''')


@pytest.mark.parametrize('invalid',['s.riemannBoundaryT[1]=4000;', 's.riemannBoundaryT[1]=0;', 's.riemannBoundaryT[1]=-1;', 's.riemannBoundaryT[1]=std::numeric_limits<Real>::quiet_NaN();', 's.gasSpecies.rho[0]=-1;s.gasSpecies.rho[2]=2;'])
def test_wall_thermodynamics_physical_errors_stop_without_trap(tmp_path,invalid):
    compile_probe(tmp_path,wall_fixture()+'\nint main(){State s;wall(s);'+invalid+r'''
Real mu=0,k=0,q=0;int active=0;
gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active);
ck(EXPECTED,"wall thermo arithmetic status differs");}
'''.replace('EXPECTED','s.gasSpecies.faceStatus[1]==0&&std::isfinite(mu)&&std::isfinite(k)&&std::isfinite(q)' if invalid in ['s.riemannBoundaryT[1]=4000;','s.gasSpecies.rho[0]=-1;s.gasSpecies.rho[2]=2;'] else 's.gasSpecies.faceStatus[1]!=0'))


def test_adiabatic_wall_and_unvalidated_device_modes(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){State s;wall(s);s.riemannBoundaryTFix[1]=0;s.riemannBoundaryUy[1]=0;
Real mass,mx,my,mz,energy;
ck(computeRiemannGasFaceFluxDevice<true>(s,1,mass,mx,my,mz,energy),"adiabatic frozen wall rejected");
near(energy,0,"adiabatic stationary wall gained heat");
s.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;
ck(computeRiemannGasFaceFluxDevice<true>(s,1,mass,mx,my,mz,energy),"implemented chemistry wall transport refused");
ck(s.gasSpecies.faceStatus[1]==0&&std::isfinite(energy),"chemistry wall transport incomplete");}
''')


@pytest.mark.parametrize('strict',[False,True])
def test_zero_wall_viscosity_diagnostic_stops_wall_algebra(tmp_path,strict):
    compile_probe(tmp_path,wall_fixture()+r'''
#include <cfenv>
int main(){State s;wall(s);s.gasMu=0;
Real mu=-77,k=-77,q=-77;int active=0;bool evaluated=false;
if(STRICT)feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,STRICT,&evaluated);
ck(s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidThermodynamics),"zero-viscosity diagnostic missing");
ck(!evaluated&&mu==0&&k==0&&q==0&&active==0,"zero-viscosity path evaluated wall algebra");}
'''.replace('STRICT','true' if strict else 'false'))
