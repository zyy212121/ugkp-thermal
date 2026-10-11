"""Shared mixture state/recovery contract; actual existing common kernels."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def compile_probe(tmp_path,body,bits=64):
    legacy=(HERE/'gas_state_probe.cpp').read_text()
    pre=legacy[:legacy.index('void hashBytes')]
    source=tmp_path/'mixture.cpp';source.write_text(pre+body)
    exe=tmp_path/'mixture'
    result=subprocess.run(['g++','-std=c++17','-O0',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr

@pytest.mark.parametrize('bits',[32,64])
def test_mixture_recovery_is_conservative_and_failed_outputs_unchanged(tmp_path,bits):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasCapabilities.H"
struct State : ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients> {
 ugkwp::GasSpeciesState<Real,2> gasSpecies;
};
void ck(bool a,const char*m){if(!a){std::fprintf(stderr,"%s\n",m);std::abort();}}
int main(){
 State s;static_assert(ugkwp::GasStateTraits<State>::speciesCount==2,"traits");
 static_assert(ugkwp::GasStateTraits<LegacyGasFixture>::speciesCount==0,"legacy no field");
 Real rho[1]={2},mx[1]={6},my[1]={-2},mz[1]={1},energy[1],ux[1]={-7},uy[1]={-7},uz[1]={-7},p[1]={-7},T[1]={-7};
 Real species[2]={.5,1.5},a[1]={-7},cp[1]={-7},R[1]={-7};int status[1]={0};
 ugkwp::SpeciesThermoData<Real> metadata[2];Real coefficients[6]={1000,0,-1e6,1100,.1,2e6};
 for(int k=0;k<2;++k){metadata[k].molarMass=k?.028:.032;metadata[k].minTemperature=100;metadata[k].maxTemperature=3000;metadata[k].referencePressure=101325;metadata[k].coefficientOffset=3*k;}
 s.gasSpecies.thermo.species=metadata;s.gasSpecies.thermo.coefficients=coefficients;s.gasSpecies.thermo.coefficientCount=6;
 s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.rho=species;s.gasSpecies.soundSpeed=a;s.gasSpecies.heatCapacity=cp;s.gasSpecies.gasConstant=R;s.gasSpecies.cellStatus=status;
 s.gasSpecies.thermoControls.relativeEnergyTolerance=Real(1e-6);s.gasSpecies.thermoControls.relativeTemperatureTolerance=Real(1e-6);
 s.nCells=1;s.rho=rho;s.rhoUx=mx;s.rhoUy=my;s.rhoUz=mz;s.rhoE=energy;s.Ux=ux;s.Uy=uy;s.Uz=uz;s.p=p;s.Tgas=T;s.rhoMin=1e-12;s.TgasMin=1e-9;
 const Real targetT=700;energy[0]=ugkwp::mixtureEnergy(species,targetT,s.gasSpecies.thermo)+(mx[0]*mx[0]+my[0]*my[0]+mz[0]*mz[0])/(2*rho[0]);
 Real saved[]={rho[0],mx[0],my[0],mz[0],energy[0],species[0],species[1]};
 blockDim.x=1;threadIdx.x=blockIdx.x=0;recoverGasPrimitivesKernel(&s);
 ck(status[0]==0,"valid mixture recovery rejected");ck(std::abs(T[0]-targetT)<Real(.005),"temperature inversion");
 ck(ux[0]==3&&uy[0]==-1&&uz[0]==Real(.5),"momentum velocity");ck(a[0]>0&&cp[0]>R[0]&&p[0]>0,"derived EOS fields");
 ck(std::abs(p[0]-rho[0]*R[0]*T[0])<Real(.02),"ideal mixture pressure");
 Real after[]={rho[0],mx[0],my[0],mz[0],energy[0],species[0],species[1]};ck(std::memcmp(saved,after,sizeof(saved))==0,"recovery altered conserved state");
 const Real oldOutputs[]={ux[0],uy[0],uz[0],p[0],T[0],a[0],cp[0],R[0]};species[1]=-1;status[0]=0;recoverGasPrimitivesKernel(&s);
 Real failedOutputs[]={ux[0],uy[0],uz[0],p[0],T[0],a[0],cp[0],R[0]};ck(status[0]!=0,"negative species accepted");ck(std::memcmp(oldOutputs,failedOutputs,sizeof(oldOutputs))==0,"failed recovery published outputs");
 species[1]=1.5;energy[0]=Real(-1e20);status[0]=0;recoverGasPrimitivesKernel(&s);ck(status[0]!=0,"out of range energy accepted");ck(T[0]==oldOutputs[4],"failed inversion changed temperature");
}
''',bits)

def test_mixture_capabilities_reject_unproved_combinations(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasCapabilities.H"
int main(){ugkwp::GasCapabilityRequest r;
for(int scheme=1;scheme<=9;++scheme){r.fluxScheme=scheme;if(!ugkwp::validateGasCapabilities(r))return 1;}
r.mode=ugkwp::GasMode::MixtureFrozen;r.fluxScheme=1;if(!ugkwp::validateGasCapabilities(r))return 2;
r.fluxScheme=4;if(ugkwp::validateGasCapabilities(r))return 3;r.fluxScheme=1;
r.reconstruction=2;if(ugkwp::validateGasCapabilities(r))return 4;r.reconstruction=0;
r.movingGeometry=true;r.timeIntegrator=2;if(ugkwp::validateGasCapabilities(r))return 5;r.timeIntegrator=1;if(!ugkwp::validateGasCapabilities(r))return 9;r.movingGeometry=false;
r.particleCoupling=true;if(ugkwp::validateGasCapabilities(r))return 6;r.particleCoupling=false;
r.turbulenceModel=3;r.sstWallTreatment=1;if(!ugkwp::validateGasCapabilities(r))return 7;r.turbulenceModel=0;r.sstWallTreatment=0;
r.mode=ugkwp::GasMode::MixtureChemistry;if(!ugkwp::validateGasCapabilities(r))return 8;}
''')

@pytest.mark.parametrize('bits',[32,64])
def test_general_rusanov_species_dissipation_transports_formation_energy(tmp_path,bits):
    compile_probe(tmp_path,r'''
int main(){
using namespace ugkpriemann;Primitive l{1,0,0,0,100000},r=l;
Conservative ul{{1,0,0,0,250000-1000000}},ur{{1,0,0,0,250000+2000000}};
const Real a=std::sqrt(Real(1.4)*l.p/l.rho);
auto f=rusanovTadmorFluxUnitNormal(l,ul,a,r,ur,a,Real(1),Real(0),Real(0),false);
Real fa=rusanovConservativeComponentFlux(Real(0),Real(0),Real(1),Real(0),a);
Real fb=rusanovConservativeComponentFlux(Real(0),Real(0),Real(0),Real(1),a);
if(!f.valid||f.flux[0]!=0||fa==0||fa+fb!=0)return 1;
if(std::abs(f.flux[4]-(Real(-1e6)*fa+Real(2e6)*fb))>Real(1e-6)*std::abs(f.flux[4]))return 2;
// Same gamma and zero offsets reduce to the original flux exactly.
ul=conservative(l,Real(1.4));ur=conservative(r,Real(1.4));
auto old=rusanovTadmorFluxUnitNormal(l,r,Real(1),Real(0),Real(0),Real(1.4),false);
auto now=rusanovTadmorFluxUnitNormal(l,ul,a,r,ur,a,Real(1),Real(0),Real(0),false);
for(int k=0;k<5;++k)if(old.flux[k]!=now.flux[k])return 3;
}
''',bits)
