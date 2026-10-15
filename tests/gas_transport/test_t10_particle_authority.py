"""Red/green actual-header coverage for T10 cache-plus-carry publication and consumers."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE = r'''
#include <cassert>
#include <cmath>
#include <limits>
#include <initializer_list>
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include "GpuPrecisionTypes.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
using Caloric=ugkwp::ParticleCaloric;
using State=ugkwp::ParticleThermalCarry<STORAGE>;
void near(double a,double b,double tol=1e-8){assert(std::abs(a-b)<tol);}
int main(){
 Caloric law; const STORAGE minT=200,maxT=4000;
 State s{300,0};
 const double q=law.enthalpyIncrement(300,3800);
 auto p=ugkwp::publishAffineRadiationThermalHeat(s,1.0,3800,minT,maxT,law);
 assert(p.valid&&!p.limited&&s.temperature==STORAGE(4000));
 near(p.acceptedSpecificHeat,q);near(law.enthalpy(double(s.temperature))+s.specificEnthalpyResidual,law.enthalpy(300)+q);
 near(ugkwp::effectiveParticleTemperature(s,law),4100);
 s={300,0};p=ugkwp::publishAffineRadiationThermalHeat(s,1.01,0,minT,maxT,law);
 assert(p.valid);near(p.acceptedSpecificHeat,law.enthalpyIncrement(300,3));
 s={300,0};p=ugkwp::publishAffineRadiationThermalHeat(s,-1,600,minT,maxT,law);
 assert(p.valid);near(ugkwp::effectiveParticleTemperature(s,law),300);
 for(double heat:{0.,-1.,1.,-200000.}){
  State outside{4100,0},old=outside;
  auto r=ugkwp::publishParticleThermalHeat(outside,heat,minT,maxT,law);
  assert(r.valid);
  const double expected=heat>0?0.:heat;
  near(r.acceptedSpecificHeat,expected);
  near((law.enthalpy(double(outside.temperature))-law.enthalpy(double(old.temperature)))+
       (outside.specificEnthalpyResidual-old.specificEnthalpyResidual),expected);
  if(heat==0){assert(outside.temperature==old.temperature&&outside.specificEnthalpyResidual==old.specificEnthalpyResidual);}
 }
 auto u=Foam::gpuThermal::radiationAffineTemperatureUpdate(10.,1.,1.,4100.,200.,4000.);
 assert(std::isfinite(u.scale)&&std::isfinite(u.offset));
 auto out=Foam::gpuThermal::applyRadiationAffineTemperature(300.,{-1.,600.});near(out,300.);
}
'''

def test_affine_full_publication_and_outside_increment(tmp_path):
    src = tmp_path/'t10.cpp'; src.write_text(SOURCE.replace('STORAGE', 'double'))
    exe = tmp_path/'t10'
    result = subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT/'common'), '-I'+str(ROOT), str(src), '-o', str(exe)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    result = subprocess.run([str(exe)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr

def test_paired_candidate_outside_old_state_keeps_small_inward_heat(tmp_path):
    from test_paired_heat_production import SOURCE as PAIR_SOURCE
    from test_mixture_state import HERE
    prefix = (HERE/'gas_state_probe.cpp').read_text().split('void hashBytes', 1)[0]
    source = PAIR_SOURCE.split('int main(){', 1)[0] + r'''
int main(){
 DeviceState s; s.count=1; s.TpMax=GpuReal(4000); s.pT[0]=GpuReal(4100);s.pThermalEnthalpyResidual[0]=0;
 s.couplingTgasOld[0]=GpuReal(4000);s.pm[0]=GpuReal(1);
 const auto p=pairedParticleHeatCandidate(s,0,1e-8);
 if(!(p.capacity>0 && p.specificHeat<0 && p.specificHeat>-100))std::abort();
 const auto law=particleCaloricDevice(s,0);
 const double ti=particleEffectiveTemperatureDevice(s,0);
 if(!(ti==4100))std::abort();
}
'''
    src=tmp_path/'candidate.cpp';src.write_text(prefix+source)
    exe=tmp_path/'candidate'
    cmd=['g++','-std=c++17','-O2','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(src),'-o',str(exe)]
    built=subprocess.run(cmd,text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr

CONTRIBUTION_SOURCE = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include "GpuPrecisionTypes.H"
#include "GpuParticleCaloric.H"
#define __device__
#define __global__
#define __shared__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
template<class T>T finiteOr(T x,T other){return std::isfinite(x)?x:other;}
template<class T>T clampMin(T x,T lower){return std::max(x,lower);}
template<class T>T clampRange(T x,T lower,T upper){return std::min(std::max(x,lower),upper);}
namespace Foam {namespace gpuThermal {constexpr int particleWallMobile=0,particleWallSolidifyingDeposition=1,particleWallColdWall2D=2;}}
struct Cold {GpuReal meltingTemperatureK=2327,mushyRangeK=20,latentHeatJkg=0,solidSpecificHeatJkgK=1000;};
struct DeviceState {
 int nCells=1,nFaces=1,particleCapacity=2,count=2,*particleCountDevice=&count;
 int pStatus[2]={1,1},pCellId[2]={},preBaseCellOffset[2]={0,2};
 int pStuck[2]={},pStuckFaceId[2]={-1,-1},compactPStuck[2]={},compactPStuckFaceId[2]={-1,-1},particleStuckCandidateMask[1]={};
 GpuReal pT[2]={4000,4000},compactPT[2]={},pm[2]={1,2},pd[2]={.001,.002},TpMin=200,TpMax=4000,particleDiameterFallback=.001;
 GpuReal radiationMobileMass[1]={},radiationMobileTemperatureMass[1]={},radiationMobileDiameterMass[1]={};
 double pThermalEnthalpyResidual[2]={},compactPThermalEnthalpyResidual[2]={};float pColdNodeSpecificEnthalpy[16]={},pCold2DNodeSpecificEnthalpy[128]={},compactPColdNodeSpecificEnthalpy[16]={},compactPCold2DNodeSpecificEnthalpy[128]={};
 Cold coldWallSolidificationParameters;
};
GpuReal particleSpecificEnthalpyDevice(GpuReal t){return GpuReal(1000)*t;}
#include "GpuParticleCaloricDevice.cuh"
#include "GpuRadiationContributions.cuh"
struct Dim {int x=0;};Dim blockIdx,threadIdx,blockDim{1},gridDim{1};
template<class T>void atomicAdd(T* p,T x){*p+=x;}
template<int N>void blockReduceComponentSums(GpuReal (&)[N],GpuReal*){}
GpuReal warpPartials[3]={};
KERNELS
int main(){
 DeviceState s;const ugkwp::ParticleCaloric law;
 s.pThermalEnthalpyResidual[0]=law.enthalpy(4100)-law.enthalpy(4000);
 s.pThermalEnthalpyResidual[1]=s.pThermalEnthalpyResidual[0];
 accumulateMobileParticleRadiationSumsAtomicKernel(&s);
 const double atomicMass=s.radiationMobileMass[0];
 const double atomicTemperature=s.radiationMobileTemperatureMass[0];
 s.radiationMobileMass[0]=s.radiationMobileTemperatureMass[0]=s.radiationMobileDiameterMass[0]=0;
 accumulatePackedMobileParticleRadiationSumsKernel(&s);
 const double packedMass=s.radiationMobileMass[0];
 const double packedTemperature=s.radiationMobileTemperatureMass[0];
 assert(std::abs(atomicTemperature/atomicMass-4000)<1e-3);
 assert(std::abs(packedTemperature/packedMass-4000)<1e-3);
 s.pT[0]=GpuReal(4100);s.pThermalEnthalpyResidual[0]=0;
 const double material=particleMaterialEnthalpyContribution(s,0);
 assert(std::abs(material-4000000)<1);
}
'''

def test_live_radiation_and_material_contributions(tmp_path):
    text=(ROOT/'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    def body(signature):
        start=text.index(signature)
        brace=text.index('{',start)
        depth=1;end=brace+1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        return text[start:end]
    kernels='\n'.join(body(s) for s in ('__global__ void accumulateMobileParticleRadiationSumsAtomicKernel',
                                          '__global__ void accumulatePackedMobileParticleRadiationSumsKernel'))
    src=tmp_path/'contributions.cpp';src.write_text(CONTRIBUTION_SOURCE.replace('KERNELS',kernels))
    exe=tmp_path/'contributions'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(src),'-o',str(exe)],text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr

def test_pressure_full_moment_uses_bounded_cache_and_adds_carry(tmp_path):
    fixture=(ROOT/'tests/fixtures/pressure/projection_cpu.cpp').read_text()
    fixture=fixture.split('void check(bool yes',1)[0]
    fixture=fixture.replace('    R pd[4]=', '    double pThermalEnthalpyResidual[4]={},compactPThermalEnthalpyResidual[4]={};\n    R pd[4]=',1)
    source=fixture+r'''
int main(){
 DeviceState s;s.TpMax=R(4000);s.pT[0]=R(4100);s.compactPT[0]=R(4100);
 PressureReal moment[7]={};int count=0;
 accumulatePressureParticleMomentsDevice<true,false>(s,0,0,moment,count);
 if(!(count==1 && std::abs(double(moment[6])-4000)<1e-3))std::abort();
 s.pT[0]=R(4000);s.pThermalEnthalpyResidual[0]=100;
 for(auto& x:moment)x=0;count=0;
 accumulatePressureParticleMomentsDevice<true,false>(s,0,0,moment,count);
 if(!(count==1 && std::abs(double(moment[6])-4100)<1e-3))std::abort();
 s.compactPT[0]=R(4100);
 for(auto& x:moment)x=0;count=0;
 accumulatePressureParticleMomentsDevice<true,true>(s,0,0,moment,count);
 if(!(count==1 && std::abs(double(moment[6])-4000)<1e-3))std::abort();
}
'''
    src=tmp_path/'pressure.cpp';src.write_text(source)
    exe=tmp_path/'pressure'
    built=subprocess.run(['g++','-std=c++17','-O2','-DTEST_REAL=double',
                          '-I'+str(ROOT/'common'),str(src),'-o',str(exe)],text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr

@pytest.mark.parametrize('mixed_energy', [True])
def test_whole_recovery_wrapper_gas_particle_and_initializer(tmp_path,mixed_energy):
    from test_full_heat_advance import cache_path_fixture
    from test_mixture_state import HERE
    fixture=cache_path_fixture().split('int main(int argc',1)[0]
    fixture=fixture.replace('struct DeviceState:State{','struct DeviceState:State{\n Real compactPT[2]={};double pThermalEnthalpyResidual[2]={},compactPThermalEnthalpyResidual[2]={};',1)
    if not mixed_energy:
        fixture=fixture.replace('GasStateView<Real,Time,ugkwp::SstCoefficients,double>',
                                'GasStateView<Real,Time,ugkwp::SstCoefficients>')
        fixture=fixture.replace('s.rhoE=new double[8]{};', 's.rhoE=new Real[8]{};')
        fixture=fixture.replace('s.rhoENext=new double[8]{};', 's.rhoENext=new Real[8]{};')
    source=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]+fixture+r'''
int main(){
 static_assert(sizeof(std::remove_reference_t<decltype(DeviceState{}.rhoE[0])>)==ENERGY_BYTES);
 blockDim.x=1;gridDim.x=1;threadIdx.x=0;blockIdx.x=0;
 for(int mode=0;mode<3;++mode){
  DeviceState s;setup(s);s.rhoMin=Real(1e-6);s.TpMin=Real(200);s.TpMax=Real(4000);
  s.rho[0]=Real(1);s.rhoUx[0]=Real(.25);s.rhoUy[0]=Real(.1);s.rhoUz[0]=Real(0);
  s.rhoE[0]=double(s.rho[0])*double(s.Rgas)*300/double(s.gammaGas-Real(1))+.5*(.25*.25+.1*.1);
  s.epsS[0]=mode==2?Real(0):Real(.01);
  const Real mass=s.epsS[0]*s.rhoSolid;
  s.rhoUsx[0]=mass*Real(2);s.rhoUsy[0]=mass*Real(-1);s.rhoUsz[0]=0;
  s.rhoEs[0]=mass*(Real(2.5)+Real(1.5)*Real(.4));s.rhoDs[0]=mass*Real(.002);
  s.solveParticleTemperature=mode==0?1:0;
  s.Tp[0]=Real(300);s.rhoHp[0]=mass*particleSpecificEnthalpyDevice(Real(300));
  s.pT[0]=Real(4000);s.pThermalEnthalpyResidual[0]=123.25;
  recoverGasPrimitiveCell(s,0);
  const auto energy=s.rhoE[0];const auto temperature=s.Tgas[0];const auto pressure=s.p[0];const auto velocity=s.Ux[0];
  recoverPrimitivesKernel(&s);
  ck(s.rhoE[0]==energy && s.Tgas[0]==temperature && s.p[0]==pressure && s.Ux[0]==velocity,
     "wrapper changed T06 gas result");
  ck(s.pT[0]==Real(4000) && s.pThermalEnthalpyResidual[0]==123.25,"wrapper reset particle authority");
  if(mode==0)ck(s.Tp[0]==Real(300) && s.rhoHp[0]>0 && s.Usx[0]==Real(2) && s.theta[0]>0,
                "populated thermal recovery omitted particle outputs");
  if(mode==1)ck(s.Tp[0]==s.TpMin && s.rhoHp[0]==0 && s.Usx[0]==Real(2),
                "thermal-off recovery omitted mechanical particle outputs");
  if(mode==2)ck(s.epsS[0]==0 && s.rhoHp[0]==0 && s.Usx[0]==0 && s.Tp[0]==s.TpMin,
                "empty-solid branch did not clear derived state");
  if(mode==0){
   const double expected=double(s.pm[0])*(double(particleSpecificEnthalpyDevice(s.pT[0]))
     +s.pThermalEnthalpyResidual[0]);
   publish(s);
   ck(std::abs(double(s.momRhoHpP[0])-expected)<256.,"atomic material moment omitted live carry");
   s.pT[0]=Real(4100);s.pThermalEnthalpyResidual[0]=0;
   const double rawExpected=double(s.pm[0])*double(particleSpecificEnthalpyDevice(s.TpMax));
   publish(s);
   ck(std::abs(double(s.momRhoHpP[0])-rawExpected)<256.,"atomic moment changed existing bounded material cache");
  }
  const auto raw=s.pT[0];const auto carry=s.pThermalEnthalpyResidual[0];
  initialiseParticleMaterialEnthalpyKernel(&s);
  ck(s.pT[0]==raw && s.pThermalEnthalpyResidual[0]==carry,
     "distinct initializer reset persistent particle state");
 }
}
'''.replace('ENERGY_BYTES','8' if mixed_energy else '4')
    src=tmp_path/'wrapper.cpp';src.write_text(source)
    exe=tmp_path/'wrapper'
    built=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),
                          str(src),'-o',str(exe)],text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr

def test_large_finite_gas_energy_reaches_paired_acceptance(tmp_path):
    from test_paired_heat_production import SOURCE as PAIR_SOURCE
    from test_mixture_state import HERE
    prefix=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]
    source=PAIR_SOURCE.split('int main(){',1)[0]+r'''
int main(){
 DeviceState s;s.count=1;s.rho[0]=GpuReal(1);s.rhoUx[0]=GpuReal(1e151);s.rhoUy[0]=0;s.rhoUz[0]=0;
 const double kinetic=ugkwp::gasLegacyKineticEnergy(1.,1e151,0.,0.);
 const double floor=pairedGasEnergyFloor(s,0);
 if(!(std::isfinite(floor)&&std::abs(floor/kinetic-1.)<1e-14))std::abort();
 s.rhoE[0]=floor+1e290;s.rhoENext[0]=1.;s.particleHeatCoefficient[0]=.5;
 s.particleHeatBudget[0]=1e290;const double before=s.rhoE[0];
 acceptPairedParticleHeatKernel(&s);
 if(!(std::isfinite(s.particleHeatCoefficient[0])&&s.particleHeatCoefficient[0]>0.
   &&s.particleHeatCoefficient[0]<=1.&&s.rhoE[0]==before&&s.particleHeatBudget[0]==0.))std::abort();
}
'''
    src=tmp_path/'large.cpp';src.write_text(prefix+source)
    exe=tmp_path/'large'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),
                          str(src),'-o',str(exe)],text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr

REPAIR_SOURCE = r'''
#include <cassert>
#include <cmath>
#include <limits>
#include "GpuPrecisionTypes.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
#include "GpuParticleThermalConsumers.H"
#include "GpuParticleCaloric.H"
using namespace Foam::gpuThermal;
using Real=GpuReal;
void close(double x,double y,double tolerance=1e-10){assert(std::isfinite(x)&&std::abs(x-y)<=tolerance*std::max(1.,std::abs(y)));}
struct CountingLaw {
 mutable int enthalpyCalls=0;
 ugkwp::ParticleCaloric law;
 double enthalpy(double t) const {++enthalpyCalls;return law.enthalpy(t);}
 double temperature(double h) const {return law.temperature(h);}
 double enthalpyIncrement(double t,double dt) const {return law.enthalpyIncrement(t,dt);}
};
int main(){
 const Real tiny=std::numeric_limits<Real>::denorm_min();
 const Real max=std::numeric_limits<Real>::max();
 assert(finiteRadiationMathValue(Real(1e31)));
 close(radiationDeltaTemperature(Real(1e31),1,1),double(Real(1e31)));
 auto signedAffine=radiationAffineTemperatureUpdate(10,1,-1,300,200,4000);
 const Real signedCapacity=Real(-1);
 const Real signedFraction=Real(10)/(signedCapacity*(Real(4000)-Real(300)));
 close(signedAffine.scale,Real(1)-signedFraction);
 close(signedAffine.offset,signedFraction*Real(4000));
 close(applyRadiationAffineTemperature(300,signedAffine),290,1e-5);
 auto negativeMass=radiationAffineTemperatureUpdate(10,-1,1,300,200,4000);
 close(negativeMass.scale,signedAffine.scale);close(negativeMass.offset,signedAffine.offset);
 auto reversed=radiationAffineTemperatureUpdate(10,1,1,300,4000,200);
 close(reversed.scale,1.1);close(reversed.offset,-20);
 close(applyRadiationAffineTemperature(300,reversed),310,1e-5);
 auto zeroEnergy=radiationAffineTemperatureUpdate(0,1,0,300,200,4000);
 assert(zeroEnergy.scale==1&&zeroEnergy.offset==0);
 close(radiationDeltaTemperature(10,1,-1),-10);
 close(radiationDeltaTemperature(10,-1,1),-10);
 assert(radiationDeltaTemperature(tiny,1,2)==0);
 close(radiationParticleEnergyChange(1,-1,10),-10);
 close(radiationParticleEnergyChange(-1,1,10),-10);
 assert(radiationParticleEnergyChange(tiny,1,Real(.5))==0);
 assert(radiationParticleEnergyChange(1,0,10)==0);
 close(applyRadiationAffineTemperature(0,{1e39,1}),1);
 assert(std::isnan(radiationAffineTemperatureUpdate(10,1,1,4000,200,4000).scale));
 assert(std::isnan(radiationDeltaTemperature(1,1,0)));
 assert(std::isnan(radiationDeltaTemperature(1,max,2)));
 assert(std::isnan(radiationParticleEnergyChange(max,2,10)));
 CountingLaw law;
 ugkwp::ParticleThermalCarry<double> state{300,0};
 auto publication=ugkwp::publishAffineRadiationThermalHeat(state,1,10,200.,1e308,law);
 assert(!publication.valid&&state.temperature==300&&state.specificEnthalpyResidual==0);
 law.enthalpyCalls=0;
 publication=ugkwp::publishAffineRadiationThermalHeat(state,1.01,7,200.,1e308,law);
 assert(publication.valid&&!publication.limited);
 close(ugkwp::effectiveParticleEnthalpy(state,law),law.law.enthalpy(300)+law.law.enthalpyIncrement(300,10));
 assert(law.enthalpyCalls==4);
 law.enthalpyCalls=0;state={300,0};
 auto identity=ugkwp::publishAffineRadiationThermalHeat(state,1,0,200.,1e308,law);
 assert(identity.valid&&state.temperature==300&&state.specificEnthalpyResidual==0);
 assert(law.enthalpyCalls<=1);
 law.enthalpyCalls=0;state={300,0};
 auto reversePublication=ugkwp::publishAffineRadiationThermalHeat(state,1,10,4000.,200.,law);
 assert(reversePublication.valid&&state.temperature==200);
 close(ugkwp::effectiveParticleEnthalpy(state,law),law.law.enthalpy(300)+law.law.enthalpyIncrement(300,10));
 law.enthalpyCalls=0;state={300,0};
 auto ordinary=ugkwp::publishParticleThermalHeat
   (state,law.law.enthalpyIncrement(300,10),200.,4000.,law);
 assert(ordinary.valid&&law.enthalpyCalls==4); // Ordinary limiter still consumes both bounds.
}
'''

def test_radiation_finite_producers_and_dead_bound_work(tmp_path):
    src=tmp_path/'radiation_repair.cpp';src.write_text(REPAIR_SOURCE)
    exe=tmp_path/'radiation_repair'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT),str(src),'-o',str(exe)],
                         text=True,capture_output=True)
    assert built.returncode==0,built.stdout+built.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
