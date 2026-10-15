"""Complete real wall packets; preserve missing closure and recompute stale reuse."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
APP=(SOURCE_ROOT / 'applications/CHMT');ROOT=APP.parents[1]
SOURCE=r'''
#include "tests/TestSupport.H"
#include "gpu/GasWallMath.H"
#include <cassert>
using namespace chmt;
int main(){auto p=chmt_test::physics();p.gasMode=ugkwp::GasMode::MixtureFrozen;
p.gasViscosity=.02;p.gasConductivity=80;
for(int s=0;s<Ns;++s){p.species[s].cp1=0;p.gasDiffusivity[s]=.01;}
p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
auto bulk=chmt_test::gas(p);GasWallInput in;in.bulk=bulk;in.temperature=600;
in.gasDistance=.2;in.area=in.gasArea=2;in.dt=.01;in.normal={1,0,0};in.velocity={0,2,0};
in.normalSpeed=.25;in.sweptVolume=.005;in.speciesRate[0]=.03;in.speciesRate[1]=.01;
in.poreRate[0]=.007;in.poreRate[1]=.003;in.condensedRate[0]=.04;
in.wallContext.ownerDistance=.2;in.wallContext.matchingDistance=.8;
in.wallContext.matchingPressure=bulk.pressure;in.wallContext.mechanicalPressure=1.7*bulk.pressure;
in.wallContext.matching.temperature=bulk.temperature;in.wallContext.matching.velocity[1]=10;
for(int s=0;s<Ns;++s)in.wallContext.matching.massFraction[s]=bulk.Y[s];
SurfacePacketIdentity id;id.gasCell=id.solidCell=id.filmFace=0;
GasWallResult original;assert(evaluateGasWall(in,id,p,original));
ugkwp::gaswall::WallOutput<Real,Ns> prepared=original.layer;(void)prepared;
const auto sampled=in.wallContext.matching;const Real sampledPressure=in.wallContext.matchingPressure;
in.preparedMatching=&sampled;in.preparedMatchingPressure=&sampledPressure;
MUTATION
GasWallResult out;out.primary.energy=917;out.pore.energy=918;out.primary.condensed[0]=919;
const bool ok=evaluateGasWall(in,id,p,out);
if(MISSING){assert(!ok&&out.primary.energy==917);return 0;}
assert(ok);auto same=[](Real a,Real b){return a==b||(std::isnan(a)&&std::isnan(b));};
assert(out.primary.species[0]==.02*in.speciesRate[0]&&out.primary.condensed[0]==.02*in.condensedRate[0]);
assert(out.pore.species[1]==.02*in.poreRate[1]);
assert(same(out.primary.energy,out.primary.advective+out.primary.conductive+out.primary.pressureWork+out.primary.viscousWork));
assert(same(out.primary.pressureWork,in.wallContext.mechanicalPressure*in.sweptVolume));
assert(same(out.primary.momentum.x,out.trace.velocity.x*out.primary.mass+(in.normal.x*in.wallContext.mechanicalPressure-out.traction.x)*.02));
assert(out.trace.temperature==in.temperature&&out.trace.pressure==in.wallContext.matchingPressure);
if(STALE){GasWallResult fresh;in.preparedLayer=nullptr;assert(evaluateGasWall(in,id,p,fresh));
assert(out.layer.traceTemperature==fresh.layer.traceTemperature&&out.primary.conductive==fresh.primary.conductive);
assert(out.primary.conductive!=original.primary.conductive||out.trace.rho!=original.trace.rho);}
if(OVERFLOW)assert(!std::isfinite(out.primary.energy)&&out.primary.energy!=917);
}
'''

@pytest.mark.parametrize('case',['heat','mechanical','stale_temperature','stale_species','stale_matching','stale_pressure','missing_workspace'])
def test_actual_wall_packet_tail(tmp_path,case):
    mutation={
        'heat':'prepared.conductiveHeatFlux=INFINITY;in.preparedLayer=&prepared;',
        'mechanical':'in.wallContext.mechanicalPressure=INFINITY;',
        'stale_temperature':'in.preparedLayer=&prepared;in.temperature=650;',
        'stale_species':'in.preparedLayer=&prepared;in.speciesRate[0]=.06;',
        'stale_matching':'in.preparedLayer=&prepared;in.wallContext.matching.temperature+=100;',
        'stale_pressure':'in.preparedLayer=&prepared;in.wallContext.matchingPressure*=1.2;',
        'missing_workspace':'p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;in.wallContext.workspace=nullptr;',
    }[case]
    code=SOURCE.replace('MUTATION',mutation).replace('MISSING',str(int(case=='missing_workspace'))).replace('STALE',str(int(case.startswith('stale')))).replace('OVERFLOW',str(int(case in ('heat','mechanical'))))
    source=tmp_path/'packet.cpp';source.write_text(code);exe=tmp_path/'packet'
    cmd=['g++','-std=c++17','-O1','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd));build=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr


def test_surface_packet_tail_and_disappearing_film_use_actual_values(tmp_path):
    code=r'''
#include "tests/TestSupport.H"
#include "ablation/InterfaceMath.H"
#include <cassert>
using namespace chmt;
int main(){auto p=chmt_test::physics();p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
SurfacePhysicsInput in;in.dt=.125;in.area=2;in.gasArea=3;in.solidArea=4;in.hasSolid=true;
in.gasNormal=in.solidNormal=in.normal={1,0,0};in.wallContext.mechanicalPressure=100003;
SurfacePhysicsResult r;r.gasState.velocity={2,3,4};r.traction={5,6,7};r.gasSpecies[0]=.03;r.gasSpecies[1]=.01;
r.condensed[0]=.04;r.gasAdvective=11;r.gasConductive=INFINITY;r.gasPressureWork=17;r.gasViscousWork=19;
r.poreSpecies[0]=.007;r.poreSpecies[1]=.003;r.poreSweep[0]=.001;r.poreEnergy=23;
SurfacePacketIdentity id;id.gasCell=id.solidCell=id.filmFace=0;ExchangePacket out[3];
for(auto& packet:out){packet.mass=901;packet.energy=902;packet.species[0]=903;}
DeviceStatus status;assert(assembleSurfacePackets(in,r,id,p,out,&status));
assert(out[0].species[0]==.25*.03&&out[0].condensed[0]==.25*.04&&std::isinf(out[0].energy));
assert(out[0].momentum.x==2*out[0].mass+(100003-5)*.375);
assert(out[1].mass==.25*.007+.25*.003&&out[1].energy==.25*23&&out[1].poreSweep[0]==.25*.001);
assert(out[2].mass==0&&out[2].energy==0&&out[2].species[0]==0&&status.code!=int(ErrorCode::Success));
// Actual disappearing-film transfer is evaluated even when the old numerical
// composition certificate disagrees with the supplied remaining species.
r={};r.wet=true;r.phaseMass=-100;r.phaseAdvective=2;r.phaseConductive=3;r.phasePressureWork=4;r.phaseViscousWork=5;
r.bottomKineticAdvection=7;r.radiationFlux=11;in.filmBase.mass=2;in.filmBase.species[0]=.4;in.filmBase.species[1]=1.6;
in.filmBase.enthalpy=901;in.filmLocalRate.enthalpy=13;in.aux.thickness=.01;
p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1;p.material.phaseFilmY[1]=0;id.oldFilmPV=103;status={};
assert(assembleSurfacePackets(in,r,id,p,out,&status));
const Real available=901-103+.125*13+.25*11;
assert(out[2].mass==-2&&out[2].species[0]==-.4&&out[2].species[1]==-1.6&&out[2].condensed[0]==-2);
assert(out[2].energy==-available-.25*7);
assert(out[2].advective==out[2].energy-out[2].conductive-out[2].pressureWork-out[2].viscousWork);
assert(status.code!=int(ErrorCode::Success));
}
'''
    source=tmp_path/'surface.cpp';source.write_text(code);exe=tmp_path/'surface'
    cmd=['g++','-std=c++17','-O1','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd));built=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(built.stdout+built.stderr);assert built.returncode==0,built.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr


@pytest.fixture(scope='module')
def prepared_velocity_runner(tmp_path_factory):
    directory=tmp_path_factory.mktemp('prepared-velocity')
    setup=SOURCE[:SOURCE.index('GasWallResult original;')]
    setup=setup.replace('#include <cassert>','#include <cassert>\n#include <cstdio>\n#include <cstdlib>')
    setup=setup.replace('int main(){',r'''
static void* tracked=nullptr;static unsigned calls=0;
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_enter(void* fn,void*){if(fn==tracked)++calls;}
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_exit(void*,void*){}
int main(int argc,char** argv){assert(argc==2);const int variant=std::atoi(argv[1]);
''')
    code=setup+r'''
auto mutate=[](GasWallInput& value,int kind){
 if(kind==0)value.velocity.y=NAN;
 if(kind==1)value.velocity.y=INFINITY;
 if(kind==2)value.normal.y=NAN;
 if(kind==3)value.normalSpeed=NAN;
};
// Variants 6..9 start with the exact terminal input being reused.
if(variant>=6&&variant<=9)mutate(in,variant-6);
if(variant==14||variant==15)in.normalSpeed=INFINITY;
GasWallResult original;assert(evaluateGasWall(in,id,p,original));
auto prepared=original.layer;const auto sampled=in.wallContext.matching;
const Real sampledPressure=in.wallContext.matchingPressure;
in.preparedLayer=&prepared;in.preparedMatching=&sampled;in.preparedMatchingPressure=&sampledPressure;
if(variant<4)mutate(in,variant);
if(variant==5)in.velocity.y=3;
if(variant==10){in.preparedMatching=nullptr;in.preparedMatchingPressure=nullptr;}
if(variant==11){in.velocity.y=NAN;p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;}
if(variant==12)in.velocity.y+=1e-13;
if(variant==13)in.normalSpeed=INFINITY;
if(variant==15)in.normalSpeed=-INFINITY;
tracked=reinterpret_cast<void*>(&evaluateInterfaceGasClosure<true>);calls=0;
GasWallResult reused;reused.primary.energy=917;reused.layer.traction[1]=918;
const bool accepted=evaluateGasWall(in,id,p,reused);const unsigned reuseCalls=calls;
if(variant==11){assert(!accepted&&calls==1&&reused.primary.energy==917&&reused.layer.traction[1]==918);return 0;}
assert(accepted);const bool changed=variant<4||variant==5||variant==13||variant==15;
std::printf("variant=%d fresh_calls=%u expected=%u\n",variant,reuseCalls,unsigned(changed));
assert(reuseCalls==unsigned(changed));
auto same=[](Real a,Real b){return a==b||(std::isnan(a)&&std::isnan(b));};
auto compareLayer=[&](const ugkwp::gaswall::WallOutput<Real,Ns>& a,const ugkwp::gaswall::WallOutput<Real,Ns>& b){
 for(auto pair:{std::pair<Real,Real>{a.traceTemperature,b.traceTemperature},{a.traceDensity,b.traceDensity},
  {a.conductiveHeatFlux,b.conductiveHeatFlux},{a.ownerOmega,b.ownerOmega},{a.wallKFlux,b.wallKFlux},
  {a.integratedKSource,b.integratedKSource},{a.residual,b.residual}})assert(same(pair.first,pair.second));
 const Real* left[]={a.traceMassFraction,a.traceVelocity,a.traction,a.wallSpeciesFlux,a.matchingSpeciesFlux,a.reactionIntegral,a.chemistryMismatch,a.speciesBalanceResidual};
 const Real* right[]={b.traceMassFraction,b.traceVelocity,b.traction,b.wallSpeciesFlux,b.matchingSpeciesFlux,b.reactionIntegral,b.chemistryMismatch,b.speciesBalanceResidual};
 const int sizes[]={Ns,3,3,Ns,Ns,Ns,Ns,Ns};
 for(int k=0;k<8;++k)for(int j=0;j<sizes[k];++j)assert(same(left[k][j],right[k][j]));
 assert(a.iterations==b.iterations&&a.chemistryMismatchAvailable==b.chemistryMismatchAvailable);
};
if(variant!=12){auto freshInput=in;freshInput.preparedLayer=nullptr;GasWallResult fresh;
 calls=0;assert(evaluateGasWall(freshInput,id,p,fresh)&&calls==1);compareLayer(reused.layer,fresh.layer);
 for(auto pair:{std::pair<Real,Real>{reused.trace.temperature,fresh.trace.temperature},{reused.trace.pressure,fresh.trace.pressure},
  {reused.trace.rho,fresh.trace.rho},{reused.trace.velocity.x,fresh.trace.velocity.x},{reused.trace.velocity.y,fresh.trace.velocity.y},
  {reused.trace.velocity.z,fresh.trace.velocity.z},{reused.trace.soundSpeed,fresh.trace.soundSpeed}})assert(same(pair.first,pair.second));
 for(int j=0;j<Ns;++j)assert(same(reused.trace.Y[j],fresh.trace.Y[j]));
 const ExchangePacket* lhs[]={&reused.primary,&reused.pore};const ExchangePacket* rhs[]={&fresh.primary,&fresh.pore};
 for(int k=0;k<2;++k){const auto&a=*lhs[k];const auto&b=*rhs[k];
  for(auto pair:{std::pair<Real,Real>{a.mass,b.mass},{a.energy,b.energy},{a.advective,b.advective},{a.conductive,b.conductive},
   {a.pressureWork,b.pressureWork},{a.viscousWork,b.viscousWork},{a.momentum.x,b.momentum.x},{a.momentum.y,b.momentum.y},
   {a.momentum.z,b.momentum.z},{a.liquidKineticAdvection,b.liquidKineticAdvection}})assert(same(pair.first,pair.second));
  for(int j=0;j<Ns;++j){assert(same(a.species[j],b.species[j]));assert(same(a.pore[j],b.pore[j]));assert(same(a.poreSweep[j],b.poreSweep[j]));}
  for(int j=0;j<Nc;++j)assert(same(a.condensed[j],b.condensed[j]));
 }
}else compareLayer(reused.layer,prepared);
if(!changed){
 // A matching prepared-only product does not require another BVP workspace,
 // including matching terminal components and the original finite tolerance.
 p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
 in.wallContext.workspace=nullptr;calls=0;GasWallResult withoutWorkspace;
 assert(evaluateGasWall(in,id,p,withoutWorkspace)&&calls==0);compareLayer(withoutWorkspace.layer,prepared);
}
}
'''
    source=directory/'reuse.cpp';source.write_text(code);exe=directory/'reuse'
    cmd=['g++','-std=c++17','-O0','-ffp-contract=off','-finstrument-functions','-fno-inline','-Wall','-Wextra','-Werror','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    (directory/'command.txt').write_text(' '.join(cmd));build=subprocess.run(cmd,capture_output=True,text=True)
    (directory/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
    return directory,exe


@pytest.mark.parametrize('variant',range(16),ids=['nan_velocity','inf_velocity','nan_normal','nan_speed','unchanged','finite_velocity','same_nan_velocity','same_inf_velocity','same_nan_normal','same_nan_speed','prepared_only','missing_new_workspace','finite_tolerance','inf_speed','same_inf_speed','opposite_inf_speed'])
def test_prepared_velocity_matches_actual_closure(prepared_velocity_runner,variant):
    directory,exe=prepared_velocity_runner
    run=subprocess.run([str(exe),str(variant)],capture_output=True,text=True)
    (directory/f'run-{variant}.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
