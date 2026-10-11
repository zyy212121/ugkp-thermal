"""Mechanical owner pressure must not replace the wall-model matching EOS pressure."""
from pathlib import Path
import subprocess

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]

SOURCE = r'''
#include "tests/TestSupport.H"
#include "gpu/GasWallMath.H"
#include "materials/MaterialDrive.H"
#include <limits>
#include <type_traits>
using namespace chmt;
using chmt_test::check; using chmt_test::near;
// Keep the regression executable on the unfixed tree so RED is behavioral.
template<class T> auto setMechanical(T& value, Real pressure, int) -> decltype(value.mechanicalPressure=pressure,void()) {value.mechanicalPressure=pressure;}
template<class T> void setMechanical(T&,Real,long) {}
template<class T> void setMechanical(T& value,Real pressure) {setMechanical(value,pressure,0);}
int main(){
 auto p=chmt_test::physics();p.gasMode=ugkwp::GasMode::MixtureFrozen;
 p.gasViscosity=.02;p.gasConductivity=.8;
 for(int s=0;s<Ns;++s){p.species[s].cp1=0;p.gasDiffusivity[s]=.01;}
 p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
 p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
 const auto bulk=chmt_test::gas(p);const Real mechanical=1.7*bulk.pressure;
 GasWallClosureContext context;context.ownerDistance=.2;context.matchingDistance=.8;
 context.matchingPressure=bulk.pressure;setMechanical(context,mechanical);
 context.matching.temperature=bulk.temperature;context.matching.velocity[1]=10;
 for(int s=0;s<Ns;++s)context.matching.massFraction[s]=bulk.Y[s];
 SurfacePacketIdentity id;id.gasCell=0;id.solidCell=0;id.filmFace=0;
 // Static, translating, and translating/blowing use the same mechanical p.
 for(int mode=0;mode<3;++mode){
  GasWallInput in;in.bulk=bulk;in.wallContext=context;in.temperature=600;
  in.gasDistance=.2;in.area=in.gasArea=2;in.dt=.01;in.normal={1,0,0};in.velocity={0,2,0};
  in.condensedRate[0]=mode==2?.03:0;in.normalSpeed=mode?-.5:0;in.sweptVolume=in.gasArea*in.dt*in.normalSpeed;
  for(int s=0;s<Ns;++s){in.speciesRate[s]=mode==2?.03*bulk.Y[s]:0;in.poreRate[s]=mode==2?.01*bulk.Y[s]:0;}
  GasWallResult result;check(evaluateGasWall(in,id,p,result),"mechanical wall evaluation");
  near(result.primary.momentum.x-result.primary.mass*result.trace.velocity.x+result.traction.x*in.gasArea*in.dt,
       mechanical*in.gasArea*in.dt,"mechanical pressure supplies normal momentum");
  near(result.primary.pressureWork,mechanical*in.sweptVolume,"same mechanical pressure supplies swept work");
  near(result.trace.pressure,bulk.pressure,"trace EOS retains matching pressure");
  Real R=0,h=0;for(int s=0;s<Ns;++s){R+=result.trace.Y[s]*p.species[s].R;h+=bulk.Y[s]*speciesH(p.species[s],600);}
  near(result.trace.rho,bulk.pressure/(R*600),"trace density retains matching pressure");
  near(result.trace.velocity.x,in.normalSpeed+(mode==2?.04:0)/result.trace.rho,"blowing uses matching density");
  near(result.primary.advective,result.primary.mass*(h+.5*dot(result.trace.velocity,result.trace.velocity)),"matching enthalpy and trace kinetic advection unchanged");
  auto baseline=in;setMechanical(baseline.wallContext,bulk.pressure);GasWallResult equal;
  check(evaluateGasWall(baseline,id,p,equal),"equal-pressure control");
  near(result.primary.energy-equal.primary.energy,(mechanical-bulk.pressure)*in.sweptVolume,"only mechanical work changes energy");
  near(result.primary.conductive,equal.primary.conductive,"matching BVP heat unchanged");
  near(result.primary.viscousWork,equal.primary.viscousWork,"matching BVP viscous work unchanged");
  SurfacePhysicsInput surface;surface.bulk=bulk;surface.wallContext=context;surface.area=surface.gasArea=surface.solidArea=2;
  surface.normal=surface.gasNormal=surface.solidNormal={1,0,0};surface.gasDistance=.2;surface.solidDistance=.1;
  surface.hasSolid=true;surface.solidVolume=1;surface.dt=.01;surface.baseVelocity={0,2,0};
  surface.useAcceptedFlux=true;surface.useSweptGeometry=true;surface.gasSweptRate=2*in.normalSpeed;
  surface.solidSweptRate=2*in.normalSpeed;surface.acceptedCondensed[0]=in.condensedRate[0];
  for(int s=0;s<Ns;++s){surface.acceptedGasSpecies[s]=in.speciesRate[s];surface.acceptedPoreSpecies[s]=in.poreRate[s];}
  MaterialPrimitive material;material.temperature=500;material.conductivity=2;
  SurfacePhysicsResult candidate;check(evaluateSurfaceTemperature(surface,p,material,500,600,false,candidate),"CPU surface mechanical-pressure evaluation");
  near(candidate.gasPressureWork*in.area*in.dt,result.primary.pressureWork,"CPU and GPU packet pressure work agree");
  near(candidate.gasEnergy*in.area*in.dt,result.primary.energy,"CPU and GPU packet energy agree");
  ExchangePacket packets[3];check(assembleSurfacePackets(surface,candidate,id,p,packets),"CPU packet assembly");
  near(packets[0].momentum.x,result.primary.momentum.x,"CPU and GPU packet momentum agree");
  // Top-liquid p*w must cancel gas p*w without changing film thermodynamic p.
  surface.hasFilm=true;surface.film.mass=surface.filmBase.mass=1;surface.film.species[0]=surface.filmBase.species[0]=1;
  surface.aux.topVelocity={0,2,0};surface.acceptedCondensed[0]=0;surface.acceptedPoreSpecies[0]=surface.acceptedPoreSpecies[1]=0;
  SurfacePhysicsResult wet,wetEqual;
  check(evaluateSurfaceTemperature(surface,p,material,500,600,false,wet),"wet surface pressure split");
  setMechanical(surface.wallContext,bulk.pressure);
  check(evaluateSurfaceTemperature(surface,p,material,500,600,false,wetEqual),"wet equal-pressure control");
  near(wet.thermalResidual,wetEqual.thermalResidual,"wet top gas and liquid pressure work cancels");
  near(wet.phasePressureWork,wetEqual.phasePressureWork,"bottom film phase work retains thermodynamic pressure");
  near(wet.phaseAdvective,wetEqual.phaseAdvective,"bottom film enthalpy retains matching pressure");
  // Existing low-Re path ignores the boundary-layer-only field.
  auto legacy=p;legacy.wallModel.family=ugkwp::gaswall::WallFamily::LowRe;
  GasWallResult old,oldEqual;check(evaluateGasWall(in,id,legacy,old)&&evaluateGasWall(baseline,id,legacy,oldEqual),"legacy pressure control");
  near(old.primary.energy,oldEqual.primary.energy,"legacy energy unchanged");near(old.primary.momentum.x,oldEqual.primary.momentum.x,"legacy momentum unchanged");
 }
 static_assert(std::is_trivially_copyable<GasWallMatchingSample>::value,"matching history is POD");
 IntervalHistory history;CouplingInterval interval;interval.end=1;std::string error;check(history.begin(interval,error),"history begin");
 GasIntervalRecord a;a.microSequence=1;a.end=.5;a.gasTrace={bulk};a.gasWallMatching.resize(1);
 auto& sample=a.gasWallMatching[0];sample.pressure=bulk.pressure;sample.state=context.matching;setMechanical(sample,mechanical);
 check(history.appendAccepted(a,error),"valid mechanical history accepted");
 auto b=a;b.microSequence=2;b.begin=.5;b.end=1;setMechanical(b.gasWallMatching[0],2*mechanical);
 check(history.appendAccepted(b,error),"changed mechanical history accepted");Real end=0;std::uint64_t count=0;
 check(materialDriveSlabEnd(history,0,1,.05,1,end,count,error),"mechanical drive slab");near(end,.5,"mechanical pressure changes limit material lag");
 for(Real invalid:{Real(0),Real(-1),std::numeric_limits<Real>::quiet_NaN()}){
  IntervalHistory bad;check(bad.begin(interval,error),"invalid history begin");setMechanical(a.gasWallMatching[0],invalid);
  check(!bad.appendAccepted(a,error),"missing or invalid mechanical pressure is rejected");
 }
}
'''

def test_packets_separate_mechanical_and_thermodynamic_pressure(tmp_path):
    source = tmp_path / 'mechanical.cpp'
    source.write_text(SOURCE)
    binary = tmp_path / 'mechanical'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-I'+str(APP), '-I'+str(ROOT/'common'), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_backend_preserves_common_reconstructed_pressure_flux(tmp_path):
    import runpy
    helpers = runpy.run_path(str(APP/'tests/test_shared_backend_compile.py'))
    source = helpers['prepare_backend'](tmp_path)
    fixture = runpy.run_path(str(APP/'tests/test_reacting_wall_storage.py'))['storage_source']()
    fixture = fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
    fixture = fixture.replace('int main(){', 'void run(int mode){')
    body = r'''
 h.solid.resize(1);h.solid[0].condensed[0]=1;h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.surface.normal={{0,0,1}};
 p.minDt=1e-12;p.maxDt=.001;p.cfl=.4;
 std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> b(chmt::createBackend(model,canonical,{},chmt::GasExecutionOptions{},h,error),chmt::destroyBackend);assert(b);
 chmt::WallProgram program;program.interval.end=.01;program.surface=h.surface;
 chmt::WallKnot a,z;a.time=0;z.time=.01;a.gasPoints=z.gasPoints=m.points;
 chmt::WallFaceSample sample;sample.temperature=600;sample.normalVelocity=mode%2?.25:0;
 sample.speciesRate[0]=sample.condensedRate[0]=mode>=2?.02:0;sample.poreRate[0]=mode>=2?.01:0;
 a.faces=z.faces={sample};program.knots={a,z};assert(chmt::beginGasWindow(*b,program,error));
 b->microTime=0;b->microDt=.0001;assert(chmt::transport::CoupledTrialPolicy::begin(b.get())==0);
 auto&v=b->storage.hostView();
 if(mode%2){
  // A rigid translating stage has zero volume change and exact closed-cell GCL.
  for(std::size_t f=0;f<m.areaVectors.size();++f)b->gasStage.sweptVolume[f]=m.areaVectors[f].z*sample.normalVelocity*b->microDt;
  assert(b->storage.bindStageGeometry(b->gasStage));b->gasGeometry=v.gasGeometry;
 }
 for(int c=0;c<3;++c){v.rho[c]=1;v.p[c]=1e5;v.Tgas[c]=700;v.Ux[c]=5;v.gasSpecies.soundSpeed[c]=400;}
 v.p[0]=140000;v.Tgas[1]=900;v.gasReconstruction=1;v.gasGradientLimiterP[0]=1;v.gradPz[0]=20000;
 assert(b->storage.refreshView());blockIdx.x=0;blockDim.x=1;threadIdx.x=0;
 assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->preparedMatching[0].pressure==100000);assert(b->preparedMatching[0].state.temperature==900);
 assert(b->preparedMatching[0].mechanicalPressure==0); // unavailable until the common flux has been evaluated.
 b->preparedMatching[0].mechanicalPressure=123;assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->preparedMatching[0].mechanicalPressure==0); // a new prepared stage cannot retain old mechanical data.
 // Inject a known nonzero normal viscous stress into the physical wall output.
 // This isolates the recovery formula's stress subtraction as well as m*u.
 auto& layer=b->preparedWallLayers[0];layer.traction[2]=7.5;
 const auto& input=v.gasBoundaryLayerModel.input[0];
 assert(ugkwp::publishGasBoundaryLayerOutput(v,0,input,layer,1.,sample.normalVelocity));
 blockIdx.x=wall;chmt::transport::computeGasInternalFaceFluxKernel<false>(b->deviceState,.0001);
 assert(v.gasSpecies.faceStatus[wall]==0);
 const chmt::Vec3 commonMomentum={v.gasPhiRhoUx[wall],v.gasPhiRhoUy[wall],v.gasPhiRhoUz[wall]};const auto commonEnergy=v.gasPhiRhoE[wall];
 assert(chmt::applyCoupledFaces(b.get(),.0001,0)==0);
 const auto& result=b->faceRates[0];
 assert(std::abs(b->preparedMatching[0].mechanicalPressure-130000)<1e-9);
 assert(result.trace.pressure==100000);
 assert(std::abs(result.primary.pressureWork-130000*sample.normalVelocity*.0001)<1e-9);
 assert(chmt::mag(chmt::Vec3{v.gasPhiRhoUx[wall],v.gasPhiRhoUy[wall],v.gasPhiRhoUz[wall]}-commonMomentum)<1e-9);
 assert(std::abs(v.gasPhiRhoE[wall]-commonEnergy)<1e-8);
 assert(chmt::accountCoupledFaces(b.get(),.0001)==0);
 assert(b->record.packets[0].pressureWork==result.primary.pressureWork);
 assert(chmt::rollbackGasWindow(*b,error));assert(!b->preparedWallReady);
 assert(chmt::beginGasWindow(*b,program,error));b->microTime=0;b->microDt=.0001;
 assert(chmt::transport::CoupledTrialPolicy::begin(b.get())==0);
 assert(chmt::applyCoupledFaces(b.get(),.0001,0)!=0); // old snapshot cannot cross rollback.
}
int main(){for(int mode=0;mode<4;++mode)run(mode);}
'''
    source.write_text(source.read_text()+fixture+body)
    binary = tmp_path/'backend-mechanical'
    subprocess.run(helpers['compiler_flags'](tmp_path)+['-I'+str(ROOT),str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
