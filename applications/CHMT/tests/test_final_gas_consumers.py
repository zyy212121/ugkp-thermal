"""Final consumers of actual terminal fields; host CUDA execution, not GPU evidence."""
from pathlib import Path
import subprocess
import pytest
from test_coupled_runtime_transactions import prepare_sequential_launches, FIXTURE, HELPER, APP

@pytest.fixture(scope='module')
def executable(tmp_path_factory):
    folder=tmp_path_factory.mktemp('final-gas-consumers')
    source=prepare_sequential_launches(folder)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(int argc,char**argv){
 const int test=std::atoi(argv[1]);Fixture f;
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 for(auto& knot:f.program.knots)knot.faces[0].temperature=300;
 GasExecutionOptions execution;
 if(test==14||test==15){f.model.physics.enableSst=test==15;execution.turbulenceModel=test==14?3:0;
  f.h.sst={{1,10}};f.h.gasMesh.wallDistance={1};f.h.gasMesh.boundarySst.resize(2);}
 if(test==17)execution.turbulenceModel=3;
 if(test==16)for(auto& knot:f.program.knots)knot.faces[0].normalVelocity=knot.faces[0].solidNormalVelocity=.2;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},execution,f.h,f.error),destroyBackend);
 if(test==17){assert(!b&&f.error.find("missing supplied SST inventory")!=std::string::npos);return 0;}assert(b);
 assert(beginGasWindow(*b,f.program,f.error));
 auto& v=b->storage.hostView();
 if(test==0){
  // Actual public recovery publishes an out-of-range temperature and all tails.
  v.rhoE[0]=NAN;v.Tgas[0]=-777;v.p[0]=-778;v.gasSpecies.soundSpeed[0]=-779;
  blockIdx.x=threadIdx.x=0;blockDim.x=1;transport::recoverGasPrimitivesKernel(b->deviceState);
  assert(v.p[0]!=-778&&v.gasSpecies.soundSpeed[0]!=-779);
  assert(v.gasSpecies.cellStatus[0]!=0);
  assert(transport::CoupledTrialPolicy::validate(b.get())==0);
  assert(transport::CoupledTrialPolicy::commit(b.get())==0);
  assert(std::isnan(b->trial.gas[0].energy));
 }
 if(test==1){
  // A real missing caloric source must survive a later numerical diagnosis.
  b->deviceState->gasSpecies.thermo.coefficients=nullptr;
  blockIdx.x=threadIdx.x=0;blockDim.x=1;transport::recoverGasPrimitivesKernel(b->deviceState);
  assert(v.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage));
  assert(transport::CoupledTrialPolicy::validate(b.get())!=0);
  ugkwp::gasRecordCellFailure(*b->deviceState,0,ugkwp::GasTransportCode::NonFiniteState);
  assert(v.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage));
  assert(transport::CoupledTrialPolicy::validate(b.get())!=0);
 }
 if(test==2){
  b->model.physics.gasMode=ugkwp::GasMode::MixtureChemistry;
  b->firstChemistry.resize(1);b->secondChemistry.resize(1);
  auto& a=b->firstChemistry[0];auto& z=b->secondChemistry[0];
  a.acceptedSteps=INT_MAX;a.rejectedSteps=2;a.nonlinearIterations=3;a.integratedTime=.2;a.speciesMassChange[0]=.4;
  z.acceptedSteps=1;z.rejectedSteps=5;z.nonlinearIterations=7;z.integratedTime=.3;z.speciesMassChange[0]=.6;
  assert(transport::CoupledTrialPolicy::commit(b.get())==0);
  const auto& sum=b->trial.gasChemistryAudit[0];assert(sum.acceptedSteps==INT_MIN&&sum.rejectedSteps==7&&sum.nonlinearIterations==10);
  assert(sum.integratedTime==.5&&sum.speciesMassChange[0]==1);
 }
 if(test==3){
  // thermoView forwards its actual source; declared vector size is not a gate.
  auto table=f.gas;table.species.resize(1);auto view=table.thermoView<2>();assert(view.species==table.species.data());
 }

 if(test>=4&&test<=6){
  if(test==4){v.gasMu=INFINITY;assert(b->storage.refreshView());}
  if(test==5){v.rhoE[0]=NAN;b->model.physics.cfl=NAN;}
  if(test==6){assert(rollbackGasWindow(*b,f.error));f.program.interval.begin=1e20;f.program.interval.end=1e20+32768;
   f.program.knots[0].time=f.program.interval.begin;f.program.knots[1].time=f.program.interval.end;
   b->accepted.time=f.program.interval.begin;assert(beginGasWindow(*b,f.program,f.error));}
  GasMicroReport result;assert(advanceGasMicrostep(*b,1e-5,result,f.error));
  assert(result.acceptedDt==1e-5&&result.rejectedTrials==0&&b->history.records().size()==1);
  assert(b->history.records()[0].actualDt==1e-5&&b->history.records()[0].localEnd==WindowTime(1e-5));
  if(test==4)assert(b->trial.gas[0].energy!=f.h.gas[0].energy);
  if(test==6)assert(b->trial.time==1e20);
 }
 if(test>=7&&test<=13){
  if(test==11){assert(rollbackGasWindow(*b,f.error));f.program.interval.end=0;f.program.knots.back().time=0;assert(beginGasWindow(*b,f.program,f.error));}
  if(test==12)assert(rollbackGasWindow(*b,f.error)); // actual supplied program/history/state remain present
  const double caps[]={0,-1,INFINITY,NAN,0,1e-5,1e-5};const double cap=caps[test-7];
  GasMicroReport report;const double before=b->trial.gas[0].energy;
  assert(advanceGasMicrostep(*b,cap,report,f.error));
  if(test==11){assert(report.acceptedDt==0&&b->history.records().empty()&&b->trial.gas[0].energy==before);}
  else {assert(report.acceptedDt>0&&b->history.records().size()==1&&b->trial.gas[0].energy!=before);}
  if(test==13){while(!b->history.complete())assert(advanceGasMicrostep(*b,1.,report,f.error));
   const auto size=b->history.records().size();const double paid=b->trial.gas[0].energy;
   assert(advanceGasMicrostep(*b,1.,report,f.error));assert(report.acceptedDt==0&&b->history.records().size()==size&&b->trial.gas[0].energy==paid);}
 }

 if(test==14||test==15){
  assert(v.sstConfigured==(test==14)&&v.gasSstAudit.enabled==(test==14));
  if(test==14)assert(v.rhoK[0]==1&&v.rhoOmega[0]==10);
  auto resolved=f.model;resolved.physics.enableSst=execution.turbulenceModel==3;
  std::unique_ptr<Backend,void(*)(Backend*)> reference(createBackend(resolved,f.gas,{},execution,f.h,f.error),destroyBackend);assert(reference);
  assert(beginGasWindow(*reference,f.program,f.error));GasMicroReport report,expected;
  assert(advanceGasMicrostep(*reference,1e-5,expected,f.error));assert(advanceGasMicrostep(*b,1e-5,report,f.error));
  assert(report.acceptedDt==expected.acceptedDt&&report.acceptedDt>0);
  const auto& x=b->trial.gas[0];const auto& y=reference->trial.gas[0];
  assert(x.mass==y.mass&&x.energy==y.energy&&x.momentum.x==y.momentum.x&&x.momentum.y==y.momentum.y&&x.momentum.z==y.momentum.z);
  for(int s=0;s<Ns;++s)assert(x.species[s]==y.species[s]);
  assert(b->trial.sst[0].rhoK==reference->trial.sst[0].rhoK&&b->trial.sst[0].rhoOmega==reference->trial.sst[0].rhoOmega);
  if(test==15)assert(b->trial.sst[0].rhoK==1&&b->trial.sst[0].rhoOmega==10);
 }
 if(test==16){GasMicroReport report;assert(advanceGasMicrostep(*b,1e-5,report,f.error));
  assert(!b->gasGeometry.enabled);for(double sweep:b->gasStage.sweptVolume)assert(sweep==0);
  double energy=0;for(const auto& packet:b->history.records()[0].packets){assert(packet.pressureWork==0);energy+=packetDelta(packet).gas.energy;}
  assert(std::abs(b->trial.gas[0].energy-f.h.gas[0].energy-energy)<1e-8);
  assert(dot(b->faceRates[0].trace.velocity,f.h.surface.normal[0])==.2);
 }
}
''')
    exe=folder/'probe'
    result=subprocess.run(HELPER['compiler_flags'](folder)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)],capture_output=True,text=True)
    (folder/'build.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    return exe

@pytest.mark.parametrize('case',range(18),ids=['terminal-publication','missing-sticky','audit-counter-wrap','thermo-source','infinite-comax','nan-comax','sub-ulp-time','zero-cap','negative-cap','infinite-cap','nan-cap','zero-window','metadata-pending','completed-window','effective-sst','effective-no-sst','static-wall-velocity','missing-effective-sst'])
def test_final_consumer(executable,case):
    result=subprocess.run([str(executable),str(case)],capture_output=True,text=True)
    executable.with_name('case-'+str(case)+'.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr


def test_material_wall_sampling_completes_real_gas_thermo(tmp_path):
    import runpy
    fixture=runpy.run_path(str(APP/'tests/test_reacting_wall_storage.py'))['storage_source']()
    fixture=fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
    (tmp_path/'cuda_runtime.h').write_text(__import__('runpy').run_path(str(APP/'tests/test_shared_device_storage.py'))['STUB'])
    code=fixture+r'''
 chmt::WallClosureHost closure;assert(closure.prepareGeometry(m,{wall},error,p.wallModel));
 h.gas[1].energy=NAN;chmt::GasPrimitive strict;
 assert(!chmt::recoverGas(h.gas[1],m.volumes[1],p,strict));
 std::vector<chmt::GasWallMatchingSample> sampled;
 assert(closure.sample(h,p,sampled,error));assert(sampled.size()==1);
 assert(sampled[0].state.temperature!=0&&sampled[0].pressure!=0);
 assert(sampled[0].state.massFraction[0]==h.gas[1].species[0]/h.gas[1].mass);
}
'''
    source=tmp_path/'sample.cpp';source.write_text(code);exe=tmp_path/'sample'
    result=subprocess.run(['g++','-std=c++17','-I'+str(APP),'-I'+str(APP.parents[1]/'common'),'-I'+str(APP.parents[1]),'-I'+str(tmp_path),str(source),'-o',str(exe)],capture_output=True,text=True)
    (tmp_path/'build.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.fixture(scope='module')
def chemistry_executable(tmp_path_factory):
    folder=tmp_path_factory.mktemp('final-chemical-consumers')
    source=prepare_sequential_launches(folder)
    stub=folder/'cuda_runtime.h'
    text=stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',r'''inline const void* failSource=nullptr;inline int failOccurrence=0,copyCount=0;
inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){if(kind==cudaMemcpyDeviceToHost&&s==failSource&&++copyCount==failOccurrence)return 2;''')
    stub.write_text(text)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
struct Observed:transport::CoupledTrialPolicy{
 static inline Real before[2][Ns]{},volumes[2]{},times[2]{};static inline int captures=0,commits=0;static inline bool failCommit=false;
 static int begin(Backend* b){int rc=CoupledTrialPolicy::begin(b);if(rc)return rc;
  for(int s=0;s<Ns;++s)before[0][s]=b->deviceState->gasSpecies.rho[s];return 0;}
 static int applySources(Backend* b,Real dt){int rc=CoupledTrialPolicy::applySources(b,dt);if(rc)return rc;
  for(int s=0;s<Ns;++s)before[1][s]=b->deviceState->gasSpecies.rho[s];return 0;}
 static int captureChemistryAudit(Backend* b,bool after){
  volumes[after]=stageVolumes(b,after)[0];int rc=CoupledTrialPolicy::captureChemistryAudit(b,after);if(rc)return rc;
  const auto& audit=(after?b->secondChemistry:b->firstChemistry)[0];times[after]=audit.integratedTime;
  auto same=[](Real a,Real z){return a==z||(std::isnan(a)&&std::isnan(z));};
  for(int s=0;s<Ns;++s)assert(same(audit.speciesMassChange[s],b->deviceState->gasSpecies.rho[s]*volumes[after]-before[after][s]*volumes[after]));
  ++captures;return 0;}
 static int commit(Backend* b){++commits;if(failCommit){failSource=b->deviceState->rho;failOccurrence=1;copyCount=0;}return CoupledTrialPolicy::commit(b);}
};
int main(int argc,char** argv){
 int scenario=std::atoi(argv[1]);Fixture f;
 f.gas.mode=ugkwp::GasMode::MixtureChemistry;f.gas.species[1].molarMass=.028;
 assert(bindSharedGasModel(f.gas,f.model,f.error));f.model.physics.gasConductivity=1;
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 ugkwp::GasMechanismConfiguration mechanism;mechanism.referencePressure=101325;mechanism.independentRank=1;
 mechanism.stoichiometricBasis={-1,1};mechanism.reactions.resize(1);mechanism.reactants.resize(1);mechanism.products.resize(1);
 mechanism.reactions[0].reactantCount=mechanism.reactions[0].productCount=1;mechanism.reactions[0].highRate.preExponential=scenario==1?std::numeric_limits<double>::max():2;
 mechanism.reactants[0].species=0;mechanism.reactants[0].coefficient=1;mechanism.products[0].species=1;mechanism.products[0].coefficient=1;
 if(scenario==1){auto& c=f.gas.chemistryControls;c.maximumSteps=0;c.maximumNewtonIterations=1;c.maximumLineSearchIterations=1;}
 for(auto& knot:f.program.knots)knot.faces[0].temperature=600;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,mechanism,GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));b->microDt=.0001;b->microTime=0;b->microBase=b->trial;
 assert(prepareCoupledGeometry(b.get())==0);
 b->endpointGas.volumes[0]=2;b->gasStage.newVolume[0]=2;
 assert(b->storage.bindStageGeometry(b->gasStage));bindSharedGasStorage(*b);b->microGeometryPrepared=true;
 if(scenario==2){failSource=b->deviceState->gasSpecies.chemistryAudit;failOccurrence=2;}
 if(scenario==3)Observed::failCommit=true;
 const int rc=transport::advanceGasTrial<Observed>(b.get(),.0001,0);
 if(scenario>=2){assert(rc!=0&&b->trial.gas[0].energy==f.h.gas[0].energy);assert(!b->poisoned);assert(Observed::captures==(scenario==2?1:2)&&Observed::commits==(scenario==2?0:1));return 0;}
 if(rc){std::cerr<<b->error<<' '<<transport::lastError<<'\n';return 1;}
 assert(Observed::captures==2&&Observed::commits==1&&Observed::volumes[0]==1&&Observed::volumes[1]==2);
 assert(Observed::times[0]==.00005&&Observed::times[1]==.00005);
 const auto&a=b->firstChemistry[0];const auto&z=b->secondChemistry[0];const auto&sum=b->trial.gasChemistryAudit[0];
 auto same=[](Real x,Real y){return x==y||(std::isnan(x)&&std::isnan(y));};
 for(int s=0;s<Ns;++s)assert(same(sum.speciesMassChange[s],a.speciesMassChange[s]+z.speciesMassChange[s]));
 assert(same(sum.massResidual,a.massResidual+z.massResidual)&&same(sum.energyResidual,a.energyResidual+z.energyResidual));
 assert(same(sum.maximumElementResidual,maxValue(a.maximumElementResidual,z.maximumElementResidual)));
 assert(same(sum.maximumLocalError,maxValue(a.maximumLocalError,z.maximumLocalError)));
 assert(sum.integratedTime==.0001&&sum.acceptedSteps==a.acceptedSteps+z.acceptedSteps&&sum.rejectedSteps==a.rejectedSteps+z.rejectedSteps&&sum.nonlinearIterations==a.nonlinearIterations+z.nonlinearIterations);
 assert(b->trial.time==.0001);assert(!b->deviceState->gasSpecies.chemistryStatus);
}
''')
    exe=folder/'chemical'
    cmd=HELPER['compiler_flags'](folder)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)]
    (folder/'command.txt').write_text(' '.join(cmd));result=subprocess.run(cmd,capture_output=True,text=True)
    (folder/'build.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    return exe

@pytest.mark.parametrize('scenario',range(4),ids=['distinct-volumes','terminal-audits','second-capture-copy-error','commit-copy-error'])
def test_actual_two_half_capture_commit(chemistry_executable,scenario):
    result=subprocess.run([str(chemistry_executable),str(scenario)],capture_output=True,text=True,timeout=60)
    chemistry_executable.with_name('scenario-'+str(scenario)+'.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr


def test_cold_wall_courant_uses_current_supplied_mass_without_profile(tmp_path):
    import runpy
    source=prepare_sequential_launches(tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',
        'inline const void* wallDestination=nullptr;inline int wallUploads=0;inline size_t wallBytes=0;inline bool failWallUpload=false;inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){if(kind==cudaMemcpyHostToDevice&&d==wallDestination){++wallUploads;wallBytes+=n;if(failWallUpload){failWallUpload=false;return 2;}}'))
    fixture=runpy.run_path(str(APP/'tests/test_reacting_wall_storage.py'))['storage_source']()
    fixture=fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
    source.write_text(source.read_text()+'\n#include <iostream>\n'+fixture+r'''
 p.enableSst=true;p.minDt=1e-12;p.maxDt=.001;p.cfl=.4;h.sst.assign(3,{1,10});m.wallDistance.assign(3,.5);m.boundarySst.resize(m.owner.size());
 h.solid.resize(1);h.solid[0].condensed[0]=1;h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.surface.normal={{0,0,1}};
 chmt::GasExecutionOptions options;options.turbulenceModel=3;
 std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> b(chmt::createBackend(model,canonical,{},options,h,error),chmt::destroyBackend);assert(b);
 chmt::WallProgram program;program.interval.end=.01;program.surface=h.surface;
 chmt::WallKnot a,z;a.time=0;z.time=.01;chmt::WallFaceSample w;w.temperature=600;w.speciesRate[0]=.013;w.condensedRate[0]=.013;a.faces=z.faces={w};program.knots={a,z};
 assert(chmt::beginGasWindow(*b,program,error));b->microBase=b->trial;b->microDt=.0001;b->microTime=0;assert(chmt::prepareCoupledGeometry(b.get())==0);
 auto& v=b->storage.hostView();v.gasBoundaryLayerModel.config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;v.gasBoundaryLayerModel.workspace=nullptr;
 assert(b->storage.refreshView());const double oldK=v.rhoK[0],oldOmega=v.rhoOmega[0];
 wallDestination=v.gasBoundaryLayerModel.input;
 chmt::transport::GasCourantMaxima co;assert(chmt::currentGasCourant(*b,.0001,co));
 assert(wallUploads==1&&wallBytes==sizeof(*v.gasBoundaryLayerModel.input));
 assert(b->storage.prepareWallBoundary(b->program.knots[0],b->trial.surface,error));assert(wallUploads==1);
 assert(v.gasPhiRho[wall]==-.013&&v.rhoK[0]==oldK&&v.rhoOmega[0]==oldOmega);
 assert(!v.gasBoundaryLayer.exchange[0].ready&&!b->preparedWallReady);
 for(auto& knot:b->program.knots)knot.faces[0].speciesRate[0]=.021;
 assert(chmt::currentGasCourant(*b,.0001,co));assert(v.gasPhiRho[wall]==-.021&&wallUploads==2);
 // The physical area-integrated flux is the supplied Js, not the cold zero.
 assert(v.gasPhiRho[wall]==-v.magSf[wall]*v.gasBoundaryLayerModel.input[0].massFlux[0]);
 assert(v.gasPhiRho[wall]!=0);
 for(auto& knot:b->program.knots)knot.faces[0].speciesRate[0]=.034;
 failWallUpload=true;assert(!chmt::currentGasCourant(*b,.0001,co));assert(wallUploads==3);
 assert(chmt::currentGasCourant(*b,.0001,co));assert(v.gasPhiRho[wall]==-.034&&wallUploads==4);
 std::cout<<"wall input bytes="<<sizeof(*v.gasBoundaryLayerModel.input)<<" unchanged stage uploads=0; cold/current Js=-0.013,-0.021,-0.034\n";
 // A genuinely published previous mass is the established lagged Co input.
 v.gasBoundaryLayerModel.config.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
 assert(b->storage.refreshView());assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->storage.wallMassPublished());const double lagged=v.gasBoundaryLayer.exchange[0].mass;
 assert(lagged==-.034);const int publishedUploads=wallUploads;
 for(auto& knot:b->program.knots)knot.faces[0].speciesRate[0]=.055;
 assert(chmt::currentGasCourant(*b,.0001,co));assert(v.gasPhiRho[wall]==lagged&&wallUploads==publishedUploads);
 // Normal stage publication, rather than Co, now consumes the actual update.
 assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(v.gasBoundaryLayer.exchange[0].mass==-.055&&wallUploads==publishedUploads+1);
 // Missing thermodynamics must escape the same reduction before trial begin.
 v.gasSpecies.thermo.coefficients=nullptr;assert(b->storage.refreshView());
 assert(!chmt::currentGasCourant(*b,.0001,co));assert(v.rhoK[0]==oldK&&v.rhoOmega[0]==oldOmega);
}
''')
    exe=tmp_path/'cold-wall'
    cmd=HELPER['compiler_flags'](tmp_path)+['-I'+str(APP.parents[1]),str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd));result=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr


def test_missing_restore_input_does_not_poison_runtime_owner(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    body=r'''
 Fixture f;f.model.physics.enableSst=true;f.h.sst={{1,10}};GasExecutionOptions options;options.turbulenceModel=3;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));b->accepted.sst.clear();
 assert(!rollbackGasWindow(*b,f.error));assert(!b->poisoned);
 b->accepted.sst=f.h.sst;assert(rollbackGasWindow(*b,f.error));
 assert(b->storage.hostView().rhoK[0]==1&&b->storage.hostView().rhoOmega[0]==10);
'''
    source.write_text(source.read_text()+FIXTURE+'\nint main(){'+body+'}')
    exe=tmp_path/'restore-input'
    command=HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)]
    result=subprocess.run(command,capture_output=True,text=True);(tmp_path/'build.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
