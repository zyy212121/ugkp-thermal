"""Actual backend hooks with host CUDA allocation shim, never GPU evidence."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import runpy
import subprocess

APP=(SOURCE_ROOT / 'applications/CHMT')
HELPER=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_shared_backend_compile.py')))
FIXTURE=r'''
#include "configuration/SharedGasModel.H"
#include <cassert>
using namespace chmt;
struct Fixture {
 ugkwp::GasModelConfiguration gas;ModelConfig model;HostState h;WallProgram program;std::string error;
 Fixture(){
 gas=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.032;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}diffusion{model none;})");
 assert(bindSharedGasModel(gas,model,error));model.physics.gasConductivity=1;
 model.physics.minDt=1e-12;model.physics.maxDt=.001;model.physics.cfl=.4;
 h.gas.resize(1);h.gas[0].mass=1;h.gas[0].species[0]=1;h.gas[0].energy=1e6;h.solid.resize(1);h.solid[0].condensed[0]=1;
 auto&m=h.gasMesh;m.volumes={1};m.cellCentres={{0,0,0}};m.owner={0,0};m.neighbour={-1,-1};m.periodicPartner={-1,-1};m.cellFaceOffsets={0,2};m.cellFaces={0,1};
 m.faceCentres={{-1,0,0},{1,0,0}};m.areaVectors={{-1,0,0},{1,0,0}};m.boundaryKind={BoundaryKind::Interface,BoundaryKind::Slip};m.boundaryPrimitive.resize(2);
 h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.area={1};h.surface.gasFace={0};h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.normal={{1,0,0}};h.surface.gasDistance={1};h.surface.solidDistance={1};
 program.interval.end=.01;program.surface=h.surface;WallKnot a,z;a.time=0;z.time=.01;
 WallFaceSample sample;sample.temperature=900;sample.primaryKind=ExchangeKind::GasSolid;a.faces={sample};z.faces={sample};program.knots={a,z};
 }
};

// Test-only comparison of independently executed physical candidates.
void samePhysicalGas(const HostState& a,const HostState& b){
 assert(a.gas.size()==b.gas.size()&&a.sst.size()==b.sst.size());
 for(size_t c=0;c<a.gas.size();++c){const auto& x=a.gas[c];const auto& y=b.gas[c];
  assert(x.mass==y.mass&&x.energy==y.energy&&x.momentum.x==y.momentum.x&&x.momentum.y==y.momentum.y&&x.momentum.z==y.momentum.z);
  for(int k=0;k<Ns;++k)assert(x.species[k]==y.species[k]);}
 for(size_t c=0;c<a.sst.size();++c)assert(a.sst[c].rhoK==b.sst[c].rhoK&&a.sst[c].rhoOmega==b.sst[c].rhoOmega);
 assert(a.gasMesh.volumes==b.gasMesh.volumes);
}
void checkPublishedSstInventory(Backend& b){
 const auto& v=b.storage.hostView();double k=0,omega=0,hostK=0,hostOmega=0;
 for(int c=0;c<b.nCells;++c){const double volume=b.trial.gasMesh.volumes[c];
  assert(b.trial.sst[c].rhoK==v.rhoK[c]*volume&&b.trial.sst[c].rhoOmega==v.rhoOmega[c]*volume);
  k+=v.rhoK[c]*volume;omega+=v.rhoOmega[c]*volume;hostK+=b.trial.sst[c].rhoK;hostOmega+=b.trial.sst[c].rhoOmega;}
 assert(k==hostK&&omega==hostOmega);
}

'''

def run_probe(tmp_path,body):
    source=HELPER['prepare_backend'](tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    text=stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',
        'inline int shimFailCopy=0; inline int cudaMemcpy(void*d,const void*s,size_t n,int){if(shimFailCopy>0&&--shimFailCopy==0)return 2;')
    stub.write_text(text)
    source.write_text(source.read_text()+FIXTURE+'\nint main(){'+body+'}\n')
    binary=tmp_path/'probe'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_failed_geometry_rollback_is_reported_and_poisoned(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));b->trial.gas[0].energy+=10;
 shimFailCopy=1;assert(!rollbackGasWindow(*b,f.error));assert(!f.error.empty());
 assert(b->accepted.gas[0].energy==f.h.gas[0].energy);
 assert(!beginGasWindow(*b,f.program,f.error));
 ''')

def test_single_active_species_uses_legacy_fields_without_species_allocation(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;assert(bindSingleGasModel(f.gas,"A",f.model,f.error));
 f.gas.mode=ugkwp::GasMode::SingleLegacy;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 const auto&v=b->storage.hostView();assert(!v.gasSpecies.rho&&!v.gasSpecies.flux&&!v.gasSpecies.soundSpeed);
 assert(v.Rgas==f.model.physics.species[0].R&&v.gasCp==1040&&v.gammaGas>1);
 assert(beginGasWindow(*b,f.program,f.error));b->microTime=0;b->microDt=.01;assert(transport::CoupledTrialPolicy::begin(b.get())==0);
 auto&w=b->storage.hostView();w.rho[0]=1;w.Tgas[0]=600;w.p[0]=w.Rgas*600;
 assert(applyCoupledFaces(b.get(),.01,0)==0);assert(accountCoupledFaces(b.get(),.01)==0);
 assert(rollbackGasWindow(*b,f.error));
 auto bad=f.h;bad.gas[0].species[1]=.1;bad.gas[0].species[0]=.9;
 assert(b->storage.uploadState(bad,f.error));
 ''')

def test_single_species_binding_keeps_math_and_admits_program_metadata(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;auto supplied=f.gas;supplied.coefficients[2]=1;
 assert(bindSingleGasModel(supplied,"A",f.model,f.error));
 assert(f.model.physics.species[0].e0==1&&f.model.physics.species[0].cp0==1040);
 assert(bindSingleGasModel(f.gas,"A",f.model,f.error));
 assert(f.model.physics.species[0].e0==0&&f.model.physics.species[0].cp0==1040);
 f.gas.mode=ugkwp::GasMode::SingleLegacy;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 f.program.knots.back().faces[0].speciesRate[1]=1;
 assert(beginGasWindow(*b,f.program,f.error));
 assert(b->program.knots.back().faces[0].speciesRate[1]==1);
 assert(b->history.interval().begin==f.program.interval.begin&&b->history.interval().end==f.program.interval.end);
 assert(b->history.records().empty()&&b->elapsed==WindowTime{});
 assert(b->trial.time==b->accepted.time&&b->trial.gas[0].energy==b->accepted.gas[0].energy);
 ''')

def test_nonzero_gas_gravity_is_explicitly_rejected(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;f.model.physics.gravity={0,-9.81,0};
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);
 assert(!b);assert(f.error.find("gravity")!=std::string::npos);
 ''')

def test_macro_commit_publishes_supplied_gas_candidate(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasIntervalRecord record;record.begin=0;record.end=.01;record.microSequence=1;
 assert(b->history.appendAccepted(record,f.error));b->trial.time=.01;auto candidate=b->trial;candidate.gas[0].energy+=1;
 assert(commitGasWindow(*b,candidate,f.error));assert(b->accepted.gas[0].energy==candidate.gas[0].energy);
 assert(rollbackGasWindow(*b,f.error));
 ''')

def prepare_sequential_launches(tmp_path):
    """Execute every common kernel thread serially on host, NOT CUDA evidence."""
    import re
    source=HELPER['prepare_backend'](tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text()+r'''
#include <vector>
inline std::vector<double> hostChemistryIntervals;
struct HostChemistryObservation {
 double dt=0;std::vector<double> volume,before,after,beforeEnergy,afterEnergy;
};
inline std::vector<HostChemistryObservation> hostChemistryObservations;
template<class State,class Time,class Volume,class Function>
void hostObserveChemistry(State* state,Time dt,const Volume* volumes,int speciesCount,Function body){
 HostChemistryObservation record;record.dt=double(dt);const int n=state->nCells;
 // Test-only snapshots of actual fields around the real translated kernel.
 for(int c=0;c<n;++c){record.volume.push_back(double(volumes[c]));record.beforeEnergy.push_back(double(state->rhoE[c]));}
 for(int k=0;k<speciesCount;++k)for(int c=0;c<n;++c)record.before.push_back(double(state->gasSpecies.rho[k*n+c]));
 body();
 for(int c=0;c<n;++c)record.afterEnergy.push_back(double(state->rhoE[c]));
 for(int k=0;k<speciesCount;++k)for(int c=0;c<n;++c)record.after.push_back(double(state->gasSpecies.rho[k*n+c]));
 hostChemistryIntervals.push_back(double(dt));hostChemistryObservations.push_back(record);
}
template<class Function> void hostLaunch(int grid,int block,Function body){
 const auto oldBlock=blockIdx,oldDim=blockDim,oldThread=threadIdx;blockDim.x=block;
 for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)for(threadIdx.x=0;threadIdx.x<block;++threadIdx.x)body();
 blockIdx=oldBlock;blockDim=oldDim;threadIdx=oldThread;
}
''')
    pattern=r'([\w:]+(?:<[^<>]+>)?)\s*<<<\s*([^,]+),\s*([^,>]+)(?:,[\s\S]*?)?>>>\s*(\([\s\S]*?\));'
    for name in ('GpuGasAdvance.cuh','GpuGasCourant.cuh'):
        original=(APP.parents[1]/'common'/name).read_text()
        def translate(m):
            launch=f'hostLaunch({m[2]}, {m[3]}, [&](){{{m[1]}{m[4]};}});'
            if m[1]=='advanceGasChemistryKernel':
                interval=re.search(r'\(\s*s->deviceState\s*,\s*([^,]+),',m[4])
                assert interval, m[4]
                volume=re.search(r'TrialPolicy::stageVolumes\(s,(?:false|true)\)',m[4])
                assert volume, m[4]
                launch=('hostObserveChemistry(s->deviceState,'+interval[1]+','+volume[0]
                        +',ugkwp::GasStateTraits<GasHostState>::speciesCount,[&](){'+launch+'});')
            return launch
        translated=re.sub(pattern,translate,original)
        assert '<<<' not in translated
        (tmp_path/'tree/common'/name).write_text(translated)
    return source

def test_complete_common_microstep_sequence_and_rollback_on_host(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 for(bool single:{false,true})for(int integrator:{1,2,3}){
 Fixture f;f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 for(auto& knot:f.program.knots)knot.faces[0].temperature=300;
 if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 GasExecutionOptions options;options.timeIntegrator=integrator;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(b->trial.time==1e-5&&b->accepted.time==0);assert(b->trial.gas[0].energy<f.h.gas[0].energy);
 const auto& record=b->history.records().front();double exchange=0;
 for(const auto& packet:record.packets)exchange+=packetDelta(packet).gas.energy;
 assert(std::abs(b->trial.gas[0].energy-f.h.gas[0].energy-exchange)<1e-9);
 // The preceding assertion independently closes actual energy against packets.
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gas[0].energy==f.h.gas[0].energy);
 }
}
''')
    binary=tmp_path/'pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_common_sst_microstep_matches_direct_physical_state_and_rolls_back(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 for(bool single:{false,true})for(int integrator:{1,2,3}){
 Fixture f;f.model.physics.enableSst=true;f.model.physics.gasViscosity=1e-5;
 f.h.sst={{1,10}};f.h.gasMesh.wallDistance={1};f.h.gasMesh.boundarySst.resize(2);
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 GasExecutionOptions options;options.timeIntegrator=integrator;options.turbulenceModel=3;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);
 if(!b){std::cerr<<f.error<<'\n';return 1;}
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 std::unique_ptr<Backend,void(*)(Backend*)> reference(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(reference);
 assert(beginGasWindow(*reference,f.program,f.error));reference->microTime=0;reference->microDt=report.acceptedDt;
 assert(transport::advanceGasTrial<transport::CoupledTrialPolicy>(reference.get(),report.acceptedDt,0)==0);
 samePhysicalGas(b->trial,reference->trial);checkPublishedSstInventory(*b);checkPublishedSstInventory(*reference);
 assert(b->trial.sst[0].rhoK!=f.h.sst[0].rhoK||b->trial.sst[0].rhoOmega!=f.h.sst[0].rhoOmega);
 assert(rollbackGasWindow(*b,f.error));samePhysicalGas(b->trial,f.h);
 }
}
''')
    binary=tmp_path/'sst-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_coupled_moving_geometry_preserves_physical_inventory_and_rollback(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
HostMesh box(double low,double high){
 HostMesh m;m.points={{0,0,low},{1,0,low},{1,1,low},{0,1,low},{0,0,high},{1,0,high},{1,1,high},{0,1,high}};
 m.faceOffsets={0,4,8,12,16,20,24};m.facePoints={0,3,2,1,4,5,6,7,0,4,7,3,1,2,6,5,0,1,5,4,3,7,6,2};
 m.owner.assign(6,0);m.neighbour.assign(6,-1);m.boundaryKind.assign(6,BoundaryKind::Slip);
 m.periodicPartner.assign(6,-1);m.boundaryPrimitive.resize(6);m.boundarySst.resize(6);
 std::string error;assert(rebuildGeometry(m,error));return m;
}
int main(){
 for(bool single:{false,true})for(bool sst:{false,true})for(int integrator:{1,2,3})for(bool diagnosed:{false,true}){
 Fixture f;f.h.gasMesh=box(0,1);f.h.solidMesh=box(-1,0);f.h.gasMesh.boundaryKind[0]=BoundaryKind::Interface;
 f.h.solidMesh.boundaryKind[1]=BoundaryKind::Interface;
 f.h.surface.gasFace={0};f.h.surface.solidFace={1};f.h.surface.normal={{0,0,1}};
 f.h.surface.centre={{.5,.5,0}};f.h.surface.baseVelocity={{0,0,0}};f.h.surface.meshVelocity={{0,0,0}};
 f.h.surface.oldArea=f.h.surface.area;f.h.surface.gasDistance={.5};f.h.surface.solidDistance={.5};
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 f.model.physics.meshMotion.policy=MeshMotionPolicy::CoupledRecession;f.model.physics.enableSst=sst;
 if(diagnosed)f.model.physics.tolerances.absoluteGeometry=-1;
 if(sst){f.h.sst={{1,10}};f.model.physics.gasViscosity=1e-5;}
 f.program.surface=f.h.surface;f.program.interval.end=1e-5;
 for(auto& knot:f.program.knots){knot.gasPoints=f.h.gasMesh.points;knot.solidPoints=f.h.solidMesh.points;
 knot.faces[0].normalVelocity=knot.faces[0].solidNormalVelocity=-.001;knot.faces[0].temperature=600;}
 auto& end=f.program.knots.back();end.time=1e-5;
 for(auto& p:end.gasPoints)if(p.z==0)p.z-=1e-8;
 for(auto& p:end.solidPoints)if(p.z==0)p.z-=1e-8;
 if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 GasExecutionOptions options;options.turbulenceModel=sst?3:0;options.timeIntegrator=integrator;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);
 if(!b){std::cerr<<f.error<<'\n';return 1;}
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.acceptedDt==1e-5);assert(b->trial.gasMesh.volumes[0]>f.h.gasMesh.volumes[0]);
 assert(std::abs(b->trial.gas[0].mass-f.h.gas[0].mass)<1e-12);
 double energy=0;for(const auto& packet:b->history.records()[0].packets)energy+=packetDelta(packet).gas.energy;
 assert(std::abs(b->trial.gas[0].energy-f.h.gas[0].energy-energy)<1e-8);
 if(sst){
  std::unique_ptr<Backend,void(*)(Backend*)> reference(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(reference);
  assert(beginGasWindow(*reference,f.program,f.error));reference->microTime=0;reference->microDt=report.acceptedDt;
  assert(transport::advanceGasTrial<transport::CoupledTrialPolicy>(reference.get(),report.acceptedDt,0)==0);
  samePhysicalGas(b->trial,reference->trial);checkPublishedSstInventory(*b);checkPublishedSstInventory(*reference);
 }
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gasMesh.volumes==f.h.gasMesh.volumes);
 assert(b->storage.hostView().gasGeometry.enabled==false);assert(b->trial.gas[0].energy==f.h.gas[0].energy);
 if(!single&&!sst&&integrator==1){assert(beginGasWindow(*b,f.program,f.error));for(auto& knot:b->program.knots)knot.gasPoints.clear();
  assert(!advanceGasMicrostep(*b,1e-5,report,f.error));assert(!b->poisoned);}
 }
}
''')
    binary=tmp_path/'moving-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_current_comax_selects_before_single_advance(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 Fixture f;f.model.physics.cfl=.001;
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,.001,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.rejectedTrials==0);assert(report.acceptedDt<.001);assert(b->trial.rejectedSteps==0);
 assert(report.nextGasDt==1.05*report.acceptedDt);assert(b->history.records().size()==1);
 assert(b->accepted.rejectedSteps==0);assert(rollbackGasWindow(*b,f.error));assert(b->trial.rejectedSteps==0);
}
''')
    binary=tmp_path/'retry-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_complete_gross_donor_payload_continues_without_gas_retry(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 Fixture f;f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 f.h.solid[0].pore[0]=10;
 for(auto& knot:f.program.knots){knot.faces[0].temperature=600;knot.faces[0].speciesRate[0]=-2e5;
 knot.faces[0].condensedRate[0]=-2e5;knot.faces[0].poreRate[0]=2e5;}
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<std::endl;return 1;}
 assert(report.rejectedTrials==0&&b->history.records().size()==1);
 assert(std::abs(b->trial.gas[0].mass-1)<1e-12);
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gas[0].mass==1);
 // A supplied pore-sweep transfer follows the existing actual transport and reservation.
 for(auto& knot:f.program.knots){knot.faces[0].speciesRate[0]=knot.faces[0].condensedRate[0]=0;
 knot.faces[0].poreRate[0]=0;knot.faces[0].poreSweepRate[0]=-2e5;}
 assert(beginGasWindow(*b,f.program,f.error));
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<std::endl;return 1;}
 assert(report.rejectedTrials==0&&b->history.records().size()==1);
}
''')
    binary=tmp_path/'gross-donor'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_failed_device_restore_after_missing_material_owner_cannot_retry(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',
        'inline bool shimInjected=false;inline int shimFailCopy=0;inline int cudaMemcpy(void*d,const void*s,size_t n,int){if(shimFailCopy>0&&--shimFailCopy==0)return 2;'))
    # Test-only one-shot fault at the actual rejected reservation boundary.
    # No instrumentation or fault controls are added to production classes.
    text=source.read_text();needle='if(!b.reserve.prepare(b.record,reservation,b.error)){'
    assert text.count(needle)==1
    text=text.replace(needle,'b.record.packets[0].solidCell=99;'+needle+'if(!shimInjected){shimInjected=true;shimFailCopy=1;}')
    source.write_text(text+FIXTURE+r'''
int main(){
 Fixture f;f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 f.h.solid[0].pore[0]=1e-6;
 for(auto& knot:f.program.knots){knot.faces[0].temperature=600;knot.faces[0].poreRate[0]=1;}
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 assert(!advanceGasMicrostep(*b,1e-5,report,f.error));assert(b->poisoned&&!report.recoverable);
 assert(f.error.find("rollback failed")!=std::string::npos);
 assert(b->accepted.time==0&&b->accepted.gas[0].mass==1&&b->history.records().empty());
}
''')
    binary=tmp_path/'poisoned-reservation'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_comax_estimate_restores_sst_and_preserves_fixed_step_outputs(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',
        'inline size_t d2dBytes=0;inline int d2dCopies=0;inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){if(kind==cudaMemcpyDeviceToDevice){d2dBytes+=n;++d2dCopies;}'))
    stub.write_text(stub.read_text().replace('inline int cudaGetLastError(){return 0;}',
        'inline int launchCalls=0,failLaunch=0;inline int cudaGetLastError(){++launchCalls;if(failLaunch&&--failLaunch==0)return 2;return 0;}'))
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 for(bool single:{false,true})for(bool sst:{false,true})for(int integrator:{1,2,3}){
  Fixture f;f.model.physics.enableSst=sst;
  f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
  if(sst){f.model.physics.gasViscosity=1e-5;f.h.sst={{1,10}};f.h.gasMesh.wallDistance={1};f.h.gasMesh.boundarySst.resize(2);}
  if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
  GasExecutionOptions options;options.timeIntegrator=integrator;options.turbulenceModel=sst?3:0;
  std::unique_ptr<Backend,void(*)(Backend*)> direct(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);
  std::unique_ptr<Backend,void(*)(Backend*)> selected(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);
  assert(direct&&selected);assert(beginGasWindow(*direct,f.program,f.error));assert(beginGasWindow(*selected,f.program,f.error));
  auto& v=selected->storage.hostView();
  const Real q[]={v.rho[0],v.rhoUx[0],v.rhoUy[0],v.rhoUz[0],v.rhoE[0],v.rhoK[0],v.rhoOmega[0]};
  selected->microBase=selected->trial;selected->microDt=1e-5;selected->microTime=0;assert(prepareCoupledGeometry(selected.get())==0);
  d2dBytes=0;d2dCopies=0;transport::GasCourantMaxima maxima;
  assert(currentGasCourant(*selected,1e-5,maxima));assert(maxima.maximum>=0);
  assert(v.rho[0]==q[0]&&v.rhoUx[0]==q[1]&&v.rhoUy[0]==q[2]&&v.rhoUz[0]==q[3]&&v.rhoE[0]==q[4]);
  assert(v.rhoK[0]==q[5]&&v.rhoOmega[0]==q[6]);assert(d2dCopies==(sst?4:0));assert(d2dBytes==(sst?32:0));
  // Reset only this diagnostic estimate before the production path below.
  assert(rollbackGasWindow(*selected,f.error));assert(beginGasWindow(*selected,f.program,f.error));
  direct->microTime=0;direct->microDt=1e-5;
  assert(transport::advanceGasTrial<transport::CoupledTrialPolicy>(direct.get(),1e-5,0)==0);
  GasMicroReport report;if(!advanceGasMicrostep(*selected,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
  assert(report.acceptedDt==1e-5&&report.rejectedTrials==0);
  const auto& a=direct->trial.gas[0];const auto& b=selected->trial.gas[0];
  assert(a.mass==b.mass&&a.energy==b.energy&&a.momentum.x==b.momentum.x&&a.momentum.y==b.momentum.y&&a.momentum.z==b.momentum.z);
  for(int i=0;i<Ns;++i)assert(a.species[i]==b.species[i]);
  if(sst){assert(direct->trial.sst[0].rhoK==selected->trial.sst[0].rhoK&&direct->trial.sst[0].rhoOmega==selected->trial.sst[0].rhoOmega);
   checkPublishedSstInventory(*direct);checkPublishedSstInventory(*selected);}
  assert(direct->record.packets.size()==selected->record.packets.size());
  for(size_t i=0;i<direct->record.packets.size();++i){const auto& x=direct->record.packets[i];const auto& y=selected->record.packets[i];
   assert(x.mass==y.mass&&x.energy==y.energy&&x.conductive==y.conductive&&x.advective==y.advective&&x.pressureWork==y.pressureWork);
   for(int j=0;j<Ns;++j)assert(x.species[j]==y.species[j]&&x.pore[j]==y.pore[j]&&x.poreSweep[j]==y.poreSweep[j]);}
 }
 {
  Fixture f;f.h.time=1099511627776.0;f.program.interval.begin=f.h.time;f.program.interval.end=f.h.time+1.0/4096;
  f.program.knots.front().time=f.h.time;f.program.knots.back().time=f.program.interval.end;
  f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
  for(auto& knot:f.program.knots)knot.faces[0].temperature=300;
  std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
  assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
  for(int i=0;i<3;++i){const double before=b->trial.gas[0].energy;
   assert(advanceGasMicrostep(*b,1e-6,report,f.error));assert(report.time==f.h.time);assert(report.acceptedDt==1e-6);
   assert(report.localTime>WindowTime{}&&b->trial.gas[0].energy<before);}
  assert(b->history.records().size()==3);assert(rollbackGasWindow(*b,f.error));
  assert(beginGasWindow(*b,f.program,f.error));launchCalls=0;failLaunch=1;
  assert(!advanceGasMicrostep(*b,1e-6,report,f.error));assert(!report.recoverable&&report.rejectedTrials==0);
  assert(launchCalls==1&&b->history.records().empty());assert(b->trial.gas[0].energy==f.h.gas[0].energy);
  assert(f.error.find("recoverGasPrimitivesKernel launch for gas Courant")!=std::string::npos);
 }
}
''')
    binary=tmp_path/'comax-identity'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_all_rk_trace_stamps_follow_paid_local_stage_time(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 for(double origin:{0.,1.,1099511627776.0})for(int integrator:{1,2,3})for(double dt:{1e-6,1e-5,.00015,.0002}){
  Fixture f;f.h.time=origin;f.program.interval.begin=f.h.time;f.program.interval.end=f.h.time+.01;
  f.program.knots.front().time=f.h.time;f.program.knots.back().time=f.program.interval.end;
  f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
  GasExecutionOptions options;options.timeIntegrator=integrator;
  std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(b);
  assert(beginGasWindow(*b,f.program,f.error));
  for(int step=0;step<12;++step){GasMicroReport report;
   if(!advanceGasMicrostep(*b,dt,report,f.error)){std::cerr<<"RK="<<integrator<<" dt="<<dt<<" step="<<step<<": "<<f.error<<'\n';return 1;}
   assert(report.acceptedDt==dt);const auto& r=b->history.records().back();
   const auto stage=r.localBegin+(integrator==1?0:integrator==2?dt:.5*dt);
   assert(r.gasTraceTime==windowAbsolute(f.h.time,stage));
  }
 }
}
''')
    binary=tmp_path/'rk-local-trace'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
