"""Actual backend hooks with host CUDA allocation shim, never GPU evidence."""
from pathlib import Path
import runpy
import subprocess

APP=Path(__file__).resolve().parents[1]
HELPER=runpy.run_path(str(APP/'tests/test_shared_backend_compile.py'))
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
 assert(!b->storage.uploadState(bad,f.error));
 ''')

def test_single_species_binding_rejects_energy_mismatch_and_foreign_emission(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;auto bad=f.gas;bad.coefficients[2]=1;
 assert(!bindSingleGasModel(bad,"A",f.model,f.error));
 assert(bindSingleGasModel(f.gas,"A",f.model,f.error));
 f.gas.mode=ugkwp::GasMode::SingleLegacy;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 f.program.knots.back().faces[0].speciesRate[1]=1;
 assert(!beginGasWindow(*b,f.program,f.error));assert(!b->pending);
 ''')

def test_nonzero_gas_gravity_is_explicitly_rejected(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;f.model.physics.gravity={0,-9.81,0};
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);
 assert(!b);assert(f.error.find("gravity")!=std::string::npos);
 ''')

def test_macro_commit_rejects_changed_gas_candidate(tmp_path):
    run_probe(tmp_path,r'''
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasIntervalRecord record;record.begin=0;record.end=.01;record.microSequence=1;
 assert(b->history.appendAccepted(record,f.error));b->trial.time=.01;auto candidate=b->trial;candidate.gas[0].energy+=1;
 assert(!commitGasWindow(*b,candidate,f.error));assert(b->accepted.gas[0].energy==f.h.gas[0].energy);
 assert(rollbackGasWindow(*b,f.error));
 ''')

def prepare_sequential_launches(tmp_path):
    """Execute every common kernel thread serially on host, NOT CUDA evidence."""
    import re
    source=HELPER['prepare_backend'](tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text()+r'''
template<class Function> void hostLaunch(int grid,int block,Function body){
 const auto oldBlock=blockIdx,oldDim=blockDim,oldThread=threadIdx;blockDim.x=block;
 for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)for(threadIdx.x=0;threadIdx.x<block;++threadIdx.x)body();
 blockIdx=oldBlock;blockDim=oldDim;threadIdx=oldThread;
}
''')
    common=APP.parents[1]/'common/GpuGasAdvance.cuh'
    pattern=r'([\w:]+(?:<[^<>]+>)?)\s*<<<\s*([^,]+),\s*([^,>]+)(?:,[\s\S]*?)?>>>\s*(\([\s\S]*?\));'
    original=common.read_text()
    translated=re.sub(pattern,lambda m:f'hostLaunch({m[2]}, {m[3]}, [&](){{{m[1]}{m[4]};}});',original)
    assert '<<<' not in translated
    (tmp_path/'tree/common/GpuGasAdvance.cuh').write_text(translated)
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
 assert(std::abs(b->trial.budget.exchangeEnergy[GasParticipant]-exchange)<1e-12);
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gas[0].energy==f.h.gas[0].energy);
 }
}
''')
    binary=tmp_path/'pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_common_sst_microstep_has_closed_integral_audit_and_rolls_back(tmp_path):
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
 const auto& audit=b->trial.gasSstAudit;
 assert(std::abs(b->trial.sst[0].rhoK-f.h.sst[0].rhoK-audit.transportK-audit.sourceK-audit.constraintK)<1e-10);
 assert(std::abs(b->trial.sst[0].rhoOmega-f.h.sst[0].rhoOmega-audit.transportOmega-audit.sourceOmega-audit.constraintOmega)<1e-9);
 assert(audit.sourceK!=0||audit.constraintOmega!=0);
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.sst[0].rhoK==1&&b->trial.gasSstAudit.sourceK==0);
 }
}
''')
    binary=tmp_path/'sst-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_coupled_moving_geometry_uses_common_volume_audit_and_rollback(tmp_path):
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
 for(bool single:{false,true})for(bool sst:{false,true}){
 Fixture f;f.h.gasMesh=box(0,1);f.h.solidMesh=box(-1,0);f.h.gasMesh.boundaryKind[0]=BoundaryKind::Interface;
 f.h.solidMesh.boundaryKind[1]=BoundaryKind::Interface;
 f.h.surface.gasFace={0};f.h.surface.solidFace={1};f.h.surface.normal={{0,0,1}};
 f.h.surface.centre={{.5,.5,0}};f.h.surface.baseVelocity={{0,0,0}};f.h.surface.meshVelocity={{0,0,0}};
 f.h.surface.oldArea=f.h.surface.area;f.h.surface.gasDistance={.5};f.h.surface.solidDistance={.5};
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 f.model.physics.meshMotion.policy=MeshMotionPolicy::CoupledRecession;f.model.physics.enableSst=sst;
 if(sst){f.h.sst={{1,10}};f.model.physics.gasViscosity=1e-5;}
 f.program.surface=f.h.surface;f.program.interval.end=1e-5;
 for(auto& knot:f.program.knots){knot.gasPoints=f.h.gasMesh.points;knot.solidPoints=f.h.solidMesh.points;
 knot.faces[0].normalVelocity=knot.faces[0].solidNormalVelocity=-.001;knot.faces[0].temperature=600;}
 auto& end=f.program.knots.back();end.time=1e-5;
 for(auto& p:end.gasPoints)if(p.z==0)p.z-=1e-8;
 for(auto& p:end.solidPoints)if(p.z==0)p.z-=1e-8;
 if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 GasExecutionOptions options;options.turbulenceModel=sst?3:0;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);
 if(!b){std::cerr<<f.error<<'\n';return 1;}
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.acceptedDt==1e-5);assert(b->trial.gasMesh.volumes[0]>f.h.gasMesh.volumes[0]);
 assert(std::abs(b->trial.gas[0].mass-f.h.gas[0].mass)<1e-12);
 double energy=0;for(const auto& packet:b->history.records()[0].packets)energy+=packetDelta(packet).gas.energy;
 assert(std::abs(b->trial.gas[0].energy-f.h.gas[0].energy-energy)<1e-8);
 if(sst){const auto& a=b->trial.gasSstAudit;
 assert(std::abs(b->trial.sst[0].rhoK-f.h.sst[0].rhoK-a.transportK-a.sourceK-a.constraintK)<1e-10);}
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gasMesh.volumes==f.h.gasMesh.volumes);
 assert(b->storage.hostView().gasGeometry.enabled==false);assert(b->trial.gas[0].energy==f.h.gas[0].energy);
 }
}
''')
    binary=tmp_path/'moving-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_rejected_gas_attempts_are_counted_only_in_committed_microstate(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    source.write_text(source.read_text()+FIXTURE+r'''
#include <iostream>
int main(){
 Fixture f;f.model.physics.cfl=.001;
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));GasMicroReport report;
 if(!advanceGasMicrostep(*b,.001,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.rejectedTrials>0);assert(b->trial.rejectedSteps==std::uint64_t(report.rejectedTrials));
 assert(b->accepted.rejectedSteps==0);assert(rollbackGasWindow(*b,f.error));assert(b->trial.rejectedSteps==0);
}
''')
    binary=tmp_path/'retry-pipeline'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_gross_wall_counterflow_cannot_borrow_incoming_gas(tmp_path):
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
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.rejectedTrials>0&&report.acceptedDt<1e-5);
 double withdrawn=0;for(const auto& packet:b->history.records()[0].packets)withdrawn+=gasPacketSpeciesWithdrawal(packet,0);
 assert(withdrawn<=1+1e-12);assert(std::abs(b->trial.gas[0].mass-1)<1e-12);
 assert(rollbackGasWindow(*b,f.error));assert(b->trial.gas[0].mass==1);
 // Darcy inflow cannot fund a distinct geometric pore-sweep withdrawal either.
 for(auto& knot:f.program.knots){knot.faces[0].speciesRate[0]=knot.faces[0].condensedRate[0]=0;
 knot.faces[0].poreRate[0]=0;knot.faces[0].poreSweepRate[0]=-2e5;}
 assert(beginGasWindow(*b,f.program,f.error));
 if(!advanceGasMicrostep(*b,1e-5,report,f.error)){std::cerr<<f.error<<'\n';return 1;}
 assert(report.rejectedTrials>0&&report.acceptedDt<1e-5);
}
''')
    binary=tmp_path/'gross-donor'
    subprocess.run(HELPER['compiler_flags'](tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_failed_device_restore_after_material_reservation_cannot_retry(tmp_path):
    source=prepare_sequential_launches(tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text().replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){',
        'inline bool shimInjected=false;inline int shimFailCopy=0;inline int cudaMemcpy(void*d,const void*s,size_t n,int){if(shimFailCopy>0&&--shimFailCopy==0)return 2;'))
    # Test-only one-shot fault at the actual rejected reservation boundary.
    # No instrumentation or fault controls are added to production classes.
    text=source.read_text();needle='if(!b.reserve.prepare(b.record,reservation,b.error)){'
    assert text.count(needle)==1
    text=text.replace(needle,needle+'if(!shimInjected){shimInjected=true;shimFailCopy=1;}')
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
