"""Computed payload consumers, executed with host CUDA shims, not GPU evidence."""
from pathlib import Path
import runpy
import subprocess
APP=Path(__file__).resolve().parents[1]
RUNTIME=runpy.run_path(str(APP/'tests/test_coupled_runtime_transactions.py'))

def test_downloaded_physical_payload_and_sst_inventory_are_consumed(tmp_path):
    RUNTIME['run_probe'](tmp_path,r'''
 Fixture f;f.model.physics.enableSst=true;f.h.sst={{1,10}};f.h.gasMesh.wallDistance={1};f.h.gasMesh.boundarySst.resize(2);
 GasExecutionOptions options;options.turbulenceModel=3;
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},options,f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));b->microTime=0;b->microDt=.01;assert(transport::CoupledTrialPolicy::begin(b.get())==0);
 auto& v=b->storage.hostView();v.rho[0]=-2;v.rhoE[0]=std::numeric_limits<Real>::infinity();v.rhoUx[0]=-7;
 v.gasSpecies.rho[0]=-3;v.rhoK[0]=-4;v.rhoOmega[0]=0;
 assert(transport::CoupledTrialPolicy::validate(b.get())==0);
 assert(transport::CoupledTrialPolicy::commit(b.get())==0);
 assert(b->trial.gas[0].mass==-2&&std::isinf(b->trial.gas[0].energy)&&b->trial.gas[0].momentum.x==-7);
 assert(b->trial.gas[0].species[0]==-3&&b->trial.sst[0].rhoK==-4&&b->trial.sst[0].rhoOmega==0);
 // Publication consumes the actual conservative state, not an audit total.
 assert(transport::CoupledTrialPolicy::commit(b.get())==0);
 const Real volume=b->trial.gasMesh.volumes[0];
 assert(b->trial.gas[0].mass==v.rho[0]*volume&&b->trial.gas[0].momentum.x==v.rhoUx[0]*volume);
 assert(b->trial.gas[0].species[0]==v.gasSpecies.rho[0]*volume);
 assert(b->trial.sst[0].rhoK==v.rhoK[0]*volume&&b->trial.sst[0].rhoOmega==v.rhoOmega[0]*volume);
 HostState output=b->trial;output.gas[0].mass=123;shimFailCopy=1;assert(!b->storage.downloadState(output,f.error));assert(output.gas[0].mass==123);
 // A real physical-state copy failure remains observable and retryable.
 const auto before=b->trial;shimFailCopy=1;assert(transport::CoupledTrialPolicy::commit(b.get())!=0);
 assert(b->trial.gas[0].mass==before.gas[0].mass&&b->trial.sst[0].rhoK==before.sst[0].rhoK);
 assert(transport::CoupledTrialPolicy::commit(b.get())==0);
 ''')

def test_actual_packet_scaling_keeps_outside_unit_ratios(tmp_path):
    RUNTIME['run_probe'](tmp_path,r'''
 for(double scale:{.5,2.,-1.}){
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));b->microTime=0;b->microDt=.01;assert(transport::CoupledTrialPolicy::begin(b.get())==0);
 auto&v=b->storage.hostView();v.rho[0]=1;v.Tgas[0]=600;v.p[0]=f.model.physics.species[0].R*600;v.gasSpecies.soundSpeed[0]=400;
 assert(applyCoupledFaces(b.get(),.01,0)==0);const auto expected=scaledIntervalPacket(b->faceRates[0].primary,scale);
 v.gasPhiRhoE[0]*=scale;v.gasPhiRhoUx[0]*=scale;v.gasPhiRhoUy[0]*=scale;v.gasPhiRhoUz[0]*=scale;v.gasPhiRho[0]*=scale;
 for(int s=0;s<Ns;++s)v.gasSpecies.flux[s*b->nFaces]*=scale;
 assert(accountCoupledFaces(b.get(),.01)==0);const auto&p=b->record.packets[0];
 assert(p.energy==expected.energy&&p.mass==expected.mass&&p.conductive==expected.conductive&&p.advective==expected.advective);
 for(int s=0;s<Ns;++s)assert(p.species[s]==expected.species[s]);
 assert(p.momentum.x==expected.momentum.x&&p.momentum.y==expected.momentum.y&&p.momentum.z==expected.momentum.z);
 assert(b->gasStageIndex==1);
 shimFailCopy=1;assert(accountCoupledFaces(b.get(),.01)!=0);assert(b->gasStageIndex==1);
 }
 ''')

def test_history_preserves_supplied_identity_quality_and_first_receiver(tmp_path):
    owner=(APP/'materials/CpuMaterialDriver.C').read_text()
    select=next(line.strip() for line in owner.splitlines() if 'for(const auto& r:history.records())if(elapsed<r.localEnd)' in line)
    consume=next(line.strip() for line in owner.splitlines() if 'trial.solid[c].energy+=radiation[f]' in line)
    source=r'''
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"
#include <cassert>
#include <cmath>
using namespace chmt;
int main(){std::string error;CouplingInterval w;w.end=1;IntervalHistory history;assert(history.begin(w,error));
 GasIntervalRecord a;a.begin=0;a.end=.5;a.microSequence=77;a.radiationEnergy={3};a.radiationReceiver={ConsumeSolid};
 ExchangePacket p;p.step=999;p.stage=0;p.geometry=123;p.mass=2;p.energy=9;p.consumerMask=0;a.packets={p,p};
 a.gasTrace.resize(1);a.gasTrace[0].temperature=std::numeric_limits<Real>::quiet_NaN();
 a.gasGradient.resize(1);a.gasGradient[0].rho.x=std::numeric_limits<Real>::infinity();a.gasTraction={{-1,2,3}};
 assert(history.appendAccepted(a,error));auto b=a;b.begin=.5;b.end=1;b.radiationEnergy={7};b.radiationReceiver={ConsumeFilm};
 assert(history.appendAccepted(b,error));assert(history.records()[1].microSequence==77);
 assert(history.records()[0].packets.size()==2&&history.records()[0].packets[0].step==999&&history.records()[0].packets[0].consumerMask==0);
 std::vector<GasPrimitive> trace;std::vector<GasGradient> gradient;std::vector<Vec3> traction;
 assert(history.sampleGasDrive(.25,trace,gradient,traction,error));assert(std::isnan(trace[0].temperature)&&std::isinf(gradient[0].rho.x));
 auto cursor=history.cursor();std::vector<ExchangePacket> packets;std::vector<Real> radiation;
 assert(cursor.takeThrough(1,packets,radiation,error));assert(packets.size()==4&&radiation[0]==10);
 Real mass=0;for(const auto& q:packets)mass+=q.mass;assert(mass==8);
 WindowTime elapsed{};const GasIntervalRecord* driveRecord=nullptr;
 SELECT
 HostState state,trial;state.surface.solidCell={0};trial.solid.resize(1);
 const bool terminalMaterial=false;MaterialFailureOrigin lastFailure=MaterialFailureOrigin::None;
 CONSUME
 assert(error.empty()&&trial.solid[0].energy==10);
}
'''.replace('SELECT',select).replace('CONSUME',consume)
    src=tmp_path/'history.cpp';src.write_text(source);exe=tmp_path/'history'
    subprocess.run(['g++','-std=c++14','-O2','-I'+str(APP),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
