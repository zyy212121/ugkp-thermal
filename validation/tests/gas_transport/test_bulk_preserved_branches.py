"""Real preserved branch calls, schedules and sparse worker coverage."""
import subprocess
import pytest
from test_mixture_state import ROOT,HERE,compile_probe
from test_legacy_ale_transport import fixture

def test_schedule_knots_endpoints_and_actual_boundary_mirror(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Time times[3]={0,1,2};Real values[3]={10,20,40};s.nPressureScheduleRows=3;s.pressureScheduleTimes=times;s.pressureScheduleValues=values;s.nScheduledInletFaces=1;s.scheduledInletTemperature=300;s.scheduledInletFaceMask[1]=1;
const Time ts[7]={-1,0,.5,1,1.5,2,3};const Real wanted[7]={10,10,15,20,30,40,40};
for(int i=0;i<7;++i){ck(linearScheduledValueDevice(times,values,3,ts[i])==wanted[i],"schedule interpolation changed");threadIdx.x=1;updateLegacyGasBoundaryMirrorKernel(&s,ts[i]);ck(s.gasBoundaryP[1]==wanted[i]&&s.riemannBoundaryP[1]==wanted[i]&&s.riemannBoundaryT[1]==300,"same-stage scheduled boundary missing");}
}
''')

def test_invariant_calls_are_branch_local_without_persistent_reuse(tmp_path):
    header=(ROOT/'common/operators/sstVelocityInvariants.cuh').read_text().replace(')\n{',')\n{\n    ++invariantCalls;',1)
    counted=tmp_path/'invariants.cuh';counted.write_text(header)
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    pre=pre.replace('#include "operators/sstVelocityInvariants.cuh"','static int invariantCalls=0;\n#include "'+str(counted)+'"')
    source=tmp_path/'calls.cpp';source.write_text(pre+fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;s.nut[0]=-77;
s.turbulenceModel=0;computeGasEddyViscosityKernel(&s);ck(invariantCalls==0&&s.nut[0]==0,"laminar performed invariant work or omitted nut write");
s.turbulenceModel=1;computeGasEddyViscosityKernel(&s);ck(invariantCalls==0&&std::isfinite(s.nut[0]),"LES branch changed");
s.turbulenceModel=3;s.sstConfigured=1;computeGasEddyViscosityKernel(&s);ck(invariantCalls==1&&std::isfinite(s.nut[0]),"SST eddy call count changed");
Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);ck(invariantCalls==2&&std::isfinite(sk)&&std::isfinite(so),"source invariants incomplete");
sstKSourceActivity(s,0,sk,Real(0));ck(invariantCalls==2,"ordinary activity repeated invariants");
int ownerSlot[2]={0,-1};s.gasBoundaryLayer.enabled=true;s.gasBoundaryLayer.count=1;s.gasBoundaryLayer.ownerSlot=ownerSlot;sstKSourceActivity(s,0,sk,Real(0));ck(invariantCalls==2,"selected-wall activity repeated constitutive work");}
''')
    exe=tmp_path/'calls';r=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stderr
    r=subprocess.run([str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr

@pytest.mark.parametrize('analytic',[False,True])
def test_actual_sparse_kernel_covers_faces_once_per_scratch_slot(tmp_path,analytic):
    header=(ROOT/'common/operators/evaluateGasBoundaryLayerKernel.cuh').read_text()
    header=header.replace('ugkwp::evaluateGasBoundaryLayerSlot(*state,slot,worker)','recordSlot(*state,slot,worker)')
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/GasBoundaryLayerModelState.H"
struct WorkerState:State{ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;};
int visits[19]={},slots[19]={};void recordSlot(WorkerState&,int face,int worker){++visits[face];slots[face]=worker;}
'''+header+r'''
int main(){WorkerState s;initialise(s);s.gasBoundaryLayer.enabled=true;s.gasBoundaryLayer.count=19;s.gasBoundaryLayerModel.workspaceCount=3;
s.gasBoundaryLayerModel.config.model=ANALYTIC?ugkwp::gaswall::BoundaryLayerModel::ConstantTransport:ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
for(int worker=0;worker<32;++worker){threadIdx.x=worker;evaluateGasBoundaryLayerKernel(&s);}for(int face=0;face<19;++face)ck(visits[face]==1&&slots[face]==(ANALYTIC?face:face%3),"sparse wall face/slot coverage changed");}
'''.replace('ANALYTIC','true' if analytic else 'false'))
