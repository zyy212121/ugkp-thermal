"""Current resident interval policy: device status, completion and history commit.

The removed rollback/retry/IPC contracts belonged to a historical backend.
This host harness executes the actual policy with explicit CUDA boundary stubs;
it makes no device arithmetic or GPU performance claim.
"""
from validation.tools.paths import SOURCE_ROOT
import re
import subprocess


def test_interval_policy_observes_failure_before_public_completion(tmp_path):
    policy = (SOURCE_ROOT / "applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh").read_text()
    policy = policy[policy.index("struct SharedGasTrialPolicy"):]
    policy = re.sub(r"<<<.*?>>>", "", policy)
    code = r'''
#include <cassert>
#include <vector>
using cudaError_t=int;constexpr int cudaSuccess=0;
int enqueueResult=0,waitResult=0,launchResult=0,dispatchResult=0;
int enqueues=0,waits=0,dispatches=0,commits=0;void* observedStream=nullptr;
std::vector<int> events;
void setLastError(const char*,int){}
int cudaGetLastError(){return launchResult;}
int cudaStreamSynchronize(void* stream){++waits;observedStream=stream;events.push_back(2);return waitResult;}
struct Owner{int observeCompletedWork(void* stream){return cudaStreamSynchronize(stream);}
 int dispatchPending(){++dispatches;events.push_back(3);return dispatchResult;}};
namespace ugkwp {int enqueueGasStatusCheck(int*,int,int*,int,void*,int*,int,int,void* stream){
 ++enqueues;observedStream=stream;events.push_back(1);return enqueueResult;}}
struct DeviceState{
 bool gasModelPoisoned=false;int fixedCellBlockThreads=128,nCells=257,nFaces=129;
 struct{int*cellStatus=nullptr;int*faceStatus=nullptr;}gasSpecies;
 struct{bool enabled=false;int count=4;}gasBoundaryLayer;
 struct{double*acceptedHistory=nullptr;}gasBoundaryLayerModel;
 void*diagnostics=nullptr;int*gasStatusFirstUnavailable=nullptr;void*gasCaptureStream=this;
 Owner*diagnosticOwner=nullptr;DeviceState*deviceState=this;double*V=nullptr;
};
void commitAcceptedWallHistory(DeviceState*){++commits;}
int applyGasGravitySource(DeviceState*,int grid,int block,double dt,bool refinement){
 assert(grid==3&&block==128&&dt==.25&&refinement);return 0;}
''' + policy + r'''
int main(){DeviceState s;Owner owner;s.diagnosticOwner=&owner;
 assert(SharedGasTrialPolicy::beginInterval(&s)==0&&SharedGasTrialPolicy::begin(&s)==0);
 assert(SharedGasTrialPolicy::applySources(&s,.25,true)==0);
 // Stage validation enqueues on the resident stream without host waiting.
 assert(SharedGasTrialPolicy::validate(&s)==0&&enqueues==1&&waits==0&&dispatches==0);
 assert(observedStream==s.gasCaptureStream);
 dispatchResult=1;
 assert(SharedGasTrialPolicy::complete(&s)!=0&&waits==1&&dispatches==1);
 assert((events==std::vector<int>{1,2,3}));
 dispatchResult=0;waitResult=1;
 assert(SharedGasTrialPolicy::complete(&s)!=0&&dispatches==1);
 waitResult=0;assert(SharedGasTrialPolicy::complete(&s)==0&&dispatches==2);
 enqueueResult=1;assert(SharedGasTrialPolicy::validate(&s)!=0);
 enqueueResult=0;s.diagnosticOwner=nullptr;
 assert(SharedGasTrialPolicy::complete(&s)==0&&dispatches==2);
 assert(SharedGasTrialPolicy::commitInterval(&s)==0&&commits==0);
 double history=0;s.gasBoundaryLayer.enabled=true;s.gasBoundaryLayerModel.acceptedHistory=&history;
 assert(SharedGasTrialPolicy::commitInterval(&s)==0&&commits==1);
 launchResult=1;assert(SharedGasTrialPolicy::commitInterval(&s)!=0);
 s.gasModelPoisoned=true;const int before=enqueues;
 assert(SharedGasTrialPolicy::beginInterval(&s)!=0&&SharedGasTrialPolicy::begin(&s)!=0);
 assert(SharedGasTrialPolicy::validate(&s)!=0&&enqueues==before);
 assert(SharedGasTrialPolicy::commit(&s)!=0&&SharedGasTrialPolicy::commitInterval(&s)!=0);
 assert(SharedGasTrialPolicy::stageVolumes(&s,false)==s.V);
}
'''
    source=tmp_path/"policy.cpp";source.write_text(code);binary=tmp_path/"policy"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror",str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_native_mixture_calls_common_requested_interval_and_commits_clock_afterward():
    backend=(SOURCE_ROOT/"applications/gasUGKP/private_backend/GpuResidentStrict.cu").read_text()
    front=(SOURCE_ROOT/"applications/gasUGKP/diluteUgkwpFoam.C").read_text()
    assert 'advanceGasRequestedInterval<SharedGasTrialPolicy>' in backend
    assert '#include "SharedGasTrialPolicy.cuh"' in backend
    mixture=front[front.index('if (sharedGasModel.active())'):]
    assert mixture.index('resident.advanceOneStep') < mixture.index('runTime++;')
