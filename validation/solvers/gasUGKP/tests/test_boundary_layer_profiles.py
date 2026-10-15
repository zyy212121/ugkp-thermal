"""Native profile opt-in/download/reconfiguration using the host CUDA shim."""
import subprocess
from test_boundary_layer_lifecycle import _publication_probe
from test_shared_gas_adapter import ROOT


def test_profile_storage_is_opt_in_and_release_resets_capture_dispatch(tmp_path):
    pre=_publication_probe().split('int main(int argc',1)[0]
    source=pre+r'''
#include "applications/gasUGKP/private_backend/BoundaryLayerProfileAdapter.cuh"
int main(){
 DeviceState state,device;state.deviceState=&device;
 assert(!state.gasBoundaryLayerModel.profile && allocations==0);
 assert(state.gasBoundaryLayerLaunch==&launchBoundaryLayer<false>);
 state.gasBoundaryLayer.enabled=true;state.gasBoundaryLayer.count=2;
 state.gasBoundaryLayerModel.config.nodes=24;
 auto& model=state.gasBoundaryLayerModel;
 assert(ugkwpGpuResidentStrictEnableBoundaryLayerProfilesV1(&state)==0);
 assert(allocations==1&&model.profile==device.gasBoundaryLayerModel.profile);
 assert(state.gasBoundaryLayerLaunch==&launchBoundaryLayer<true>);
 assert((state.gasBoundaryLayerSnapshot==&sharedGasTrialDetail::snapshotWallRecords<true,DeviceState>));
 const auto* allocated=model.profile;
 assert(ugkwpGpuResidentStrictEnableBoundaryLayerProfilesV1(&state)==0&&allocations==1&&model.profile==allocated);
 for(int i=0;i<2*24*9;++i)model.profile[i]=.5+i;
 ugkwp::GasBoundaryLayerExchange<double> exchange[2];state.gasBoundaryLayer.exchange=exchange;
 double rows[2*24*9],stageTime=-1;
 assert(ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1(&state,2,24,2,rows,&stageTime)!=0);
 exchange[0].ready=exchange[1].ready=true;state.gasBoundaryLayerStageTime=1.125;
 assert(ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1(&state,2,24,2,rows,&stageTime)==0);
 assert(stageTime==1.125);for(int i=0;i<2*24*9;++i)assert(rows[i]==.5+i);
 state.gasBoundaryLayer.exchange=nullptr; // stack-owned shim readiness array
 assert(releaseBoundaryLayerStorage(state)==0&&allocations==0);
 assert(!model.profile&&state.gasBoundaryLayerLaunch==&launchBoundaryLayer<false>);
 assert((state.gasBoundaryLayerSnapshot==&sharedGasTrialDetail::snapshotWallRecords<false,DeviceState>));
 // Reconfigure without requesting output: no profile allocation or capture.
 state.gasBoundaryLayer.enabled=true;state.gasBoundaryLayer.count=1;
 assert(!model.profile&&state.gasBoundaryLayerLaunch==&launchBoundaryLayer<false>);
 assert(releaseBoundaryLayerStorage(state)==0&&allocations==0);
}
'''
    cpp=tmp_path/'profiles.cpp';cpp.write_text(source);exe=tmp_path/'profiles'
    build=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
