
"""Reachability contracts and existing zero-grid early failure, without CUDA."""
from pathlib import Path
import re,subprocess
import pytest
from test_low_level_occupancy_behavior import function
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app',['gasUGKP','FSH','CHT'])
def test_legacy_private_gather_has_no_production_owner(app):
    source=(ROOT/'applications'/app/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu').read_text()
    for old in ['launchGatherCellLocalParticles','CsrGatherOperation','gatherCsrSegmentedParticlesKernel','gatherThermalSegmentedParticlesKernel','indexCellLocalParticlesKernel']:
        assert old not in source
    assert 'GpuCellLocalGather.cuh' not in source
    if app!='gasUGKP':
        advance=source[source.index('extern "C" int ugkwpGpuResidentStrictAdvance'):]
        assert 'const bool skipParticlePath = !s->particlesMayBePresent;' in advance
        assert 'particleGrid > 0 && s->csrCellLocalPathEnabled != 0' in advance
        assert advance.index('prepareParticleDirectoryAndSchedule(s, block)')<advance.index('exactSurvivorDirectory')
        assert 'if (!exactSurvivorDirectory)' not in advance
        assert 'gatherCellLocalParticlePayloadKernel<false>' not in advance
        assert 'int preBaseDirectoryReady = 0;' in source
        upload=source[source.index('extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror'):]
        assert upload.index('s->preBaseDirectoryReady = 0;')<upload.index('if (nParticles == 0)')
def test_existing_full_directory_zero_grid_error_precedes_tasks(tmp_path):
    source=(ROOT/'common/GpuParticleDirectoryHost.cuh').read_text()
    body=function(source,'binParticlesByCell')
    # Only model the launch API, retaining the actual host flow/error guards.
    launches=[]
    def replace(m):
        launches.append(m[1]);return 'hostLaunch('+m[2].split(',')[0].strip()+');'
    body=re.sub(r'(\w+)(?:<[^>]*>)?\s*<<<([\s\S]*?)>>>\s*\([^;]*?\);',replace,body)
    assert len(launches)==10,launches
    code=r"""
#include <cassert>
#include <cstddef>
#include <string>
#include <algorithm>
struct DeviceState {
 int nCells=4,particleWorkGrid=0,particleBlockThreads=128,csrTasksReady=99,csrWarpAggregatedBinning=0;
 int *cellParticleCount=nullptr,*cellParticleOffset=nullptr;void* cellScanTempStorage=nullptr;size_t cellScanTempBytes=0;
 DeviceState* deviceState=this;
};
using cudaError_t=int;constexpr int cudaSuccess=0;
int pending=0,launches=0,scans=0,tasks=0;std::string error;
void hostLaunch(int grid){++launches;pending=grid>0?0:9;}
int cudaGetLastError(){int e=pending;pending=0;return e;}
void setLastError(const char* s,int){error=s;}
void selectParticleDirectory(DeviceState*,bool){}
int fullParticleDirectoryKind(){return 0;}
int prepareParticleDirectoryTasks(DeviceState*,int,int){++tasks;return 0;}
namespace cub {struct DeviceScan {template<class... Args>static int ExclusiveSum(Args...){++scans;return 0;}};}
@BODY@
int main(){
 for(int warp:{0,1})for(bool survivors:{false,true})for(int grid:{0,1,7}){
  DeviceState s;s.csrWarpAggregatedBinning=warp;s.particleWorkGrid=grid;
  launches=scans=tasks=pending=0;error.clear();
  int result=binParticlesByCell(&s,128,survivors);
  assert(s.csrTasksReady==0);
  if(grid==0){assert(result==1&&launches==2&&scans==0&&tasks==0);assert(error=="countParticlesByCellKernel launch");}
  else assert(result==0&&launches==4&&scans==1&&tasks==1);
 }
}
""".replace('@BODY@',body)
    cpp=tmp_path/'boundary.cpp';cpp.write_text(code);exe=tmp_path/'boundary'
    build=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
def test_survivor_fixtures_use_current_consumer():
    for app in ['FSH','CHT']:
        text=(ROOT/f'tests/fixtures/thermal_workers/gather_{app}.cu.in').read_text()
        assert 'gatherCellLocalParticlePayloadKernel<true>' in text
        assert 'gatherThermalSegmentedParticlesKernel' not in text
    assert not (ROOT/'gpu/thermal/CsrSegmentedGather.cuh').exists()
    # This header is retained intentionally for the frozen downstream fluid
    # library. Removing a test-clone include does not authorize breaking it.
    assert (ROOT/'common/GpuCellLocalGather.cuh').is_file()
