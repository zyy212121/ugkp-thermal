"""Execute the production launch configuration with deterministic occupancy inputs."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def test_cold_wall_default_inherits_particle_grid(tmp_path):
    source = (ROOT / 'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    start = source.index('int configureLaunchOccupancy(DeviceState* s)')
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    configuration = source[start:end]
    helper = (ROOT / 'common/GpuParticleLaunchConfiguration.cuh').read_text()
    function = helper.replace('#pragma once', '') + '\n' + configuration
    fields = sorted(set(re.findall(r's->(\w+)', function)))
    fields.remove('deviceState') if 'deviceState' in fields else None
    for override in [0, 48]:
        body = r"""
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstddef>
using GpuReal=double;
using cudaError_t=int;
const int cudaSuccess=0,cudaDevAttrCooperativeLaunch=1;
const int coldWallBlockThreads=32;
struct DeviceState {
""" + ''.join(' int '+name+'=0;\n' for name in fields) + r"""
};
template<bool GatherSurvivors> int accumulateCsrSegmentedMomentTasksPersistentKernel=0;
int trackParticlesLocalFaceWalkKernel=0;
template<bool SkipDead, bool Survivors> int countParticlesByCellKernel=0;
int accumulatePoissonPoolParticlesByCellKernel=0;
template<bool Heavy, bool Gather=false> int accumulateParticleMomentsSegmentedKernel=0;
int completeMobilePackingProjectionCooperativeKernel=0;
int relaxColdWall1DParticlesToResidentGasKernel=0;
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,int,int block,size_t){*p=768/block;return 0;}
int thermalPoolLaunchOccupancy(int* p,int block,size_t){*p=768/block;return 0;}
int cudaGetDevice(int* p){*p=0;return 0;}
int cudaDeviceGetAttribute(int* p,int,int){*p=1;return 0;}
void setLastError(const char*,int){}
void setLastErrorText(const char*){}
int syncDeviceState(DeviceState*,const char*){return 0;}
""" + 'constexpr int coldWallSmBlocks='+str(override)+';\n' + re.search(r'constexpr bool postTransportFusePayload = [^;]+;', source).group(0) + '\n' + function + r"""
int main(){
 for(int block:{32,64,128,256})for(int capacity:{1,513,1000000})for(int heavy:{0,1}){
  DeviceState s;s.reductionBlockThreads=block;s.particleBlockThreads=block;
  s.particleCapacity=capacity;s.multiprocessorCount=24;s.csrHeavyReductionEnabled=heavy;
  assert(configureLaunchOccupancy(&s)==0);
  int particleGrid=std::min((capacity+block-1)/block,24*(768/block));
  assert(s.particleWorkGrid==particleGrid);
  int wallGrid=coldWallSmBlocks>0?24*coldWallSmBlocks:particleGrid;
  assert(s.coldWallWorkGrid==wallGrid);
 }
}
"""
        cpp=tmp_path/f'configuration_{override}.cpp';cpp.write_text(body)
        exe=cpp.with_suffix('')
        build=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
        assert build.returncode==0,build.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
