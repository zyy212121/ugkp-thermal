
"""Compare exact host configurations/query traces against fa71f477 without CUDA."""
from pathlib import Path
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
def function(text,name):
    m=re.search(r'(?:int|inline void)\s+'+name+r'\([^;{}]*\)\s*\{',text)
    assert m,name
    i=text.index('{',m.start())+1;depth=1
    while depth:depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[m.start():i]
@pytest.mark.parametrize('app,real,override',[('FSH','double',0),('CHT','double',0),('CHT','float',48),('CHT','float',0)])
def test_occupancy_configuration_and_final_sync(app,real,override,tmp_path):
    source=(ROOT/'applications'/app/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu').read_text()
    old=(ROOT/'tests/fixtures/low_level/reference'/f'{app}-occupancy.cpp.in').read_text().replace('configureLaunchOccupancy','oldConfiguration')
    helper=(ROOT/'common/GpuParticleLaunchConfiguration.cuh').read_text().replace('#pragma once','')
    hook=function(source,'configureAdditionalLaunchWorkGrids')
    new=(ROOT/'common/GpuThermalLaunchOccupancy.cuh').read_text().replace('#pragma once','')
    fields=sorted(set(re.findall(r's->(\w+)',helper+old+hook+new)))
    code=r"""
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using GpuReal=@REAL@;
#define GPU_OPERATOR_REAL GpuReal
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaDevAttrCooperativeLaunch=1;
constexpr int coldWallBlockThreads=32,coldWallSmBlocks=@OVERRIDE@;
constexpr bool postTransportFusePayload=@FUSE@;
struct DeviceState {
@FIELDS@
};
DeviceState synced{};
std::vector<std::string> events;
int callNumber=0,failAt=0,zeroAt=0,cooperative=1;
template<bool Fused> int accumulateCsrSegmentedMomentTasksPersistentKernel=11+Fused;
template<bool Heavy,bool Fused=false> int accumulateParticleMomentsSegmentedKernel=21+Heavy*2+Fused;
template<bool A,bool B> int countParticlesByCellKernel=31+A*2+B;
int accumulatePoissonPoolParticlesByCellKernel=40,completeMobilePackingProjectionCooperativeKernel=41;
int trackParticlesLocalFaceWalkKernel=42;
int api(const std::string& event){events.push_back(event);return ++callNumber==failAt?73:0;}
int occupancy(int* p,int kernel,int block,size_t bytes){
 int e=api("occupancy:"+std::to_string(kernel)+":"+std::to_string(block)+":"+std::to_string(bytes));
 *p=callNumber==zeroAt?0:std::max(1,768/block-(kernel%3));return e;
}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,int kernel,int block,size_t bytes){return occupancy(p,kernel,block,bytes);}
int thermalPoolLaunchOccupancy(int* p,int block,size_t bytes){return occupancy(p,50,block,bytes);}
int cudaGetDevice(int* p){*p=0;return api("getDevice");}
int cudaDeviceGetAttribute(int* p,int attr,int device){*p=cooperative;return api("attribute:"+std::to_string(attr)+":"+std::to_string(device));}
void setLastError(const char* p,int e){events.push_back(std::string("error:")+p+":"+std::to_string(e));}
void setLastErrorText(const char* p){events.push_back(std::string("errorText:")+p);}
int syncDeviceState(DeviceState* s,const char* p){synced=*s;return api(std::string("sync:")+p);}
@HELPER@
@OLD@
@HOOK@
@NEW@
int main(){
 for(int block:{32,64,128,256})for(int capacity:{0,1,513,1000000})for(int heavy:{0,1})for(int jamming:{0,1})
 for(int fail=0;fail<=10;++fail)for(int zero:{0,1,2,5,6})for(int coop:{0,1}){
  DeviceState a{},b{};a.reductionBlockThreads=block;a.particleBlockThreads=128;
  a.fixedCellBlockThreads=128;a.fixedFaceBlockThreads=64;a.multiprocessorCount=24;
  a.particleCapacity=capacity;a.csrHeavyReductionEnabled=heavy;a.jammingPressureEnabled=jamming;
  a.coldWallWorkGrid=-7;b=a;
  failAt=fail;zeroAt=zero;cooperative=coop;callNumber=0;events.clear();synced={};
  int oldResult=oldConfiguration(&a);auto oldEvents=events;auto oldSynced=synced;
  callNumber=0;events.clear();synced={};int newResult=configureLaunchOccupancy(&b);
  assert(oldResult==newResult && oldEvents==events);
  assert(std::memcmp(&a,&b,sizeof a)==0);
  assert(std::memcmp(&oldSynced,&synced,sizeof synced)==0);
  if(newResult==0){
   assert(synced.particleWorkGrid==b.particleWorkGrid);
   assert(synced.trackingWorkGrid==b.trackingWorkGrid);
   assert(synced.coldWallWorkGrid==b.coldWallWorkGrid);
   if(@CHT@)assert(synced.coldWallWorkGrid==(coldWallSmBlocks>0?24*coldWallSmBlocks:b.particleWorkGrid));
   assert(events.back()=="sync:sync occupancy-derived launch geometry");
  }
 }
 puts("PASS host occupancy: exact fields, query order/resources/errors and final sync snapshot");
}
"""
    values={'REAL':real,'OVERRIDE':str(override),'FUSE':str(app=='FSH').lower(),'CHT':str(int(app=='CHT')),
        'FIELDS':'\n'.join(' int '+f+'=0;' for f in sorted(set(fields)|{'coldWallWorkGrid'})),
        'HELPER':helper,'OLD':old,'HOOK':hook,'NEW':new}
    for key,value in values.items():code=code.replace('@'+key+'@',value)
    cpp=tmp_path/'occupancy.cpp';cpp.write_text(code);exe=tmp_path/'occupancy'
    build=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
