"""CPU intrinsic adapter checks the actual CUDA team API; no CUDA target executes."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

CPP=r"""
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
struct Barrier{std::mutex mutex;std::condition_variable cv;int count=0,generation=0;
 void wait(){std::unique_lock<std::mutex> lock(mutex);int g=generation;
  if(++count==32){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return g!=generation;});}} barrier;
struct Index{int x;};thread_local Index threadIdx{};
thread_local bool ordered=false;thread_local int explicitBarriers=0;
int flags[32]{},flag=0,integer=0,workspace[32]{};std::uint64_t scalar=0;
void __syncwarp(unsigned mask){assert(mask==0xffffffffu);++explicitBarriers;barrier.wait();ordered=true;}
void requireOrder(unsigned mask){assert(mask==0xffffffffu&&ordered);ordered=false;}
int __all_sync(unsigned mask,int value){requireOrder(mask);flags[threadIdx.x]=value;barrier.wait();
 if(threadIdx.x==0){flag=1;for(int i=0;i<32;++i)flag=flag&&flags[i];}barrier.wait();int out=flag;barrier.wait();return out;}
int __shfl_sync(unsigned mask,int value,int leader,int width=32){assert(width==32);requireOrder(mask);assert(leader==0);
 if(threadIdx.x==0)integer=value;barrier.wait();int out=integer;barrier.wait();return out;}
double __shfl_sync(unsigned mask,double value,int leader,int width=32){assert(width==32);requireOrder(mask);assert(leader==0);
 if(threadIdx.x==0)std::memcpy(&scalar,&value,sizeof(value));barrier.wait();double out;std::memcpy(&out,&scalar,sizeof(out));barrier.wait();return out;}
double shuffleValues[32]{};
double __shfl_down_sync(unsigned mask,double value,int delta,int width=32){
 assert(mask==0xffffffffu&&width==32);shuffleValues[threadIdx.x]=value;barrier.wait();
 const double result=shuffleValues[threadIdx.x+delta<32?threadIdx.x+delta:threadIdx.x];
 barrier.wait();return result;
}
#define __device__
#include "common/gasWall/WallCudaTeam.cuh"
int main(){using ugkwp::gaswall::detail::CudaWallTeam;static_assert(CudaWallTeam::width==32,"one full warp");
 std::vector<std::thread> threads;
 for(int lane=0;lane<32;++lane)threads.emplace_back([lane]{threadIdx.x=lane;CudaWallTeam team;
  assert(team.rank()==lane&&team.size()==32&&team.leader()==(lane==0));
  for(int phase=0;phase<40;++phase){workspace[lane]=phase*32+lane;
   assert(team.all(phase%2==0||lane!=7)==(phase%2==0));
   for(int j=0;j<32;++j)assert(workspace[j]==phase*32+j);
   assert(team.broadcast(lane==0?(phase%2==0):(phase%2!=0))==(phase%2==0));
   // Signed zero and quiet NaN payloads must survive the register broadcast.
   std::uint64_t expected=phase%2?0x8000000000000000ULL:0x7ff8000000000123ULL;
   std::uint64_t supplied=lane==0?expected:0x3ff0000000000000ULL;double value;
   std::memcpy(&value,&supplied,sizeof(value));value=team.broadcast(value);
   std::uint64_t actual;std::memcpy(&actual,&value,sizeof(actual));assert(actual==expected);
   team.sync();ordered=false;
  }assert(explicitBarriers==160);
 });for(auto& t:threads)t.join();}
"""

def test_actual_warp_team_contract_with_cpu_intrinsic_adapter(tmp_path):
    source=tmp_path/'warp.cpp';source.write_text(CPP);exe=tmp_path/'warp'
    build=subprocess.run(['g++','-std=c++17','-Werror','-O2','-pthread','-I'+str(SOURCE_ROOT),'-I'+str(SOURCE_ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.mark.parametrize('primitive',[
    'sync();return __all_sync(mask,int(value))!=0;',
    'sync();return __shfl_sync(mask,int(value),0)!=0;',
    'sync();return __shfl_sync(mask,value,0);',
])
def test_adapter_rejects_missing_workspace_ordering(tmp_path,primitive):
    # Fault injection is confined to a private header copy; production is read-only.
    root=tmp_path/'mutant'
    for name in ['common/gasWall/WallCudaTeam.cuh','common/gasWall/WallExecution.H','common/gasTransport/GasModel.H','common/Diagnostics.H']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((SOURCE_ROOT/name).read_bytes())
    header=root/'common/gasWall/WallCudaTeam.cuh';text=header.read_text()
    assert text.count(primitive)==1
    header.write_text(text.replace(primitive,primitive.replace('sync();','')))
    source=tmp_path/'mutant.cpp';source.write_text(CPP);exe=tmp_path/'mutant-probe'
    build=subprocess.run(['g++','-std=c++17','-Werror','-O2','-pthread','-I'+str(root),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert 'ordered' in result.stderr
