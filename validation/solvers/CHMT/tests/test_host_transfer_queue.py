"""Delayed host CUDA shim: transfer ordering and lifetime, no GPU execution."""
from pathlib import Path
import subprocess
from validation.tools.paths import SOURCE_ROOT

STUB = r'''
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
using cudaError_t=int;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaStreamNonBlocking=1;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3;
struct Copy{void* to;const void* from;size_t bytes;void* stream;};
inline std::vector<Copy> pending;
inline int pinnedAllocations=0,waits=0,failCopy=0;
inline void* expectedStream=reinterpret_cast<void*>(1);
inline int cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:2;}
inline int cudaFree(void*p){std::free(p);return 0;}
inline int cudaMallocHost(void**p,size_t n){++pinnedAllocations;return cudaMalloc(p,n);}
inline int cudaFreeHost(void*p){return cudaFree(p);}
inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
inline int cudaMemcpyAsync(void*d,const void*s,size_t n,int,void* stream){
 assert(stream==expectedStream);if(failCopy){failCopy=0;return 3;}pending.push_back({d,s,n,stream});return 0;
}
inline int cudaMemsetAsync(void*p,int v,size_t n,void*){std::memset(p,v,n);return 0;}
inline int cudaStreamSynchronize(void* stream){
 ++waits;for(auto it=pending.begin();it!=pending.end();){
  if(it->stream==stream){std::memcpy(it->to,it->from,it->bytes);it=pending.erase(it);}else ++it;
 }return 0;
}
inline int cudaStreamCreateWithFlags(void**p,int){*p=expectedStream;return 0;}
inline int cudaStreamDestroy(void*){return 0;}
inline const char* cudaGetErrorString(int){return "shim error";}
inline int cudaPeekAtLastError(){return 0;}inline int cudaGetLastError(){return 0;}
'''

SOURCE = r'''
#include "gpu/HostTransferQueue.H"
#include "GpuDiagnosticsOwner.cuh"
#include <cassert>
int main(){
 chmt::CudaFault fault;chmt::HostTransferQueue queue(fault);queue.bindStream(expectedStream);
 int device=0,output=-1;
 {int source=17;assert(queue.upload(&device,&source,1));source=91;}
 assert(device==0);assert(queue.download(&output,&device,1));assert(output==-1);
 // An unrelated owner's work must remain pending at this completion point.
 int other=0,otherSource=32;pending.push_back({&other,&otherSource,sizeof(int),reinterpret_cast<void*>(2)});
 assert(queue.finish());assert(device==17&&output==17&&other==0&&waits==1);
 const int allocations=pinnedAllocations;
 int source=28;assert(queue.upload(&device,&source,1));assert(queue.download(&output,&device,1));
 assert(queue.finish());assert(output==28&&pinnedAllocations==allocations);
 // A failed submission drains earlier copies without publishing partial reads.
 output=-7;assert(queue.download(&output,&device,1));failCopy=1;
 assert(!queue.download(&output,&device,1));assert(!queue.finish());assert(output==-7);
 assert(queue.download(&output,&device,1));assert(queue.finish());assert(output==28);
 // Destructor completes DMA but never writes into an unwound destination.
 int untouched=-9;
 {chmt::HostTransferQueue abandoned(fault);abandoned.bindStream(expectedStream);
  assert(abandoned.download(&untouched,&device,1));}
 assert(untouched==-9&&other==0);
 // Explicit stream initialization and batched diagnostics snapshot consumption.
 ugkp::diagnostics::GpuOwner diagnostics;assert(diagnostics.initialise(expectedStream)==cudaSuccess);
 ugkp::diagnostics::DeviceContext snapshot{};
 assert(queue.download(&snapshot,diagnostics.device(),1));assert(queue.finish());
 assert(snapshot.instanceId==diagnostics.instanceId());diagnostics.consumeCompletedSnapshot(snapshot);
 assert(!diagnostics.terminal());
 ugkp::diagnostics::recordFatal(&snapshot,ugkp::diagnostics::makeRecord(
  ugkp::diagnostics::Code::MissingOperator,2600,26,0,ugkp::diagnostics::EntityKind::Cell,3,-1));
 diagnostics.consumeCompletedSnapshot(snapshot);assert(diagnostics.terminal());
 assert(other==0&&pending.size()==1);cudaStreamSynchronize(reinterpret_cast<void*>(2));assert(other==32);
}
'''

def test_delayed_owner_stream_transfers_and_diagnostics(tmp_path):
    (tmp_path / 'cuda_runtime.h').write_text(STUB)
    source = tmp_path / 'probe.cpp'
    source.write_text(SOURCE)
    binary = tmp_path / 'probe'
    subprocess.run([
        'g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
        '-I' + str(tmp_path), '-I' + str(SOURCE_ROOT / 'applications/CHMT'),
        '-I' + str(SOURCE_ROOT / 'common'), str(source), '-o', str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_staging_allocation_exception_discards_earlier_readbacks(tmp_path):
    (tmp_path / 'cuda_runtime.h').write_text(STUB)
    source = tmp_path / 'exceptions.cpp'
    source.write_text(r'''
#include <cstdlib>
#include <new>
static bool failNextHostAllocation=false;
void* operator new(std::size_t n){
 if(failNextHostAllocation){failNextHostAllocation=false;throw std::bad_alloc();}
 if(void* p=std::malloc(n))return p;throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#include "gpu/HostTransferQueue.H"
int main(){
 chmt::CudaFault fault;chmt::HostTransferQueue queue(fault);queue.bindStream(expectedStream);
 int device=17,abandoned=-7,next=-9;bool threw=false;
 assert(queue.download(&abandoned,&device,1));
 failNextHostAllocation=true;
 try{queue.download(&next,&device,1);}catch(const std::bad_alloc&){threw=true;}
 assert(threw&&pending.empty()&&abandoned==-7&&next==-9);
 assert(queue.download(&next,&device,1)&&queue.finish());
 assert(next==17&&abandoned==-7);
}
''')
    binary = tmp_path / 'exceptions'
    subprocess.run(['g++', '-std=c++17', '-I' + str(tmp_path),
                    '-I' + str(SOURCE_ROOT / 'applications/CHMT'),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_destination_resize_exception_discards_earlier_readbacks(tmp_path):
    import runpy
    fixture = runpy.run_path(str(Path(__file__).with_name('test_shared_device_storage.py')))['SOURCE']
    source = fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')] + r'''
 chmt::SharedGasDeviceStorage storage;storage.bindStream(expectedStream);
 assert(storage.configure(model,gas,{},h,error));
 std::vector<double> abandoned(1,-7),tooLarge;bool threw=false;
 assert(storage.enqueueRead(storage.hostView().rho,1,abandoned));
 try{storage.enqueueRead(storage.hostView().rho,std::numeric_limits<std::size_t>::max(),tooLarge);}
 catch(const std::exception&){threw=true;}
 assert(threw&&pending.empty()&&abandoned[0]==-7);
 assert(storage.observeDiagnostics()&&abandoned[0]==-7);
 std::vector<double> next;assert(storage.read(storage.hostView().rho,1,next)&&next[0]==1);
}
'''
    (tmp_path / 'cuda_runtime.h').write_text(STUB)
    path = tmp_path / 'resize.cpp'
    path.write_text(source)
    binary = tmp_path / 'resize'
    subprocess.run(['g++', '-std=c++17', '-I' + str(tmp_path),
                    '-I' + str(SOURCE_ROOT / 'applications/CHMT'),
                    '-I' + str(SOURCE_ROOT / 'common'), str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
