"""Host ownership semantics with a fake CUDA allocator, not GPU execution."""
import pathlib
import subprocess
import tempfile
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
class GpuDiagnosticOwner(unittest.TestCase):
    def test_sticky_instances_and_bounded_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=pathlib.Path(tmp)
            (folder/'cuda_runtime.h').write_text('''#pragma once
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
static unsigned hostReads=0;
inline int cudaMalloc(void** p,size_t n){*p=std::malloc(n);return *p?0:1;}
inline int cudaFree(void* p){std::free(p);return 0;}
inline int cudaMemcpy(void* d,const void* s,size_t n,int direction){if(direction==2)++hostReads;std::memcpy(d,s,n);return 0;}
''')
            source=folder/'test.cpp';source.write_text('''#include "common/GpuDiagnosticsOwner.cuh"
#include <cassert>
using namespace ugkp::diagnostics;
int main(){
 GpuOwner a,b;assert(a.initialise()==0&&b.initialise()==0);assert(a.device()!=b.device());
 auto warning=makeRecord(Code::AccuracyResidual,2101);recordWarning(a.device(),warning);
 Snapshot snapshot; a.finish(snapshot);assert(snapshot.warnings.empty()&&hostReads==0);
 a.observeCompletedWork();a.finish(snapshot);assert(snapshot.warnings.size()==1&&snapshot.warnings[0].count==1&&hostReads==1);
 a.observeCompletedWork();a.finish(snapshot);assert(snapshot.warnings.empty());
 b.observeCompletedWork();b.finish(snapshot);assert(snapshot.warnings.empty());
 auto fatal=withOperation(makeRecord(Code::InvalidIndex,2301),8,7,0.5);a.recordHostFatal(fatal);
 a.observeCompletedWork();a.finish(snapshot);assert(snapshot.hasFatal&&snapshot.fatal.step==7&&snapshot.fatal.simulationTime==0.5);assert(findSource(snapshot,2301)!=nullptr);
 a.finish(snapshot);assert(snapshot.hasFatal);b.finish(snapshot);assert(!snapshot.hasFatal);
 recordFatal(b.device(),makeRecord(Code::NonFinite,2102));b.observeCompletedWork();b.finish(snapshot);assert(snapshot.hasFatal);
}
''')
            binary=folder/'test'
            subprocess.run(['g++','-std=c++17','-I'+tmp,'-I'+str(ROOT),str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
if __name__=='__main__':unittest.main()
