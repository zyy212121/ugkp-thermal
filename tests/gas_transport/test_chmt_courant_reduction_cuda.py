"""CHMT Courant scratch tests: real CUDA compilation/execution, never CPU shims."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "applications/CHMT"


def test_courant_reductions_keep_algorithms_and_owned_scratch():
    source = (APP / "gpu/Backend.cu").read_text()
    assert "CoupledCourantStorage courantReductionStorage;" in source
    assert "unsigned char initializedTail[7]{};" in source
    assert source.count("thrust::cuda::par(b.courantReductionStorage).on(stream)") == 2
    assert "thrust::transform_reduce(" in source and "thrust::reduce(" in source
    assert "a!=a?a:b!=b?b:fmax(a,b)" in source
    assert "a.unavailable||b.unavailable" in source
    assert "CoupledCourantValue{-std::numeric_limits<double>::infinity(),false}" in source


@pytest.fixture(scope="module")
def native_courant_probe(tmp_path_factory):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        pytest.skip("CUDA compiler unavailable; no native compile/runtime claim")
    folder = tmp_path_factory.mktemp("chmt-courant-native")
    backend = (APP / "gpu/Backend.cu").read_text()
    first = backend.index("struct CoupledCourantMaximum {")
    last = backend.index("static bool currentGasCourant(", first)
    source = folder / "courant_reduction.cu"
    source.write_text(PREAMBLE + backend[first:last] + PROBE)
    output = folder / "courant_reduction"
    command = [nvcc, "-std=c++17", "-O2",
               "-arch=" + os.environ.get("CHMT_CUDA_ARCH", "sm_89"),
               "-I" + str(APP), "-I" + str(ROOT / "common"),
               str(source), "-o", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_courant_native_cuda_compile(native_courant_probe):
    assert native_courant_probe.is_file()


def test_courant_native_gpu_reuse_and_equivalence(native_courant_probe):
    if os.environ.get("CHMT_COURANT_GPU_TESTS") != "1":
        pytest.skip("Set CHMT_COURANT_GPU_TESTS=1 on a real GPU; no CPU fallback")
    result = subprocess.run([str(native_courant_probe)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA Courant reuse/equivalence PASS" in result.stdout


PREAMBLE = r'''
#include "gpu/CourantReductionStorage.cuh"
#include "gasTransport/GasCapabilities.H"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
using chmt::CoupledCourantStorage;
'''

PROBE = r'''
static_assert(sizeof(CoupledCourantValue)==16,"Preserve reduction value size");
static_assert(offsetof(CoupledCourantValue,unavailable)==8,"Preserve bool offset");
static_assert(offsetof(CoupledCourantValue,initializedTail)==9,"Cover all tail bytes");
static_assert(std::is_same<decltype(CoupledCourantValue::value),double>::value,"FP64 value");
static_assert(std::is_same<decltype(CoupledCourantValue::unavailable),bool>::value,"Bool status");
static_assert(!std::is_copy_constructible<CoupledCourantStorage>::value,"Unique owner");
static_assert(!std::is_move_constructible<CoupledCourantStorage>::value,"Stable fault owner");
void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}}
void sameBits(double a,double b){assert(std::memcmp(&a,&b,sizeof(double))==0);}
void runCase(int count,int variant,cudaStream_t stream){
    std::vector<double> input(count);
    std::vector<int> status(count);
    for(int i=0;i<count;++i){input[i]=double(i%53)-17.;status[i]=i%17==3?int(ugkwp::GasTransportCode::InvalidStorage):0;}
    if(variant==1){for(int i=0;i<count;++i)input[i]=i%2?-0.0:0.0;}
    if(variant==2 && count){
        const std::uint64_t first=0x7ff8000000000042ULL,second=0x7ff8000000000137ULL;
        std::memcpy(&input.front(),&first,sizeof(double));
        if(count>1)std::memcpy(&input.back(),&second,sizeof(double));
    }
    double* field=nullptr;int* deviceStatus=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&field),std::max(count,1)*sizeof(double)));
    check(cudaMalloc(reinterpret_cast<void**>(&deviceStatus),std::max(count,1)*sizeof(int)));
    if(count){check(cudaMemcpyAsync(field,input.data(),count*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(deviceStatus,status.data(),count*sizeof(int),cudaMemcpyHostToDevice,stream));}
    CoupledCourantStorage storage;
    const auto first=thrust::make_counting_iterator(0);
    const CoupledCourantValue initial{-std::numeric_limits<double>::infinity(),false};
    const auto expected=thrust::transform_reduce(thrust::cuda::par.on(stream),first,first+count,
        CoupledCourantRead{field,deviceStatus},initial,CoupledCourantReduce{});
    const auto scalar=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(field),
        thrust::device_pointer_cast(field)+count,initial.value,CoupledCourantMaximum{});
    for(int repeat=0;repeat<100;++repeat){
        const auto value=thrust::transform_reduce(thrust::cuda::par(storage).on(stream),first,first+count,
            CoupledCourantRead{field,deviceStatus},initial,CoupledCourantReduce{});
        sameBits(value.value,expected.value);assert(value.unavailable==expected.unavailable);
        for(auto byte:value.initializedTail)assert(byte==0);
        const auto maximum=thrust::reduce(thrust::cuda::par(storage).on(stream),thrust::device_pointer_cast(field),
            thrust::device_pointer_cast(field)+count,initial.value,CoupledCourantMaximum{});
        sameBits(maximum,scalar);
    }
    check(cudaFree(deviceStatus));check(cudaFree(field));
}
int main(){
    int devices=0;check(cudaGetDeviceCount(&devices));assert(devices>0);
    {
        CoupledCourantStorage storage;
        char* first=storage.allocate(17);storage.deallocate(first,17);
        char* smaller=storage.allocate(9);assert(first==smaller);
        char* simultaneous=storage.allocate(17);assert(smaller!=simultaneous);
        storage.deallocate(smaller,9);storage.deallocate(simultaneous,17);
        char* grown=storage.allocate(4096);check(cudaMemset(grown,0x5a,4096));storage.deallocate(grown,4096);
        char* retained=storage.allocate(4096);assert(retained==grown);storage.deallocate(retained,4096);
        bool failed=false;try{storage.allocate(std::numeric_limits<std::ptrdiff_t>::max());}
        catch(const std::exception&){failed=true;}assert(failed);
        char* retry=storage.allocate(17);check(cudaMemset(retry,0,17));storage.deallocate(retry,17);
    }
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(auto selected:{cudaStream_t{},stream})for(int count:{0,1,31,32,33,513,32768})
        for(int variant:{0,1,2})runCase(count,variant,selected);
    check(cudaStreamDestroy(stream));
    std::puts("CUDA Courant reuse/equivalence PASS");
}
'''
