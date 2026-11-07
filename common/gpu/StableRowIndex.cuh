#pragma once
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cstddef>
namespace ugkp { namespace gpu {
// CSR index for an arbitrary item->row device mapping. Negative/out-of-range
// mappings are inactive. Stable sorting retains the original order within each
// row, so downstream FP64 sums use the same ordering as host row construction.
struct StableRowIndexView {
    int* keys=nullptr;int* sortedKeys=nullptr;int* inputIndices=nullptr;
    int* indices=nullptr;int* counts=nullptr;int* offsets=nullptr;
};
template<class Row>
__global__ void prepareStableRowIndex(int items,int rows,Row row,StableRowIndexView view){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=items)return;
    const int key=row(i);const bool active=key>=0&&key<rows;
    view.keys[i]=active?key:rows;view.inputIndices[i]=i;
    if(active)atomicAdd(view.counts+key,1);
}
inline cudaError_t stableRowIndexWorkspace(int items,int rows,const StableRowIndexView& view,
    std::size_t& bytes,cudaStream_t stream){
    bytes=0;if(items<0||rows<0)return cudaErrorInvalidValue;
    std::size_t scan=0,sort=0;
    auto error=cub::DeviceScan::ExclusiveSum(nullptr,scan,view.counts,view.offsets,rows+1,stream);
    if(error==cudaSuccess&&items)
        error=cub::DeviceRadixSort::SortPairs(nullptr,sort,view.keys,view.sortedKeys,
            view.inputIndices,view.indices,items,0,int(sizeof(int)*8),stream);
    bytes=scan>sort?scan:sort;return error;
}
template<class Row>
cudaError_t buildStableRowIndex(int items,int rows,Row row,const StableRowIndexView& view,
    void* scratch,std::size_t bytes,cudaStream_t stream){
    if(items<0||rows<0)return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(view.counts,0,std::size_t(rows+1)*sizeof(int),stream);
    if(error==cudaSuccess&&items){
        prepareStableRowIndex<<<(items+255)/256,256,0,stream>>>(items,rows,row,view);
        error=cudaGetLastError();
    }
    if(error==cudaSuccess){auto available=bytes;
        error=cub::DeviceScan::ExclusiveSum(scratch,available,view.counts,view.offsets,rows+1,stream);}
    if(error==cudaSuccess&&items){auto available=bytes;
        error=cub::DeviceRadixSort::SortPairs(scratch,available,view.keys,view.sortedKeys,
            view.inputIndices,view.indices,items,0,int(sizeof(int)*8),stream);}
    return error;
}
}}
