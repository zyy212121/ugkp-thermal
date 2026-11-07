#include "../../common/gpu/StableRowIndex.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Row {const int* keys;__device__ int operator()(int i)const{return keys[i];}};
static void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::abort();}}
int main(){
    using namespace ugkp::gpu;
    cudaStream_t stream;check(cudaStreamCreate(&stream));int cases=0;
    for(int rows:{0,1,31,1025})for(int items:{0,1,17,4097}){
        std::vector<int> keys(items),expected,offsets(rows+1);
        for(int i=0;i<items;++i)keys[i]=rows?((i*19+7)%(rows+3))-1:-1;
        for(int r=0;r<rows;++r){
            offsets[r]=int(expected.size());for(int i=0;i<items;++i)if(keys[i]==r)expected.push_back(i);
        }
        offsets[rows]=int(expected.size());
        int* input;check(cudaMalloc(&input,std::max(items,1)*sizeof(int)));
        check(cudaMemcpyAsync(input,keys.data(),items*sizeof(int),cudaMemcpyHostToDevice,stream));
        StableRowIndexView v;
        for(auto p:{&v.keys,&v.sortedKeys,&v.inputIndices,&v.indices})check(cudaMalloc(p,std::max(items,1)*sizeof(int)));
        check(cudaMalloc(&v.counts,(rows+1)*sizeof(int)));check(cudaMalloc(&v.offsets,(rows+1)*sizeof(int)));
        std::size_t bytes=0;check(stableRowIndexWorkspace(items,rows,v,bytes,stream));void* scratch;check(cudaMalloc(&scratch,bytes));
        // The same buffers must support repeated overwrite, not accumulated counts.
        for(int repeat=0;repeat<3;++repeat){
            check(buildStableRowIndex(items,rows,Row{input},v,scratch,bytes,stream));
            std::vector<int> actualOffsets(rows+1),actual(expected.size());
            check(cudaMemcpyAsync(actualOffsets.data(),v.offsets,(rows+1)*sizeof(int),cudaMemcpyDeviceToHost,stream));
            if(!actual.empty())check(cudaMemcpyAsync(actual.data(),v.indices,actual.size()*sizeof(int),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));assert(actualOffsets==offsets);assert(actual==expected);
        }
        for(auto p:{input,v.keys,v.sortedKeys,v.inputIndices,v.indices,v.counts,v.offsets})check(cudaFree(p));
        check(cudaFree(scratch));++cases;
    }
    check(cudaStreamDestroy(stream));std::printf("PASS %d stable CSR row layouts, three reuses each\n",cases);
}
