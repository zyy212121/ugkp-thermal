#include "GpuPrecisionTypes.H"
#include "SparseConjugateGradient.cu"
#include <cstdio>
using namespace ugkp::gpu;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
__global__ void gridProbeA(int* output){if(!threadIdx.x)atomicAdd(output,1);}
__global__ void gridProbeB(int* output){if(!threadIdx.x)atomicAdd(output,1);}
__global__ void gridRepeat(int* remaining,cudaGraphConditionalHandle handle){if(!threadIdx.x)cudaGraphSetConditional(handle,--*remaining>0);}
static int selectedBlocks(KernelGridBuckets buckets,int count){
 int i=0;while(i+1<buckets.count&&buckets.blocks[i]<count)++i;return count?buckets.blocks[i]:0;
}
static int kernelNodes(cudaGraph_t graph,void* function){
 size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);ck(cudaGraphGetNodes(graph,nodes.data(),&count));int found=0;
 for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeKernel){cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));found+=p.func==function;}}
 return found;
}
int main(){
 try{
  int *counts,*output,*remaining,*mode;ck(cudaMallocManaged(&counts,2*sizeof(int)));ck(cudaMallocManaged(&output,3*sizeof(int)));ck(cudaMallocManaged(&remaining,sizeof(int)));ck(cudaMallocManaged(&mode,sizeof(int)));*mode=2;
  cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  cudaGraph_t graph,body=nullptr;cudaGraphExec_t executable;ck(cudaGraphCreate(&graph,0));
  VCycle<GpuReal> hierarchy;
  hierarchy.withGridPublication(graph,{},[&](const std::vector<cudaGraphNode_t>& before){
   cudaGraphConditionalHandle loop;ck(cudaGraphConditionalHandleCreate(&loop,graph,1,cudaGraphCondAssignDefault));
   auto node=executionGraphDetail::addWhile(graph,loop,before,body);
   sparseGraphDetail::appendCaptured(body,{},stream,[&]{
    hierarchy.dispatch(gridProbeA,128,0,1024,stream,counts,1,0,[&](int blocks,cudaStream_t s){gridProbeA<<<blocks,128,0,s>>>(output);});
    hierarchy.dispatch(gridProbeB,128,0,1024,stream,counts+1,1,0,[&](int blocks,cudaStream_t s){gridProbeB<<<blocks,128,0,s>>>(output+1);});
    {
     VCycle<GpuReal>::GridSelectionGuard directLeaf(hierarchy,nullptr,nullptr,mode);
     hierarchy.dispatch(gridProbeA,128,0,1024,stream,nullptr,1,0,[&](int blocks,cudaStream_t s){gridProbeA<<<blocks,128,0,s>>>(output+2);});
    }
    gridRepeat<<<1,1,0,stream>>>(remaining,loop);
   });return node;
  });
  require(kernelNodes(body,reinterpret_cast<void*>(selectKernelGrid))==0,"selection kernel remains inside repeated body");
  require(kernelNodes(graph,reinterpret_cast<void*>(publishKernelGrids))==1,"missing single pre-loop publication");
  ck(cudaGraphInstantiateWithFlags(&executable,graph,0));
  KernelGridBuckets a,b;ck(kernelGridBuckets(gridProbeA,128,0,1024,1,a));ck(kernelGridBuckets(gridProbeB,128,0,1024,1,b));
  for(int replay=0;replay<2;++replay){
   counts[0]=replay?257:1;counts[1]=replay?1:257;output[0]=output[1]=output[2]=0;*remaining=3;
   ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
   require(output[0]==3*selectedBlocks(a,counts[0])&&output[1]==3*selectedBlocks(b,counts[1]),"independent handle publication/replay mismatch");
   require(output[2]==0,"direct leaf read unused null count or executed inactive kernel");
   std::printf("gridPublication bits=%zu replay=%d first=%d second=%d passed=1\n",sizeof(GpuReal)*8,replay,output[0],output[1]);
  }
  ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));cudaFree(counts);cudaFree(output);cudaFree(remaining);cudaFree(mode);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
