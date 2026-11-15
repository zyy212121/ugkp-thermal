

#include "wall/GpuColdWall1DHandles.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* alloc(int n){T* p=nullptr;check(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<class R>struct State{
 int *pColdHandle,*compactPColdHandle,*coldFreeHandles,*coldFreeCounts;
 int particleCapacity,*pStatus,*particleCountDevice,*identity;
 R *pColdNodeSpecificEnthalpy,*pColdRingSolidMass,*pColdFrozenArea;double* pColdContactAge;
};
template<class R>State<R>* owner(int cap,int count){
 auto* s=alloc<State<R>>(1);s->particleCapacity=cap;s->particleCountDevice=alloc<int>(1);*s->particleCountDevice=count;
 s->pColdHandle=alloc<int>(cap);s->compactPColdHandle=alloc<int>(cap);s->coldFreeHandles=alloc<int>(cap);s->coldFreeCounts=alloc<int>(1);s->pStatus=alloc<int>(cap);s->identity=alloc<int>(cap);
 s->pColdNodeSpecificEnthalpy=alloc<R>(8*cap);s->pColdRingSolidMass=alloc<R>(8*cap);s->pColdFrozenArea=alloc<R>(cap);s->pColdContactAge=alloc<double>(cap);
 Foam::gpuWall::ColdWall1DHandles pool{s->pColdHandle,s->compactPColdHandle,s->coldFreeHandles,s->coldFreeCounts,cap};
 Foam::gpuWall::initialiseColdWall1DHandlesKernel<<<(cap+127)/128,128>>>(pool,count);check(cudaDeviceSynchronize());
 for(int i=0;i<cap;++i){s->pStatus[i]=i<count;s->identity[i]=i;for(int k=0;k<8;++k)s->pColdNodeSpecificEnthalpy[8*i+k]=R(16*i+k);}return s;
}
template<class R>void destroy(State<R>* s){
 for(int* p:{s->pColdHandle,s->compactPColdHandle,s->coldFreeHandles,s->coldFreeCounts,s->pStatus,s->particleCountDevice,s->identity})check(cudaFree(p));
 check(cudaFree(s->pColdNodeSpecificEnthalpy));check(cudaFree(s->pColdRingSolidMass));check(cudaFree(s->pColdFrozenArea));check(cudaFree(s->pColdContactAge));check(cudaFree(s));
}
template<class R>__global__ void deaths(State<R>* s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<*s->particleCountDevice&&i%3==0){s->pStatus[i]=0;Foam::gpuWall::releaseColdWall1DHandle(*s,i);Foam::gpuWall::releaseColdWall1DHandle(*s,i);}}
template<class R>__global__ void compact(State<R>* s){
 if(threadIdx.x||blockIdx.x)return;int count=0;for(int i=0;i<*s->particleCountDevice;++i)if(s->pStatus[i]){s->compactPColdHandle[count]=s->pColdHandle[i];s->identity[count]=s->identity[i];s->pStatus[count++]=1;}
 int* old=s->pColdHandle;s->pColdHandle=s->compactPColdHandle;s->compactPColdHandle=old;*s->particleCountDevice=count;
}
template<class R>__global__ void inject(State<R>* s,int requested,int identityBase){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=requested)return;int slot=atomicAdd(s->particleCountDevice,1);if(slot>=s->particleCapacity){atomicSub(s->particleCountDevice,1);return;}
 Foam::gpuWall::allocateColdWall1DHandle(*s,slot);s->pStatus[slot]=1;s->identity[slot]=identityBase+k;int h=s->pColdHandle[slot];for(int j=0;j<8;++j)s->pColdNodeSpecificEnthalpy[8*h+j]=R(16*s->identity[slot]+j);
}
template<class R>void verify(State<R>* s){check(cudaDeviceSynchronize());int n=*s->particleCountDevice;assert(n>=0&&n<=s->particleCapacity);assert(s->coldFreeCounts[0]+n==s->particleCapacity);std::set<int> used;for(int i=0;i<n;++i){int h=s->pColdHandle[i];assert(h>=0&&h<s->particleCapacity&&used.insert(h).second);for(int k=0;k<8;++k)assert(s->pColdNodeSpecificEnthalpy[8*h+k]==R(16*s->identity[i]+k));}for(int i=0;i<s->coldFreeCounts[0];++i)assert(used.insert(s->coldFreeHandles[i]).second);assert(used.size()==size_t(s->particleCapacity));}
struct Legacy{};
__global__ void nonCold(int* out){Legacy legacy;Foam::gpuWall::releaseColdWall1DHandle(legacy,0);Foam::gpuWall::allocateColdWall1DHandle(legacy,0);State<float> s{};Foam::gpuWall::releaseColdWall1DHandle(s,0);Foam::gpuWall::allocateColdWall1DHandle(s,0);float ordinary=0;out[0]=Foam::gpuWall::coldWallSlot(legacy,7);out[1]=Foam::gpuWall::coldWallFieldSlot(s,&ordinary,9);}
template<class R>void exercise(){
 auto* s=owner<R>(257,257);verify(s);

 std::vector<int> id(257);for(int i=0;i<257;++i){int src=i*17%257;id[i]=s->identity[src];s->compactPColdHandle[i]=s->pColdHandle[src];}for(int i=0;i<257;++i)s->identity[i]=id[i];int* old=s->pColdHandle;s->pColdHandle=s->compactPColdHandle;s->compactPColdHandle=old;verify(s);
 deaths<<<3,128>>>(s);


 check(cudaDeviceSynchronize());
 R* deadPacked=alloc<R>(257*8);Foam::gpuWall::gatherColdWall1DStateKernel<R,8><<<17,128>>>(s,s->pColdNodeSpecificEnthalpy,deadPacked,257);check(cudaDeviceSynchronize());for(int i=0;i<257;++i)for(int k=0;k<8;++k)assert(deadPacked[8*i+k]==(i%3==0?R(0):R(16*s->identity[i]+k)));check(cudaFree(deadPacked));
 compact<<<1,1>>>(s);inject<<<3,128>>>(s,300,1000);verify(s);assert(*s->particleCountDevice==257);
 cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamCreate(&stream));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));deaths<<<3,128,0,stream>>>(s);compact<<<1,1,0,stream>>>(s);inject<<<1,128,0,stream>>>(s,32,2000);check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int step=0;step<200;++step){check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));verify(s);}check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));

 int n=*s->particleCountDevice;R* packed=alloc<R>(n*8);Foam::gpuWall::gatherColdWall1DStateKernel<R,8><<<17,128>>>(s,s->pColdNodeSpecificEnthalpy,packed,n);check(cudaDeviceSynchronize());auto* restored=owner<R>(401,n);for(int i=0;i<n;++i){restored->identity[i]=s->identity[i];for(int j=0;j<8;++j)restored->pColdNodeSpecificEnthalpy[8*i+j]=packed[8*i+j];}verify(restored);inject<<<4,128>>>(restored,500,3000);verify(restored);

 std::vector<int> freeBefore(s->coldFreeHandles,s->coldFreeHandles+s->coldFreeCounts[0]);int freeCount=s->coldFreeCounts[0];for(int step=0;step<100;++step)inject<<<1,128>>>(s,0,0);check(cudaDeviceSynchronize());assert(s->coldFreeCounts[0]==freeCount);for(int i=0;i<freeCount;++i)assert(s->coldFreeHandles[i]==freeBefore[i]);
 check(cudaFree(packed));destroy(restored);destroy(s);
}
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;exercise<float>();exercise<double>();int* out=alloc<int>(2);nonCold<<<1,1>>>(out);check(cudaDeviceSynchronize());assert(out[0]==7&&out[1]==9);check(cudaFree(out));std::puts("cold-wall handle lifecycle CUDA fixture passed");}
