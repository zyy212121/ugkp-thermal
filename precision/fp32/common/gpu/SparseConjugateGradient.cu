#include "SparseConjugateGradient.H"
#include "PhysicalDiffusionOperator.cuh"
#include "ExecutionGraph.cuh"
#include "GraphContinuationTask.cuh"
#include "KernelResourceLaunch.cuh"
#include "WorkPlan.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <cstddef>
#include <new>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA sparse solve: ")+cudaGetErrorString(code));
}
namespace sparseGraphDetail {
template<class Capture> cudaGraphNode_t appendCaptured(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,cudaStream_t stream,Capture capture){
 checked(cudaStreamBeginCaptureToGraph(stream,graph,dependencies.data(),nullptr,dependencies.size(),cudaStreamCaptureModeThreadLocal));
 cudaGraph_t completed=nullptr;
 try{
  capture();cudaStreamCaptureStatus state;cudaGraph_t current=nullptr;const cudaGraphNode_t* tail=nullptr;size_t count=0;
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&current,&tail,nullptr,&count));
  std::vector<cudaGraphNode_t> ends(tail,tail+count);
  checked(cudaStreamEndCapture(stream,&completed));
  if(ends.size()==1)return ends[0];
  cudaGraphNode_t joined=nullptr;checked(cudaGraphAddEmptyNode(&joined,graph,ends.data(),ends.size()));return joined;
 }catch(...){cudaStreamCaptureStatus state;if(cudaStreamIsCapturing(stream,&state)==cudaSuccess&&state!=cudaStreamCaptureStatusNone)cudaStreamEndCapture(stream,&completed);throw;}
}
}
template<class T> struct DeviceArray {
    T* ptr=nullptr;std::size_t capacity=0;
    ~DeviceArray(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t count){
        if(count<=capacity)return;
        T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),count*sizeof(T)));
        if(ptr)cudaFree(ptr);ptr=replacement;capacity=count;
    }
    template<class Host> void upload(const Host* host,std::size_t count,PinnedTransferQueue& queue){
        reserve(count);queue.uploadConverted(ptr,host,count);
    }
};

__host__ __device__ inline int sparseThreads(){
#ifdef __CUDA_ARCH__
    return blockDim.x;
#else
    return workPlan().sparseBlockThreads;
#endif
}
#define threads sparseThreads()


template<class T> struct VcPointer {
 T* direct=nullptr;void* const* slot=nullptr;
 __host__ __device__ VcPointer()=default;
 __host__ __device__ VcPointer(T* p):direct(p){}
 __host__ __device__ VcPointer(T* p,void* const* s):direct(p),slot(s){}
 template<class U> __host__ __device__ VcPointer(const VcPointer<U>& p):direct(p.direct),slot(p.slot){}
 __device__ T* get()const{return slot?static_cast<T*>(*slot):direct;}
 __device__ operator T*()const{return get();}
};
enum VcRowSlot { vcRowOff,vcRowMap,vcRowCount,vcRowMembers,vcRowMemberOffsets,vcRowInverseOffsets,vcRowPaddedMembers,vcRowCursor,vcRowTemporaryMap,vcRowTemporaryCount,vcRowTemporaryMembers,vcRowChoice,vcRowLeader,vcRowRootFlag,vcRowCompact,vcRowScanOutput,vcRowScanScratch,vcRowMaxima,vcRowInverse,vcRowBounds,vcRowSmooth,vcRowResidual,vcRowR,vcRowZ,vcRowL,vcRowSlotCount };
constexpr int vcScanTile=256,vcScanStages=8;
struct VcStorage {
 void* values=nullptr;void* previous=nullptr;int* columns=nullptr;unsigned char* strong=nullptr;unsigned long long* keys=nullptr;
 std::size_t entries=0,hash=0,rows=0,inverseEntries=0;
 void* row[vcRowSlotCount]{};
 int scanCount[vcScanStages]{},scanTiles[vcScanStages]{},scanSumOffset[vcScanStages]{},scanPrefixOffset[vcScanStages]{};
 int scanPass[vcScanStages]{},scanAdd[vcScanStages]{},scanDepth=0;
};
template<class T> struct VcRowArray {
 VcPointer<T> ptr;
 void bind(VcStorage* storage,VcRowSlot slot){ptr=VcPointer<T>(nullptr,reinterpret_cast<void* const*>(reinterpret_cast<char*>(storage)+offsetof(VcStorage,row))+slot);}
};
struct VcPoolBlock{unsigned char* data=nullptr;std::size_t capacity=0,used=0;VcPoolBlock* next=nullptr;};
struct VcPool{VcPoolBlock* first=nullptr;VcStorage scratch{};};
struct VcCursor{
 unsigned char* base;std::size_t used=0;
 __host__ __device__ void* take(std::size_t bytes){used=(used+255)&~std::size_t(255);void* result=base?base+used:nullptr;used+=bytes;return result;}
};
__host__ __device__ std::size_t vcHighWater(std::size_t old,std::size_t requested){if(requested<=old)return old;std::size_t n=old?old:1;while(n<requested)n*=2;return n;}
template<class R> __host__ __device__ void vcStoragePlan(VcStorage& next,VcStorage& scratch,std::size_t rows,std::size_t entries,std::size_t inverse,std::size_t hash,bool fine,VcCursor& cursor){
 rows=vcHighWater(next.rows,rows);entries=vcHighWater(next.entries,entries);inverse=vcHighWater(next.inverseEntries,inverse);hash=vcHighWater(next.hash,hash);
 if(rows>next.rows){
  next.row[vcRowOff]=cursor.take((rows+1)*sizeof(int));
  next.row[vcRowMap]=cursor.take(rows*sizeof(int));
  next.row[vcRowCount]=cursor.take(rows*sizeof(int));
  next.row[vcRowMembers]=cursor.take(rows*sizeof(int));
  next.row[vcRowMemberOffsets]=cursor.take((rows+1)*sizeof(int));
  next.row[vcRowInverseOffsets]=cursor.take((rows+1)*sizeof(int));
  next.row[vcRowMaxima]=cursor.take(rows*sizeof(R));
  next.row[vcRowBounds]=cursor.take(rows*sizeof(R));
  next.row[vcRowSmooth]=cursor.take(rows*sizeof(R));
  next.row[vcRowResidual]=cursor.take(rows*sizeof(R));
  next.row[vcRowR]=cursor.take(rows*sizeof(R));
  next.row[vcRowZ]=cursor.take(rows*sizeof(R));
  next.row[vcRowL]=cursor.take(1024*sizeof(R));
  next.rows=rows;
 }
 if(entries>next.entries){next.previous=cursor.take(entries*sizeof(R));next.strong=static_cast<unsigned char*>(cursor.take(entries));if(!fine){next.values=cursor.take(entries*sizeof(R));next.columns=static_cast<int*>(cursor.take(entries*sizeof(int)));}next.entries=entries;}
 if(inverse>next.inverseEntries){next.row[vcRowInverse]=cursor.take(inverse*sizeof(R));next.inverseEntries=inverse;}
 if(hash>next.hash){next.keys=static_cast<unsigned long long*>(cursor.take(hash*sizeof(unsigned long long)));next.hash=hash;}
 if(rows>scratch.rows){
  std::size_t scan=0;for(std::size_t count=rows;count;){count=(count+vcScanTile-1)/vcScanTile;scan+=2*count;if(count==1)break;}
  scratch.row[vcRowPaddedMembers]=cursor.take(rows*8*sizeof(int));
  scratch.row[vcRowCursor]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowTemporaryMap]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowTemporaryCount]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowTemporaryMembers]=cursor.take(rows*8*sizeof(int));
  scratch.row[vcRowChoice]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowLeader]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowRootFlag]=cursor.take(rows*sizeof(int));
  scratch.row[vcRowCompact]=cursor.take((rows+1)*sizeof(int));
  scratch.row[vcRowScanOutput]=cursor.take((rows+1)*sizeof(int));
  scratch.row[vcRowScanScratch]=cursor.take(scan*sizeof(int));
  scratch.rows=rows;
 }
 next.row[vcRowPaddedMembers]=scratch.row[vcRowPaddedMembers];
 next.row[vcRowCursor]=scratch.row[vcRowCursor];
 next.row[vcRowTemporaryMap]=scratch.row[vcRowTemporaryMap];
 next.row[vcRowTemporaryCount]=scratch.row[vcRowTemporaryCount];
 next.row[vcRowTemporaryMembers]=scratch.row[vcRowTemporaryMembers];
 next.row[vcRowChoice]=scratch.row[vcRowChoice];
 next.row[vcRowLeader]=scratch.row[vcRowLeader];
 next.row[vcRowRootFlag]=scratch.row[vcRowRootFlag];
 next.row[vcRowCompact]=scratch.row[vcRowCompact];
 next.row[vcRowScanOutput]=scratch.row[vcRowScanOutput];
 next.row[vcRowScanScratch]=scratch.row[vcRowScanScratch];
}
struct VcAllocationRequest {int state=0,error=0;std::size_t poolBytes=0;};
template<class R> __global__ void vcRequestStorage(const int* rows,VcPointer<const int> offsets,VcStorage* storage,VcAllocationRequest* request,int* flags,const int* gateFlags,int* valid,int stage,bool root,int* pending,VcPool* pool,bool fineStorage){
 if(threadIdx.x||*pending)return;
 if(request->state==2){request->state=0;if(request->error){*valid=0;flags[19]=1;return;}}
 bool enabled=gateFlags[14]&&(root||*valid==1);
 if(stage==1)enabled=enabled&&!gateFlags[15];if(stage==2)enabled=enabled&&!gateFlags[15]&&gateFlags[1];if(stage==3)enabled=enabled&&gateFlags[15]!=2&&gateFlags[0];
 if(!enabled)return;
 std::size_t needed=stage==3?storage->entries:std::size_t(offsets[*rows]),hash=storage->hash,rowCount=stage==3?storage->rows:std::size_t(*rows),inverse=stage==3?std::size_t(offsets[*rows]):storage->inverseEntries;
 if(stage==1){hash=1;while(hash<2*needed)hash*=2;needed=storage->entries;}
 if(needed<=storage->entries&&rowCount<=storage->rows&&inverse<=storage->inverseEntries&&hash<=storage->hash&&pool->scratch.rows>=storage->rows){
  for(int slot=vcRowPaddedMembers;slot<=vcRowScanScratch;++slot){void* pointer=pool->scratch.row[slot];if(storage->row[slot]!=pointer)storage->row[slot]=pointer;}
  flags[19]=0;return;
 }
 bool reset=needed>storage->entries||rowCount>storage->rows;
 VcStorage next=*storage,scratch=pool->scratch;VcCursor measure{nullptr};vcStoragePlan<R>(next,scratch,rowCount,needed,inverse,hash,fineStorage,measure);
 if(measure.used){
  VcPoolBlock* block=pool->first;std::size_t at=0;for(;block;block=block->next){at=(block->used+255)&~std::size_t(255);if(at<=block->capacity&&measure.used<=block->capacity-at)break;}
  if(!block){request->poolBytes=measure.used;request->error=0;request->state=1;__threadfence_system();*pending=1;return;}
  next=*storage;scratch=pool->scratch;VcCursor fill{block->data+at};vcStoragePlan<R>(next,scratch,rowCount,needed,inverse,hash,fineStorage,fill);block->used=at+fill.used;
 }
 *storage=next;pool->scratch=scratch;if(reset)flags[2]=0;flags[19]=0;
}
__global__ void vcBeginContinuation(int count,int** flags,int* progressing,const int* pending){
 if(threadIdx.x||*pending||*progressing)return;for(int i=0;i<count;++i)flags[i][20]=0;*progressing=1;
}
__global__ void vcSelectStage(int* flags,int stage,const int* pending){if(!threadIdx.x)flags[21]=!*pending&&flags[20]==stage;}
__global__ void vcFinishStage(int* flags,int stage,const int* pending){if(!threadIdx.x&&!*pending)flags[20]=stage+1;}
__global__ void vcSelectReady(const int* pending,cudaGraphConditionalHandle handle){if(!threadIdx.x)cudaGraphSetConditional(handle,!pending||!*pending);}
__global__ void vcFinishContinuation(int* progressing){if(!threadIdx.x)*progressing=0;}

struct MatrixOperator {
    VcPointer<const int> offsets;
    const int* columns;
    const GpuReal* values;
    const GpuReal* completedDiagonal;
    const int* entrySource;
    const int* entryKind;
    const GpuReal* upper;
    const GpuReal* lower;
    const GpuReal* boundarySource;
    SparsePhysicalDiffusionValues diffusion{};
    const int* activeRows=nullptr;
    const int* refreshFlags=nullptr;
    const VcStorage* storage=nullptr;
    __device__ int column(int entry)const{return storage?storage->columns[entry]:columns[entry];}
    __device__ GpuReal diagonal(int row)const{
        return completedDiagonal?completedDiagonal[row]:entryCoefficient(offsets[row]);
    }
    __device__ GpuReal entryCoefficient(int entry)const{
        if(!completedDiagonal)return storage?static_cast<const GpuReal*>(storage->values)[entry]:values[entry];
        const int face=entrySource[entry],kind=entryKind[entry];
        if(diffusion.active)return kind==0?completedDiagonal[face]:kind==3?-physicalDiffusionDetail::boundarySource(diffusion,face):physicalDiffusionDetail::internal(diffusion,face);
        return kind==0?completedDiagonal[face]:kind==1?lower[face]:kind==2?upper[face]:-boundarySource[face];
    }
    __device__ GpuReal product(int row,const GpuReal* input)const{
        GpuReal value=(diffusion.active?physicalDiffusionDetail::completedDiagonal(diffusion,row):diagonal(row))*input[row];
        for(int e=offsets[row]+1;e<offsets[row+1];++e)value+=entryCoefficient(e)*input[column(e)];
        return value;
    }
};


__global__ void initializeDispatch(int* state){if(!threadIdx.x)*state=2;}
__global__ void classifyCoupling(int n,MatrixOperator matrix,int* state){
 if(matrix.refreshFlags&&(!matrix.refreshFlags[0]||matrix.refreshFlags[19]))return;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
 GpuReal diagonal=matrix.diagonal(i),off=0;
 for(int e=matrix.offsets[i]+1;e<matrix.offsets[i+1];++e)off+=fabs(matrix.entryCoefficient(e));
 if(!(diagonal>GpuReal(0))||!isfinite(diagonal)||!isfinite(off)||!(off<diagonal)||!(off/diagonal<=GpuReal(.98)))atomicExch(state,1);
}
}
__global__ void selectDispatch(const int* state,cudaGraphConditionalHandle diagonal,cudaGraphConditionalHandle multigrid){
 if(!threadIdx.x){cudaGraphSetConditional(diagonal,*state==2);cudaGraphSetConditional(multigrid,*state!=2);}
}


template<class R> __device__ R vcConnectionStrength(R value,R diagonalA,R diagonalB){
 R a=sqrt(fabs(diagonalA)),b=sqrt(fabs(diagonalB));
 if(!(a>R(0))||!(b>R(0)))return R(0);
 return (fabs(value)/fmax(a,b))/fmin(a,b);
}
__device__ unsigned long long vcPairPriority(int a,int b){
 unsigned long long low=static_cast<unsigned int>(a<b?a:b),high=static_cast<unsigned int>(a<b?b:a);
 unsigned long long key=(high<<32)|low;
 key+=0x9e3779b97f4a7c15ull;key=(key^(key>>30))*0xbf58476d1ce4e5b9ull;
 key=(key^(key>>27))*0x94d049bb133111ebull;return key^(key>>31);
}
__global__ void vcCacheBegin(int* flags,const int* parent){
 if(threadIdx.x||!flags[14])return;flags[0]=!flags[2]||flags[3]||(parent&&parent[0]);flags[1]=!flags[2]||flags[4]||(parent&&parent[1]);flags[5]=parent&&parent[0];flags[10]=parent&&parent[1];
}
template<class R> __global__ void vcCacheRows(int n,MatrixOperator matrix,VcStorage* storage,VcPointer<R> maxima,int* flags,const int* valid){
 R* previous=static_cast<R*>(storage->previous);
 if((valid&&*valid!=1)||!flags[14]||flags[19])return;
 if(matrix.activeRows)n=*matrix.activeRows;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
 R largest=R(0);
 for(int e=matrix.offsets[i];e<matrix.offsets[i+1];++e){
  R value=matrix.entryCoefficient(e);if(!flags[2]||value!=previous[e]){atomicExch(flags,1);atomicExch(flags+5,1);}previous[e]=value;
  if(matrix.column(e)!=i){R strength=vcConnectionStrength(value,matrix.diagonal(i),matrix.diagonal(matrix.column(e)));largest=fmax(largest,strength);}
 }
 maxima[i]=largest;
}
}
template<class R> __global__ void vcCacheEdges(int n,MatrixOperator matrix,VcPointer<const R> maxima,VcStorage* storage,int* flags,const int* valid){
 unsigned char* strong=storage->strong;
 if((valid&&*valid!=1)||!flags[14]||flags[19])return;
 if(matrix.activeRows)n=*matrix.activeRows;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
 for(int e=matrix.offsets[i];e<matrix.offsets[i+1];++e){
  int j=matrix.column(e);R strength=vcConnectionStrength(matrix.entryCoefficient(e),matrix.diagonal(i),matrix.diagonal(j));
  unsigned char edge=j!=i&&strength>R(0)&&strength>=R(.25)*fmax(maxima[i],maxima[j]);
  if(!flags[2]||strong[e]!=edge){atomicExch(flags+1,1);atomicExch(flags+10,1);}strong[e]=edge;
 }
}
}
__global__ void vcCacheObserve(int* flags,const int* valid){
 if(threadIdx.x||(valid&&*valid!=1)||!flags[14]||flags[19])return;
 flags[2]=1;flags[3]|=flags[0];flags[4]|=flags[1];if(flags[0])flags[12]=0;if(flags[1])flags[11]=0;
 flags[6]+=flags[5]!=0;flags[7]+=flags[10]!=0;
}
__global__ void vcCacheCommit(int* flags,int* built,const int* valid){
 if(threadIdx.x||*valid!=1||!flags[14])return;
 flags[8]=flags[6];flags[9]=flags[7];flags[3]=0;flags[4]=0;flags[11]=1;flags[12]=1;*built=1;
}
__global__ void vcCommitHierarchy(int count,int** allFlags,int** allBuilt,const int* depth,const int* valid){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||i>*depth||*valid!=1)return;int* flags=allFlags[i];
 if(!flags[14]||flags[19])return;flags[8]=flags[6];flags[9]=flags[7];flags[3]=0;flags[4]=0;flags[11]=1;flags[12]=1;*allBuilt[i]=1;
}
__global__ void vcNextLevel(const int* parent,const int* valid,int* next){if(!threadIdx.x)next[14]=*valid==1&&parent[14]&&!parent[15];}
__global__ void vcDispatchBegin(const int* flags,const int* saved,int* valid){if(!threadIdx.x)*valid=flags[19]?0:(flags[0]?2:*saved);}
__global__ void vcDispatchSave(const int* valid,int* saved,int* rootFlags){if(!threadIdx.x){*saved=*valid;if(*valid==0)rootFlags[3]=1;}}
__global__ void vcLevelBegin(int level,const int* depth,int* flags){
 if(!threadIdx.x){flags[14]=level<=*depth;flags[19]=0;}
}
__global__ void vcPrepareGrouping(int level,int last,const int* rows,int* depth,int* flags,const int* built,const int* valid){
 if(threadIdx.x)return;flags[13]=0;
 if(!flags[14]||*valid!=1)return;
 if(*rows<=32){flags[15]=2;*depth=level;return;}
 if(!*built){flags[0]=flags[1]=flags[3]=flags[4]=1;}
 if(flags[1]){flags[13]=1;flags[15]=0;*depth=last;}
}
__global__ void vcFinishGrouping(int level,const int* rows,const int* groups,int* depth,int* flags){
 if(threadIdx.x||!flags[13])return;
 if(static_cast<long long>(*groups)*5>=static_cast<long long>(*rows)*4){flags[15]=1;*depth=level;}
}
__global__ void vcHashExtent(MatrixOperator matrix,int* flags,const int* valid){
 if(threadIdx.x||*valid!=1||!flags[14]||flags[15])return;
 int n=*matrix.activeRows,needed=2*matrix.offsets[n],capacity=1;
 while(capacity<needed)capacity*=2;flags[17]=capacity;
}

__global__ void vcScanSetup(VcStorage* storage,const int* count,const int* enabled){
 if(threadIdx.x)return;int n=*enabled?*count:0,offset=0,depth=0;
 for(int i=0;i<vcScanStages;++i){
  storage->scanCount[i]=n;storage->scanPass[i]=n>0;
  int tiles=(n+vcScanTile-1)/vcScanTile;storage->scanTiles[i]=tiles;
  storage->scanSumOffset[i]=offset;offset+=tiles;storage->scanPrefixOffset[i]=offset;offset+=tiles;
  storage->scanAdd[i]=tiles>1;if(n)depth=i;
  n=tiles>1?tiles:0;
 }
 storage->scanDepth=depth;
}
__global__ void vcScanTiles(VcStorage* storage,VcPointer<const int> input,VcPointer<int> output,int square,int depth){
 int n=storage->scanCount[depth],tiles=storage->scanTiles[depth];
 int* scratch=static_cast<int*>(storage->row[vcRowScanScratch]);
 const int* source=depth?scratch+storage->scanSumOffset[depth-1]:input.get();
 int* target=depth?scratch+storage->scanPrefixOffset[depth-1]:output.get();
 __shared__ int partial[vcScanTile];
 for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x){
  int i=tile*vcScanTile+threadIdx.x,value=i<n?source[i]:0;
  if(!depth&&square)value*=value;partial[threadIdx.x]=value;__syncthreads();
  for(int shift=1;shift<vcScanTile;shift*=2){int add=threadIdx.x>=shift?partial[threadIdx.x-shift]:0;__syncthreads();partial[threadIdx.x]+=add;__syncthreads();}
  if(i<n)target[i]=partial[threadIdx.x]-value;
  if(threadIdx.x==vcScanTile-1)scratch[storage->scanSumOffset[depth]+tile]=partial[threadIdx.x];
  __syncthreads();
 }
}
__global__ void vcScanAdd(VcStorage* storage,VcPointer<int> output,int depth){
 int n=storage->scanCount[depth];int* scratch=static_cast<int*>(storage->row[vcRowScanScratch]);
 int* target=depth?scratch+storage->scanPrefixOffset[depth-1]:output.get();
 const int* prefix=scratch+storage->scanPrefixOffset[depth];
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)target[i]+=prefix[i/vcScanTile];
}
__global__ void vcScanTotal(VcStorage* storage,VcPointer<int> output,const int* count,const int* enabled){
 if(threadIdx.x||!*enabled)return;int n=*count;
 output[n]=n?static_cast<int*>(storage->row[vcRowScanScratch])[storage->scanSumOffset[storage->scanDepth]]:0;
}
__global__ void vcPublishScanOffsets(int capacity,const int* groups,VcPointer<const int> scanned,VcPointer<int> offsets,const int* enabled){
 if(!*enabled)return;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<=*groups;i+=blockDim.x*gridDim.x){offsets[i]=scanned[i];
}
}
__global__ void vcPackMembers(const int* groups,VcPointer<const int> count,VcPointer<const int> padded,VcPointer<const int> offsets,VcPointer<int> packed,const int* enabled){
 if(!*enabled)return;for(int g=blockIdx.x*blockDim.x+threadIdx.x;g<*groups;g+=blockDim.x*gridDim.x){
 for(int i=0;i<count[g];++i)packed[offsets[g]+i]=padded[8*g+i];
}
}
template<class R> __global__ void matchingInitialize(const int* rows,int* groups,VcPointer<int> map,VcPointer<int> count,VcPointer<int> members,const int* enabled){
 if(!*enabled)return;
 int n=*rows;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){map[i]=i;count[i]=1;members[8*i]=i;if(!i)*groups=n;
}
}
template<class R> __global__ void matchingRowMaximum(const int* rows,MatrixOperator matrix,VcPointer<R> rowMaximum,const int* enabled){
 if(!*enabled)return;
 int n=*rows;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){R best=R(0),di=fabs(matrix.diagonal(i));
 for(int e=matrix.offsets[i]+1;e<matrix.offsets[i+1];++e){int j=matrix.column(e);R dj=fabs(matrix.diagonal(j));R strength=vcConnectionStrength(matrix.entryCoefficient(e),di,dj);best=fmax(best,strength);}rowMaximum[i]=best;
}
}
template<class R> __global__ void matchingChoose(int capacity,const int* groups,int target,MatrixOperator matrix,VcPointer<const int> map,VcPointer<const int> count,VcPointer<const int> members,VcPointer<const R> rowMaximum,VcPointer<int> choice,const int* enabled){
 if(!*enabled)return;
 for(int g=blockIdx.x*blockDim.x+threadIdx.x;g<*groups;g+=blockDim.x*gridDim.x){R best=R(-1);unsigned long long bestPriority=0;int selected=-1;
 for(int slot=0;slot<count[g];++slot){int i=members[8*g+slot];R di=fabs(matrix.diagonal(i));
  for(int e=matrix.offsets[i]+1;e<matrix.offsets[i+1];++e){int j=matrix.column(e),h=map[j];if(h==g||count[g]+count[h]>target)continue;R dj=fabs(matrix.diagonal(j));R strength=vcConnectionStrength(matrix.entryCoefficient(e),di,dj);
   if(!(strength>R(0))||strength<R(.25)*fmax(rowMaximum[i],rowMaximum[j]))continue;
   unsigned long long priority=vcPairPriority(g,h);
   if(strength>best||(strength==best&&(priority>bestPriority||(priority==bestPriority&&(selected<0||h<selected))))){best=strength;bestPriority=priority;selected=h;}
  }
 }
 choice[g]=selected;
}
}
__global__ void matchingLeaders(int capacity,const int* groups,VcPointer<const int> choice,VcPointer<int> leader,VcPointer<int> rootFlag,const int* enabled){
 if(!*enabled)return;
 for(int g=blockIdx.x*blockDim.x+threadIdx.x;g<*groups;g+=blockDim.x*gridDim.x){int h=choice[g];int root=h>=0&&choice[h]==g?(h<g?h:g):g;leader[g]=root;rootFlag[g]=root==g?1:0;
}
}
__global__ void matchingScanTotal(const int* groups,VcPointer<const int> compact,int* nextGroups,const int* enabled){
 if(!threadIdx.x&&*enabled)*nextGroups=compact[*groups];
}
__global__ void matchingPublish(int capacity,const int* groups,VcPointer<const int> choice,VcPointer<const int> leader,VcPointer<const int> compact,VcPointer<const int> count,VcPointer<const int> members,VcPointer<int> nextMap,VcPointer<int> nextCount,VcPointer<int> nextMembers,const int* enabled){
 if(!*enabled)return;
 for(int g=blockIdx.x*blockDim.x+threadIdx.x;g<*groups;g+=blockDim.x*gridDim.x){if(leader[g]!=g)continue;int out=compact[g],used=0;
 for(int k=0;k<count[g];++k){int cell=members[8*g+k];nextMembers[8*out+used++]=cell;nextMap[cell]=out;}
 int h=choice[g];if(h>=0&&h!=g&&leader[h]==g)for(int k=0;k<count[h];++k){int cell=members[8*h+k];nextMembers[8*out+used++]=cell;nextMap[cell]=out;}
 nextCount[out]=used;
}
}

template<class R> __global__ void vcLineFactor(const int* groups,VcPointer<const int> map,VcPointer<const int> count,VcPointer<const int> members,VcPointer<const int> memberOffsets,VcPointer<const int> inverseOffsets,MatrixOperator matrix,VcPointer<R> inverse,VcPointer<R> bounds,int* valid){
 if(*valid!=1||!matrix.refreshFlags[14]||!matrix.refreshFlags[0])return;
 if(matrix.refreshFlags[15]==2)return;
 int nc=*groups;for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<nc;c+=blockDim.x*gridDim.x){
 int size=count[c];if(!size){bounds[c]=R(1);continue;}R factor[64],reciprocal[64],external[8];int good=1;
 for(int i=0;i<64;++i){factor[i]=R(0);reciprocal[i]=R(0);}
 for(int i=0;i<size;++i){
  int row=members[memberOffsets[c]+i];external[i]=R(0);
  for(int e=matrix.offsets[row];e<matrix.offsets[row+1];++e){
   int j=matrix.column(e);R value=matrix.entryCoefficient(e);
   if(map[j]!=c)external[i]+=fabs(value);
   else for(int k=0;k<size;++k)if(members[memberOffsets[c]+k]==j){factor[i*8+k]+=value;break;}
  }
 }
 for(int i=0;i<size;++i)for(int j=0;j<=i;++j){
  R value=factor[i*8+j];for(int k=0;k<j;++k)value-=factor[i*8+k]*factor[j*8+k];
  if(i==j){if(!(value>R(0))||!isfinite(value)){good=0;value=R(1);}factor[i*8+j]=sqrt(value);}
  else factor[i*8+j]=value/factor[j*8+j];
 }
 for(int i=0;i<size;++i)for(int j=0;j<=i;++j){
  R value=i==j?R(1):R(0);for(int k=j;k<i;++k)value-=factor[i*8+k]*reciprocal[k*8+j];
  reciprocal[i*8+j]=value/factor[i*8+i];
 }
 R bound=R(1);
 for(int i=0;i<size;++i){
  R rowBound=R(1);
  for(int j=0;j<size;++j){
   R value=R(0);for(int k=(i>j?i:j);k<size;++k)value+=reciprocal[k*8+i]*reciprocal[k*8+j];
   inverse[inverseOffsets[c]+i*size+j]=value;rowBound+=fabs(value)*external[j];
   if(!isfinite(value))good=0;
  }
  if(!(rowBound>R(0))||!isfinite(rowBound))good=0;else bound=fmax(bound,rowBound);
 }
 bounds[c]=bound;if(!good)atomicExch(valid,0);
}
}
template<class R> __global__ void vcLineWeight(const int* groups,VcPointer<const R> bounds,R* weight,int* valid,const int* flags){
 if(*valid!=1||!flags[14]||flags[15]==2||!flags[0])return;
 int nc=*groups;
 extern __shared__ unsigned char bytes[];R* tmp=reinterpret_cast<R*>(bytes);
 R value=R(1);for(int i=threadIdx.x;i<nc;i+=blockDim.x)value=fmax(value,bounds[i]);
 tmp[threadIdx.x]=value;__syncthreads();
 for(int width=blockDim.x;width>1;width=(width+1)/2){if(threadIdx.x<width/2)tmp[threadIdx.x]=fmax(tmp[threadIdx.x],tmp[threadIdx.x+(width+1)/2]);__syncthreads();}
 if(!threadIdx.x){R result=R(1)/tmp[0];*weight=result;if(!(result>R(0))||!isfinite(result))atomicExch(valid,0);}
}
__global__ void vcInitializeCoarse(int capacity,VcStorage* storage,const int* groups,VcPointer<int> counts,const int* valid,const int* flags){
 if(*valid!=1||!flags[14]||flags[15]||!flags[1])return;
 capacity=flags[17];int nc=*groups;auto keys=storage->keys;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<capacity;i+=blockDim.x*gridDim.x){keys[i]=~0ull;if(i<nc)counts[i]=0;}
}
__global__ void vcCoarseKeys(int n,const int* groups,MatrixOperator matrix,VcPointer<const int> map,VcStorage* storage,int mask,const int* valid){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15]||!matrix.refreshFlags[1])return;
 int nc=*groups;mask=matrix.refreshFlags[17]-1;auto keys=storage->keys;
 if(matrix.activeRows)n=*matrix.activeRows;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){int row=map[i];
 for(int e=matrix.offsets[i];e<matrix.offsets[i+1];++e){
  int col=map[matrix.column(e)];if(row>col)continue;
  unsigned long long key=static_cast<unsigned long long>(row)*nc+col;
  unsigned long long mixed=key; mixed^=mixed>>33;mixed*=0xff51afd7ed558ccdull;mixed^=mixed>>33;
  int slot=int(mixed&static_cast<unsigned long long>(mask));
  for(;;){unsigned long long old=atomicCAS(keys+slot,~0ull,key);if(old==~0ull||old==key)break;slot=(slot+1)&mask;}
 }
}
}
__global__ void vcCoarseCounts(int capacity,const int* groups,const VcStorage* storage,VcPointer<int> counts,const int* valid,const int* flags){
 if(*valid!=1||!flags[14]||flags[15]||!flags[1])return;
 capacity=flags[17];int nc=*groups;auto keys=storage->keys;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<capacity;i+=blockDim.x*gridDim.x){unsigned long long key=keys[i];if(key==~0ull)continue;
 int row=int(key/nc),col=int(key%nc);atomicAdd(counts+row,1);if(row!=col)atomicAdd(counts+col,1);}
}
__global__ void vcCoarseScanEnabled(int* flags,const int* valid){
 if(!threadIdx.x)flags[18]=*valid==1&&flags[14]&&!flags[15]&&flags[1];
}
__global__ void vcCoarseOffsets(int capacity,const int* groups,VcPointer<const int> scanned,VcPointer<int> counts,VcPointer<int> offsets,VcStorage* storage,const int* valid,const int* flags){
 if(*valid!=1||!flags[14]||flags[15]||!flags[1])return;
 int* columns=storage->columns;auto values=static_cast<GpuReal*>(storage->values);
 int nc=*groups;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<=nc;i+=blockDim.x*gridDim.x){offsets[i]=scanned[i];
 if(i<nc){int start=scanned[i];counts[i]=start+1;columns[start]=i;values[start]=1;}
}
}
template<class R> __global__ void vcCoarseValues(int capacity,const int* groups,const VcStorage* storage,VcPointer<const int> map,VcPointer<const int> count,VcPointer<const int> members,VcPointer<const int> memberOffsets,MatrixOperator matrix,VcPointer<int> cursor,VcPointer<const int> offsets,VcStorage* output,const int* valid){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15]||!matrix.refreshFlags[0])return;
 capacity=matrix.refreshFlags[17];int nc=*groups;auto keys=storage->keys;auto columns=output->columns;auto values=static_cast<R*>(output->values);
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<capacity;i+=blockDim.x*gridDim.x){unsigned long long key=keys[i];if(key==~0ull)continue;
 int row=int(key/nc),col=int(key%nc);R sum=R(0);
 for(int k=0;k<count[row];++k){int cell=members[memberOffsets[row]+k];for(int e=matrix.offsets[cell];e<matrix.offsets[cell+1];++e)if(map[matrix.column(e)]==col)sum+=matrix.entryCoefficient(e);}
 if(row==col)values[offsets[row]]=sum;
 else{int a=-1,b=-1;
  if(matrix.refreshFlags[1]){a=atomicAdd(cursor+row,1);b=atomicAdd(cursor+col,1);columns[a]=col;columns[b]=row;}
  else{for(int e=offsets[row]+1;e<offsets[row+1];++e)if(columns[e]==col){a=e;break;}for(int e=offsets[col]+1;e<offsets[col+1];++e)if(columns[e]==row){b=e;break;}}
  values[a]=sum;values[b]=sum;}}
}
template<class R> __global__ void vcLineApply(const int* groups,VcPointer<const int> count,VcPointer<const int> members,VcPointer<const int> memberOffsets,VcPointer<const int> inverseOffsets,VcPointer<const R> inverse,const R* weight,VcPointer<const R> rhs,VcPointer<R> result,const R* stop,const int* valid,bool add,const int* flags,int terminal){
 if(*valid!=1||!flags[14]||(flags[15]!=terminal)||(stop&&stop[7]))return;
 int nc=*groups;
 for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<nc;c+=blockDim.x*gridDim.x){int size=count[c];R local[8];
 for(int i=0;i<size;++i)local[i]=rhs[members[memberOffsets[c]+i]];
 for(int i=0;i<size;++i){R value=R(0);for(int j=0;j<size;++j)value+=inverse[inverseOffsets[c]+i*size+j]*local[j];int row=members[memberOffsets[c]+i];value*=*weight;if(add)result[row]+=value;else result[row]=value;}
}
}
template<class R> __global__ void vcDown(const int* groups,VcPointer<const int> count,VcPointer<const int> members,VcPointer<const int> memberOffsets,MatrixOperator matrix,VcPointer<const R> smooth,VcPointer<const R> r,VcPointer<R> rc,const R* stop,const int* valid){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15])return;
 if(stop&&stop[7])return;int nc=*groups;for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<nc;c+=blockDim.x*gridDim.x){
 R sum=R(0);for(int t=0;t<count[c];++t){int i=members[memberOffsets[c]+t];sum+=r[i]-matrix.product(i,smooth);}rc[c]=sum;
}
}
template<class R> __global__ void vcLeafFactor(int n,MatrixOperator matrix,VcPointer<R> L,int* valid){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15]!=2||!matrix.refreshFlags[0])return;
 if(matrix.activeRows)n=*matrix.activeRows;if(n>32)return;
 const int i=threadIdx.x;if(i>=32)return;R row[32];int good=1;
 for(int j=0;j<32;++j){R q=R(0);if(i<n&&j<n){for(int e=matrix.offsets[i];e<matrix.offsets[i+1];++e)if(matrix.column(e)==j)q+=matrix.entryCoefficient(e);}else if(i==j)q=R(1);row[j]=q;}
 for(int k=0;k<32;++k){if(i==k){R q=row[k];for(int j=0;j<k;++j)q-=row[j]*row[j];if(!(q>R(0))||!isfinite(q)){good=0;q=R(1);}row[k]=sqrt(q);}R diag=__shfl_sync(0xffffffffu,row[k],k);R q=row[k];for(int j=0;j<k;++j){R v=__shfl_sync(0xffffffffu,row[j],k);if(i>k)q-=row[j]*v;}if(i>k)row[k]=q/diag;}
 for(int j=0;j<32;++j)L[i*32+j]=i==j?R(1)/row[j]:row[j];if(!__all_sync(0xffffffffu,good)&&i==0)*valid=0;
}
template<class R> __global__ void vcLeafSolve(int n,MatrixOperator matrix,VcPointer<const R> L,VcPointer<const R> r,VcPointer<R> z,const R* stop,const int* valid){
 if((stop&&stop[7])||*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15]!=2)return;if(matrix.activeRows)n=*matrix.activeRows;
 if(n>32)return;
 if(blockIdx.x||threadIdx.x>=32)return;int i=threadIdx.x;R q=i<n?r[i]:R(0);
 for(int k=0;k<32;++k){if(i==k)q*=L[i*32+i];R v=__shfl_sync(0xffffffffu,q,k);if(i>k)q-=L[i*32+k]*v;}
 for(int k=31;k>=0;--k){if(i==k)q*=L[i*32+i];R v=__shfl_sync(0xffffffffu,q,k);if(i<k)q-=L[k*32+i]*v;}
 if(i<n)z[i]=q;
}
template<class R> __global__ void vcUp(int n,VcPointer<const int> map,MatrixOperator matrix,VcPointer<const R> smooth,VcPointer<const R> r,VcPointer<const R> ec,VcPointer<R> z,VcPointer<R> residual,const R* stop,const int* valid,bool finest){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15])return;
 if(matrix.activeRows)n=*matrix.activeRows;
 if(stop&&stop[7])return;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
 if(*valid!=1){if(finest)z[i]=r[i]/matrix.diagonal(i);continue;}
 R ax=R(0);for(int e=matrix.offsets[i];e<matrix.offsets[i+1];++e){int j=matrix.column(e);ax+=matrix.entryCoefficient(e)*(smooth[j]+ec[map[j]]);}
 z[i]=smooth[i]+ec[map[i]];residual[i]=r[i]-ax;
}
}
template<class R> __global__ void vcTerminalResidual(MatrixOperator matrix,VcPointer<const R> rhs,VcPointer<R> result,VcPointer<R> residual,const R* stop,const int* valid,bool first){
 if(*valid!=1||!matrix.refreshFlags[14]||matrix.refreshFlags[15]!=1||(stop&&stop[7]))return;
 int n=*matrix.activeRows;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
 if(first){result[i]=R(0);residual[i]=rhs[i];}
 else residual[i]=rhs[i]-matrix.product(i,result);
}
}
template<class R> __global__ void vcApplyFallback(int n,MatrixOperator matrix,VcPointer<const R> rhs,VcPointer<R> result,const R* stop,const int* valid){
 if(*valid==1||(stop&&stop[7]))return;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){result[i]=rhs[i]/matrix.diagonal(i);
}
}
template<class R> __global__ void vcSelectApply(const R* stop,const int* valid,cudaGraphConditionalHandle fallback,cudaGraphConditionalHandle hierarchy){
 if(threadIdx.x)return;bool running=!stop||!stop[7];cudaGraphSetConditional(fallback,running&&*valid!=1);cudaGraphSetConditional(hierarchy,running&&*valid==1);
}
template<class R> __global__ void vcSelectLevel(const int* flags,const R* stop,const int* valid,cudaGraphConditionalHandle normal,cudaGraphConditionalHandle direct,cudaGraphConditionalHandle terminal){
 if(threadIdx.x)return;bool running=*valid==1&&flags[14]&&(!stop||!stop[7]);
 cudaGraphSetConditional(normal,running&&flags[15]==0);cudaGraphSetConditional(direct,running&&flags[15]==2);cudaGraphSetConditional(terminal,running&&flags[15]==1);
}
__global__ void vcSelectFlag(const int* flag,cudaGraphConditionalHandle handle){if(!threadIdx.x)cudaGraphSetConditional(handle,*flag!=0);}
template<class R> struct VCycle{
 struct Level{
  int n=0,nc=0,hashCapacity=0;
  DeviceArray<int> seed,flags,built,rows,temporaryGroups;
  VcRowArray<int> off,map,count,members,memberOffsets,inverseOffsets,paddedMembers,cursor,temporaryMap,temporaryCount,temporaryMembers,choice,leader,rootFlag,compact,scanOutput,scanScratch;
  VcRowArray<R> maxima,inverse,bounds,smooth,residual,r,z,L;
  DeviceArray<R> weight;
  DeviceArray<VcStorage> storage;
  VcAllocationRequest* request=nullptr;VcAllocationRequest* deviceRequest=nullptr;
  VcStorage hostStorage{};
 };
 std::vector<std::unique_ptr<Level>> levels;
 DeviceArray<int> savedValid,activeDepth,pending,progressing;
 DeviceArray<int*> commitFlags,commitBuilt;
 std::vector<cudaStream_t> captureStreams;
 std::vector<KernelGridSelection>* applyGridSelections=nullptr;
 const int* applyGridValid=nullptr;const int* applyGridActive=nullptr;const int* applyGridMode=nullptr;
 struct GridSelectionGuard {
  VCycle& owner;const int* valid;const int* active;const int* mode;
  GridSelectionGuard(VCycle& o,const int* v,const int* a,const int* m):owner(o),valid(o.applyGridValid),active(o.applyGridActive),mode(o.applyGridMode){o.applyGridValid=v;o.applyGridActive=a;o.applyGridMode=m;}
  ~GridSelectionGuard(){owner.applyGridValid=valid;owner.applyGridActive=active;owner.applyGridMode=mode;}
 };
 DeviceArray<VcPool> pool;
 struct PoolAllocation{void* data;VcPoolBlock* descriptor;};
 std::vector<PoolAllocation> poolBlocks;cudaStream_t poolUploadStream=nullptr;std::size_t poolChunkBytes=1u<<20;cudaMemPool_t backingPool=nullptr;long long maximumHashTasks=1;
 cudaStream_t captureStream=nullptr;
 cudaGraph_t applyGraph=nullptr;cudaGraphExec_t applyExecutable=nullptr;ExecutionGraphSignature applySignature;
 cudaStream_t ownerStream=nullptr;
 GraphContinuationTask refreshTask;
 cudaGraph_t refreshGraph=nullptr;cudaGraphExec_t refreshExecutable=nullptr;ExecutionGraphSignature refreshSignature;
 ~VCycle(){try{release();}catch(...){}}
 cudaStream_t borrowCaptureStream(){
  for(auto candidate:captureStreams){cudaStreamCaptureStatus state;checked(cudaStreamIsCapturing(candidate,&state));if(state==cudaStreamCaptureStatusNone)return candidate;}
  cudaStream_t created=nullptr;checked(cudaStreamCreateWithFlags(&created,cudaStreamNonBlocking));captureStreams.push_back(created);return created;
 }
 void clearRefresh(){if(refreshExecutable)cudaGraphExecDestroy(refreshExecutable);if(refreshGraph)cudaGraphDestroy(refreshGraph);refreshExecutable=nullptr;refreshGraph=nullptr;}
 void release(){
  std::exception_ptr failure;try{refreshTask.wait();}catch(...){failure=std::current_exception();}refreshTask.reset();
  if(ownerStream)cudaStreamSynchronize(ownerStream);clearApply();clearRefresh();
  for(auto& p:levels)if(p->request)cudaFreeHost(p->request);
  for(auto block:poolBlocks){cudaFreeAsync(block.data,poolUploadStream);cudaFreeAsync(block.descriptor,poolUploadStream);}if(poolUploadStream)cudaStreamSynchronize(poolUploadStream);poolBlocks.clear();
  if(backingPool)cudaMemPoolDestroy(backingPool);backingPool=nullptr;
  if(poolUploadStream)cudaStreamDestroy(poolUploadStream);poolUploadStream=nullptr;
  levels.clear();if(captureStream)cudaStreamDestroy(captureStream);for(auto s:captureStreams)cudaStreamDestroy(s);captureStreams.clear();captureStream=nullptr;ownerStream=nullptr;
  if(failure)std::rethrow_exception(failure);
 }
 cudaError_t appendPoolBlock(std::size_t needed){
  std::size_t bytes=std::max(poolChunkBytes,(needed+255)&~std::size_t(255));void* data=nullptr;VcPoolBlock* descriptor=nullptr;
  auto error=cudaMallocFromPoolAsync(&data,bytes,backingPool,poolUploadStream);if(error!=cudaSuccess)return error;
  error=cudaMallocFromPoolAsync(reinterpret_cast<void**>(&descriptor),sizeof(VcPoolBlock),backingPool,poolUploadStream);if(error!=cudaSuccess){cudaFreeAsync(data,poolUploadStream);cudaStreamSynchronize(poolUploadStream);return error;}
  VcPoolBlock node{static_cast<unsigned char*>(data),bytes,0,nullptr};
  error=cudaMemcpyAsync(descriptor,&node,sizeof(node),cudaMemcpyHostToDevice,poolUploadStream);
  if(error==cudaSuccess)error=cudaStreamSynchronize(poolUploadStream);
  if(error!=cudaSuccess){cudaFreeAsync(descriptor,poolUploadStream);cudaFreeAsync(data,poolUploadStream);cudaStreamSynchronize(poolUploadStream);return error;}
  auto link=poolBlocks.empty()?reinterpret_cast<VcPoolBlock**>(reinterpret_cast<char*>(pool.ptr)+offsetof(VcPool,first)):reinterpret_cast<VcPoolBlock**>(reinterpret_cast<char*>(poolBlocks.back().descriptor)+offsetof(VcPoolBlock,next));
  error=cudaMemcpyAsync(link,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice,poolUploadStream);
  if(error==cudaSuccess)error=cudaStreamSynchronize(poolUploadStream);
  if(error!=cudaSuccess){cudaFreeAsync(descriptor,poolUploadStream);cudaFreeAsync(data,poolUploadStream);cudaStreamSynchronize(poolUploadStream);return error;}
  poolBlocks.push_back({data,descriptor});return cudaSuccess;
 }
 bool servicePending(){
  bool worked=false;for(auto& p:levels){auto& q=*p->request;if(q.state!=1)continue;worked=true;q.error=static_cast<int>(appendPoolBlock(q.poolBytes));q.state=2;}
  if(worked){checked(cudaMemsetAsync(pending.ptr,0,sizeof(int),poolUploadStream));checked(cudaStreamSynchronize(poolUploadStream));}
  return worked;
 }
 void clearApply(){if(applyExecutable)cudaGraphExecDestroy(applyExecutable);if(applyGraph)cudaGraphDestroy(applyGraph);applyExecutable=nullptr;applyGraph=nullptr;}
 void build(const std::vector<int>& off,const std::vector<int>& col,PinnedTransferQueue& transfers){
  release();checked(cudaStreamCreateWithFlags(&captureStream,cudaStreamNonBlocking));
  pending.reserve(1);progressing.reserve(1);checked(cudaMemset(pending.ptr,0,sizeof(int)));checked(cudaMemset(progressing.ptr,0,sizeof(int)));
  int n=int(off.size())-1;
  pool.reserve(1);checked(cudaMemset(pool.ptr,0,sizeof(VcPool)));checked(cudaStreamCreateWithFlags(&poolUploadStream,cudaStreamNonBlocking));
  cudaMemPoolProps properties{};properties.allocType=cudaMemAllocationTypePinned;properties.location.type=cudaMemLocationTypeDevice;checked(cudaGetDevice(&properties.location.id));checked(cudaMemPoolCreate(&backingPool,&properties));int reuse=0;
  checked(cudaMemPoolSetAttribute(backingPool,cudaMemPoolReuseAllowInternalDependencies,&reuse));checked(cudaMemPoolSetAttribute(backingPool,cudaMemPoolReuseFollowEventDependencies,&reuse));checked(cudaMemPoolSetAttribute(backingPool,cudaMemPoolReuseAllowOpportunistic,&reuse));
  maximumHashTasks=1;while(maximumHashTasks<2LL*col.size())maximumHashTasks*=2;

  VcStorage initial{},scratch{};VcCursor estimate{nullptr};vcStoragePlan<R>(initial,scratch,n,col.size(),std::size_t(n)*8,std::size_t(col.size())*2,true,estimate);
  poolChunkBytes=std::max(std::size_t(1u<<20),(estimate.used+255)&~std::size_t(255));
  checked(appendPoolBlock(poolChunkBytes));checked(appendPoolBlock(poolChunkBytes/2));

  savedValid.reserve(1);checked(cudaMemset(savedValid.ptr,0,sizeof(int)));activeDepth.reserve(1);
  for(;;){
   auto l=std::unique_ptr<Level>(new Level);l->n=n;l->nc=n;
   l->storage.reserve(1);checked(cudaMemset(l->storage.ptr,0,sizeof(VcStorage)));
   checked(cudaHostAlloc(reinterpret_cast<void**>(&l->request),sizeof(VcAllocationRequest),cudaHostAllocMapped));
   const auto freeUnownedRequest=[](VcAllocationRequest* request) noexcept {if(request)cudaFreeHost(request);};
   std::unique_ptr<VcAllocationRequest,decltype(freeUnownedRequest)> requestGuard(l->request,freeUnownedRequest);
   new(l->request) VcAllocationRequest;
   checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&l->deviceRequest),l->request,0));
   l->built.reserve(1);checked(cudaMemset(l->built.ptr,0,sizeof(int)));
   l->flags.reserve(22);checked(cudaMemset(l->flags.ptr,0,22*sizeof(int)));
   l->rows.reserve(1);checked(cudaMemcpy(l->rows.ptr,&n,sizeof(int),cudaMemcpyHostToDevice));
   l->weight.reserve(1);l->seed.reserve(1);l->temporaryGroups.reserve(1);
   l->off.bind(l->storage.ptr,vcRowOff);
   l->map.bind(l->storage.ptr,vcRowMap);
   l->count.bind(l->storage.ptr,vcRowCount);
   l->members.bind(l->storage.ptr,vcRowMembers);
   l->memberOffsets.bind(l->storage.ptr,vcRowMemberOffsets);
   l->inverseOffsets.bind(l->storage.ptr,vcRowInverseOffsets);
   l->paddedMembers.bind(l->storage.ptr,vcRowPaddedMembers);
   l->cursor.bind(l->storage.ptr,vcRowCursor);
   l->temporaryMap.bind(l->storage.ptr,vcRowTemporaryMap);
   l->temporaryCount.bind(l->storage.ptr,vcRowTemporaryCount);
   l->temporaryMembers.bind(l->storage.ptr,vcRowTemporaryMembers);
   l->choice.bind(l->storage.ptr,vcRowChoice);
   l->leader.bind(l->storage.ptr,vcRowLeader);
   l->rootFlag.bind(l->storage.ptr,vcRowRootFlag);
   l->compact.bind(l->storage.ptr,vcRowCompact);
   l->scanOutput.bind(l->storage.ptr,vcRowScanOutput);
   l->scanScratch.bind(l->storage.ptr,vcRowScanScratch);
   l->maxima.bind(l->storage.ptr,vcRowMaxima);
   l->inverse.bind(l->storage.ptr,vcRowInverse);
   l->bounds.bind(l->storage.ptr,vcRowBounds);
   l->smooth.bind(l->storage.ptr,vcRowSmooth);
   l->residual.bind(l->storage.ptr,vcRowResidual);
   l->r.bind(l->storage.ptr,vcRowR);
   l->z.bind(l->storage.ptr,vcRowZ);
   l->L.bind(l->storage.ptr,vcRowL);
   levels.push_back(std::move(l));requestGuard.release();if(n<=32)break;n=int((static_cast<long long>(n)*4-1)/5);
  }
  int last=int(levels.size())-1;checked(cudaMemcpy(activeDepth.ptr,&last,sizeof(int),cudaMemcpyHostToDevice));std::vector<int*> flagPointers,builtPointers;for(auto& p:levels){flagPointers.push_back(p->flags.ptr);builtPointers.push_back(p->built.ptr);}
  commitFlags.upload(flagPointers.data(),flagPointers.size(),transfers);commitBuilt.upload(builtPointers.data(),builtPointers.size(),transfers);
  transfers.finish();
 }
 MatrixOperator at(size_t k,MatrixOperator fine)const{
  if(k){const auto& l=*levels[k];fine={l.off.ptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};fine.storage=l.storage.ptr;fine.activeRows=levels[k-1]->seed.ptr;}
  else fine.activeRows=levels[0]->rows.ptr;
  fine.refreshFlags=levels[k]->flags.ptr;return fine;
 }
 template<class Build> cudaGraphNode_t withGridPublication(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,Build build){
  std::vector<KernelGridSelection> selections;
  auto previous=applyGridSelections;
  cudaGraphNode_t entry=nullptr;checked(cudaGraphAddEmptyNode(&entry,graph,dependencies.data(),dependencies.size()));
  applyGridSelections=&selections;
  try{
   const auto result=build(std::vector<cudaGraphNode_t>{entry});
   applyGridSelections=previous;
   std::vector<cudaGraphNode_t> before=dependencies;
   // One publication phase per solve/submission, before all V-cycle uses,
   // including batched CG iterations and true-residual restarts. Each handle
   // is independent; CUDA's per-launch defaults are overwritten every time.
   for(std::size_t first=0;first<selections.size();first+=16){
    KernelGridPublication publication;
    publication.count=int(std::min(std::size_t(16),selections.size()-first));
    for(int j=0;j<publication.count;++j)publication.selections[j]=selections[first+j];
    void* arguments[]={&publication};
    cudaKernelNodeParams parameters{};parameters.func=reinterpret_cast<void*>(publishKernelGrids);
    parameters.gridDim=dim3(1);parameters.blockDim=dim3(32);parameters.kernelParams=arguments;
    cudaGraphNode_t published=nullptr;
    checked(cudaGraphAddKernelNode(&published,graph,before.data(),before.size(),&parameters));
    before={published};
   }
   if(!selections.empty()){
#if CUDART_VERSION >= 13000
    checked(cudaGraphAddDependencies(graph,before.data(),&entry,nullptr,1));
#else
    checked(cudaGraphAddDependencies(graph,before.data(),&entry,1));
#endif
   }
   return result;
  }catch(...){applyGridSelections=previous;throw;}
 }
 template<class Kernel,class Launch> void dispatch(Kernel kernel,int blockThreads,std::size_t shared,long long bound,cudaStream_t stream,const int* count,int divisor,int extra,Launch launch){
  KernelGridBuckets buckets;checked(kernelGridBuckets(kernel,blockThreads,shared,bound+extra,divisor,buckets));
  if(buckets.count==1){launch(buckets.blocks[0],stream);return;}
  cudaStreamCaptureStatus state;cudaGraph_t graph=nullptr;const cudaGraphNode_t* dependencies=nullptr;size_t size=0;
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&size));
  cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
  if(applyGridSelections)applyGridSelections->push_back({count,divisor,extra,buckets,handle,applyGridValid,applyGridActive,applyGridMode});
  else selectKernelGrid<<<1,1,0,stream>>>(count,divisor,extra,buckets,handle);
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&size));
  cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeSwitch;params.conditional.size=buckets.count;
  cudaGraphNode_t node=nullptr;
#if CUDART_VERSION >= 13000
  checked(cudaGraphAddNode(&node,graph,dependencies,nullptr,size,&params));
#else
  checked(cudaGraphAddNode(&node,graph,dependencies,size,&params));
#endif
  for(int i=0;i<buckets.count;++i){cudaStream_t branch=borrowCaptureStream();sparseGraphDetail::appendCaptured(params.conditional.phGraph_out[i],{},branch,[&]{launch(buckets.blocks[i],branch);});}
  checked(cudaStreamUpdateCaptureDependencies(stream,&node,nullptr,1,cudaStreamSetCaptureDependencies));
 }
 template<class Body> void when(cudaStream_t stream,const int* flag,Body body){
  cudaStream_t branchStream=borrowCaptureStream();
  cudaStreamCaptureStatus state;cudaGraph_t graph=nullptr;const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&count));
  cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
  vcSelectFlag<<<1,1,0,stream>>>(flag,condition);
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&count));
  std::vector<cudaGraphNode_t> copied(dependencies,dependencies+count);
  cudaGraph_t child=nullptr;auto finished=executionGraphDetail::addIf(graph,condition,copied,child);
  sparseGraphDetail::appendCaptured(child,{},branchStream,[&]{body(branchStream);});
  checked(cudaStreamUpdateCaptureDependencies(stream,&finished,nullptr,1,cudaStreamSetCaptureDependencies));
 }
 static long long scanTaskBound(int n,int depth){long long count=n;for(int i=0;i<=depth;++i)count=(count+vcScanTile-1)/vcScanTile;return count;}
 void scanLevel(Level& l,VcPointer<const int> values,VcPointer<int> output,int square,int depth,cudaStream_t stream){
  dispatch(vcScanTiles,vcScanTile,0,scanTaskBound(l.n,depth),stream,reinterpret_cast<const int*>(reinterpret_cast<const char*>(l.storage.ptr)+offsetof(VcStorage,scanTiles))+depth,1,0,[&](int blocks,cudaStream_t workStream){vcScanTiles<<<blocks,vcScanTile,0,workStream>>>(l.storage.ptr,values,output,square,depth);});
  if(depth+1<vcScanStages&&scanTaskBound(l.n,depth)>1){
   const int* next=reinterpret_cast<const int*>(reinterpret_cast<const char*>(l.storage.ptr)+offsetof(VcStorage,scanPass))+depth+1;
   when(stream,next,[&](cudaStream_t branch){
    scanLevel(l,values,output,square,depth+1,branch);
    dispatch(vcScanAdd,vcScanTile,0,scanTaskBound(l.n,depth),branch,reinterpret_cast<const int*>(reinterpret_cast<const char*>(l.storage.ptr)+offsetof(VcStorage,scanTiles))+depth,1,0,[&](int blocks,cudaStream_t workStream){vcScanAdd<<<blocks,vcScanTile,0,workStream>>>(l.storage.ptr,output,depth);});
   });
  }
 }
 void scan(Level& l,VcPointer<const int> values,const int* count,const int* enabled,int square,VcPointer<int> output,cudaStream_t stream){
  vcScanSetup<<<1,1,0,stream>>>(l.storage.ptr,count,enabled);
  const int* first=reinterpret_cast<const int*>(reinterpret_cast<const char*>(l.storage.ptr)+offsetof(VcStorage,scanPass));
  when(stream,first,[&](cudaStream_t branch){scanLevel(l,values,output,square,0,branch);});
  vcScanTotal<<<1,1,0,stream>>>(l.storage.ptr,output,count,enabled);
 }
 template<class Body> void unless(cudaStream_t stream,const int* flag,Body body){
  cudaStream_t branchStream=borrowCaptureStream();cudaStreamCaptureStatus state;cudaGraph_t graph=nullptr;const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&count));
  cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
  vcSelectReady<<<1,1,0,stream>>>(flag,condition);checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&count));
  std::vector<cudaGraphNode_t> copied(dependencies,dependencies+count);cudaGraph_t child=nullptr;auto finished=executionGraphDetail::addIf(graph,condition,copied,child);
  sparseGraphDetail::appendCaptured(child,{},branchStream,[&]{body(branchStream);});checked(cudaStreamUpdateCaptureDependencies(stream,&finished,nullptr,1,cudaStreamSetCaptureDependencies));
 }
 template<class Body> void ready(cudaStream_t stream,Body body){unless(stream,pending.ptr,body);}
 template<class Body> void stage(Level& l,int phase,cudaStream_t stream,Body body){
  vcSelectStage<<<1,1,0,stream>>>(l.flags.ptr,phase,pending.ptr);
  when(stream,l.flags.ptr+21,[&](cudaStream_t branch){body(branch);vcFinishStage<<<1,1,0,branch>>>(l.flags.ptr,phase,pending.ptr);});
 }
 void ensureStorage(Level& l,const int* rows,VcPointer<const int> offsets,const int* gateFlags,int* valid,int stage,bool root,cudaStream_t stream){
  vcRequestStorage<R><<<1,1,0,stream>>>(rows,offsets,l.storage.ptr,l.deviceRequest,l.flags.ptr,gateFlags,valid,stage,root,pending.ptr,pool.ptr,&l==levels.front().get());
 }
 void refresh(MatrixOperator fine,int* valid,cudaStream_t stream,int B){
  ownerStream=stream;cudaStreamCaptureStatus state;checked(cudaStreamIsCapturing(stream,&state));
  if(state==cudaStreamCaptureStatusActive){refreshBody(fine,valid,stream,B);return;}
  refreshTask.wait();
  ExecutionGraphSignature signature;signature.add(fine);signature.add(valid);signature.add(B);
  if(!refreshExecutable||!(signature==refreshSignature)){
   clearRefresh();refreshSignature=signature;checked(cudaGraphCreate(&refreshGraph,0));
   cudaStream_t recording=stream?stream:borrowCaptureStream();
   sparseGraphDetail::appendCaptured(refreshGraph,{},recording,[&]{refreshBody(fine,valid,recording,B);});
   checked(cudaGraphInstantiateWithFlags(&refreshExecutable,refreshGraph,0));
  }
  refreshTask.start(stream,refreshExecutable,[this]{return servicePending();},[]{});
 }
 void refreshBody(MatrixOperator fine,int* valid,cudaStream_t stream,int B){
  vcBeginContinuation<<<1,1,0,stream>>>(int(levels.size()),commitFlags.ptr,progressing.ptr,pending.ptr);
  refreshLevel(0,fine,valid,stream,B);
  ready(stream,[&](cudaStream_t finalStream){
   vcCommitHierarchy<<<(levels.size()+B-1)/B,B,0,finalStream>>>(int(levels.size()),commitFlags.ptr,commitBuilt.ptr,activeDepth.ptr,valid);
   vcDispatchSave<<<1,1,0,finalStream>>>(valid,savedValid.ptr,levels[0]->flags.ptr);
   vcFinishContinuation<<<1,1,0,finalStream>>>(progressing.ptr);
  });
 }
 void refreshLevel(size_t k,MatrixOperator fine,int* valid,cudaStream_t stream,int B){
  auto& l=*levels[k];const auto matrix=at(k,fine);
  stage(l,0,stream,[&](cudaStream_t stream){
   vcLevelBegin<<<1,1,0,stream>>>(int(k),activeDepth.ptr,l.flags.ptr);
   ensureStorage(l,matrix.activeRows,matrix.offsets,l.flags.ptr,valid,0,k==0,stream);

  });
  stage(l,1,stream,[&](cudaStream_t stream){
   vcCacheBegin<<<1,1,0,stream>>>(l.flags.ptr,k?levels[k-1]->flags.ptr:nullptr);
   dispatch(vcCacheRows<R>,B,0,l.n,stream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){vcCacheRows<R><<<blocks,B,0,workStream>>>(l.n,matrix,l.storage.ptr,l.maxima.ptr,l.flags.ptr,k?valid:nullptr);});
   dispatch(vcCacheEdges<R>,B,0,l.n,stream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){vcCacheEdges<R><<<blocks,B,0,workStream>>>(l.n,matrix,l.maxima.ptr,l.storage.ptr,l.flags.ptr,k?valid:nullptr);});
   vcCacheObserve<<<1,1,0,stream>>>(l.flags.ptr,k?valid:nullptr);
   if(!k){vcDispatchBegin<<<1,1,0,stream>>>(l.flags.ptr,savedValid.ptr,valid);dispatch(classifyCoupling,B,0,l.n,stream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){classifyCoupling<<<blocks,B,0,workStream>>>(l.n,matrix,valid);});}
   vcPrepareGrouping<<<1,1,0,stream>>>(int(k),int(levels.size())-1,matrix.activeRows,activeDepth.ptr,l.flags.ptr,l.built.ptr,valid);
   when(stream,l.flags.ptr+13,[&](cudaStream_t groupStream){
   dispatch(matchingInitialize<R>,B,0,l.n,groupStream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){matchingInitialize<R><<<blocks,B,0,workStream>>>(matrix.activeRows,l.temporaryGroups.ptr,l.temporaryMap.ptr,l.temporaryCount.ptr,l.temporaryMembers.ptr,l.flags.ptr+13);});
   VcPointer<int> inputMap=l.temporaryMap.ptr;VcPointer<int> inputCount=l.temporaryCount.ptr;VcPointer<int> inputMembers=l.temporaryMembers.ptr;int* inputGroups=l.temporaryGroups.ptr;
   VcPointer<int> outputMap=l.map.ptr;VcPointer<int> outputCount=l.count.ptr;VcPointer<int> outputMembers=l.paddedMembers.ptr;int* outputGroups=l.seed.ptr;
   for(int target:{2,4,8}){
    dispatch(matchingChoose<R>,B,0,l.n,groupStream,inputGroups,B,0,[&](int blocks,cudaStream_t workStream){matchingChoose<R><<<blocks,B,0,workStream>>>(l.n,inputGroups,target,matrix,inputMap,inputCount,inputMembers,l.maxima.ptr,l.choice.ptr,l.flags.ptr+13);});
    dispatch(matchingLeaders,B,0,l.n,groupStream,inputGroups,B,0,[&](int blocks,cudaStream_t workStream){matchingLeaders<<<blocks,B,0,workStream>>>(l.n,inputGroups,l.choice.ptr,l.leader.ptr,l.rootFlag.ptr,l.flags.ptr+13);});
    scan(l,l.rootFlag.ptr,inputGroups,l.flags.ptr+13,0,l.compact.ptr,groupStream);
    matchingScanTotal<<<1,1,0,groupStream>>>(inputGroups,l.compact.ptr,outputGroups,l.flags.ptr+13);
    dispatch(matchingPublish,B,0,l.n,groupStream,inputGroups,B,0,[&](int blocks,cudaStream_t workStream){matchingPublish<<<blocks,B,0,workStream>>>(l.n,inputGroups,l.choice.ptr,l.leader.ptr,l.compact.ptr,inputCount,inputMembers,outputMap,outputCount,outputMembers,l.flags.ptr+13);});
    std::swap(inputMap,outputMap);std::swap(inputCount,outputCount);std::swap(inputMembers,outputMembers);std::swap(inputGroups,outputGroups);
   }
   scan(l,l.count.ptr,l.seed.ptr,l.flags.ptr+13,0,l.scanOutput.ptr,groupStream);
   dispatch(vcPublishScanOffsets,B,0,l.n,groupStream,l.seed.ptr,B,1,[&](int blocks,cudaStream_t workStream){vcPublishScanOffsets<<<blocks,B,0,workStream>>>(l.n,l.seed.ptr,l.scanOutput.ptr,l.memberOffsets.ptr,l.flags.ptr+13);});
   scan(l,l.count.ptr,l.seed.ptr,l.flags.ptr+13,1,l.scanOutput.ptr,groupStream);
   dispatch(vcPublishScanOffsets,B,0,l.n,groupStream,l.seed.ptr,B,1,[&](int blocks,cudaStream_t workStream){vcPublishScanOffsets<<<blocks,B,0,workStream>>>(l.n,l.seed.ptr,l.scanOutput.ptr,l.inverseOffsets.ptr,l.flags.ptr+13);});
   dispatch(vcPackMembers,B,0,l.n,groupStream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcPackMembers<<<blocks,B,0,workStream>>>(l.seed.ptr,l.count.ptr,l.paddedMembers.ptr,l.memberOffsets.ptr,l.members.ptr,l.flags.ptr+13);});
   });
   vcFinishGrouping<<<1,1,0,stream>>>(int(k),matrix.activeRows,l.seed.ptr,activeDepth.ptr,l.flags.ptr);

  });
  stage(l,2,stream,[&](cudaStream_t stream){
   ensureStorage(l,l.seed.ptr,l.inverseOffsets.ptr,l.flags.ptr,valid,3,false,stream);

  });
  stage(l,3,stream,[&](cudaStream_t stream){
   dispatch(vcLineFactor<R>,B,0,l.n,stream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcLineFactor<R><<<blocks,B,0,workStream>>>(l.seed.ptr,l.map.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,l.inverseOffsets.ptr,matrix,l.inverse.ptr,l.bounds.ptr,valid);});
   vcLineWeight<R><<<1,B,B*sizeof(R),stream>>>(l.seed.ptr,l.bounds.ptr,l.weight.ptr,valid,l.flags.ptr);
   vcLeafFactor<R><<<1,B,0,stream>>>(l.n,matrix,l.L.ptr,valid);

  });
  if(k+1<levels.size()){
   auto& next=*levels[k+1];
  stage(l,4,stream,[&](cudaStream_t stream){
ensureStorage(l,matrix.activeRows,matrix.offsets,l.flags.ptr,valid,1,false,stream);
  });
  stage(l,5,stream,[&](cudaStream_t stream){
vcHashExtent<<<1,1,0,stream>>>(matrix,l.flags.ptr,valid);
    dispatch(vcInitializeCoarse,B,0,maximumHashTasks,stream,l.flags.ptr+17,B,0,[&](int blocks,cudaStream_t workStream){vcInitializeCoarse<<<blocks,B,0,workStream>>>(l.hashCapacity,l.storage.ptr,l.seed.ptr,l.cursor.ptr,valid,l.flags.ptr);});
    dispatch(vcCoarseKeys,B,0,l.n,stream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){vcCoarseKeys<<<blocks,B,0,workStream>>>(l.n,l.seed.ptr,matrix,l.map.ptr,l.storage.ptr,l.hashCapacity-1,valid);});
    dispatch(vcCoarseCounts,B,0,maximumHashTasks,stream,l.flags.ptr+17,B,0,[&](int blocks,cudaStream_t workStream){vcCoarseCounts<<<blocks,B,0,workStream>>>(l.hashCapacity,l.seed.ptr,l.storage.ptr,l.cursor.ptr,valid,l.flags.ptr);});
    vcCoarseScanEnabled<<<1,1,0,stream>>>(l.flags.ptr,valid);
    when(stream,l.flags.ptr+18,[&](cudaStream_t scanStream){scan(l,l.cursor.ptr,l.seed.ptr,l.flags.ptr+18,0,l.scanOutput.ptr,scanStream);});

  });
  stage(l,6,stream,[&](cudaStream_t stream){
    ensureStorage(next,l.seed.ptr,l.scanOutput.ptr,l.flags.ptr,valid,2,false,stream);

  });
  stage(l,7,stream,[&](cudaStream_t stream){
    dispatch(vcCoarseOffsets,B,0,l.n,stream,l.seed.ptr,B,1,[&](int blocks,cudaStream_t workStream){vcCoarseOffsets<<<blocks,B,0,workStream>>>(l.n,l.seed.ptr,l.scanOutput.ptr,l.cursor.ptr,next.off.ptr,next.storage.ptr,valid,l.flags.ptr);});
    dispatch(vcCoarseValues<R>,B,0,maximumHashTasks,stream,l.flags.ptr+17,B,0,[&](int blocks,cudaStream_t workStream){vcCoarseValues<R><<<blocks,B,0,workStream>>>(l.hashCapacity,l.seed.ptr,l.storage.ptr,l.map.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,matrix,l.cursor.ptr,next.off.ptr,next.storage.ptr,valid);});

  });
   stage(l,8,stream,[&](cudaStream_t stream){
    vcNextLevel<<<1,1,0,stream>>>(l.flags.ptr,valid,next.flags.ptr);
    when(stream,next.flags.ptr+14,[&](cudaStream_t nextStream){refreshLevel(k+1,fine,valid,nextStream,B);});
   });
  }
 }
 cudaGraphNode_t appendLevel(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,size_t k,MatrixOperator fine,const R* r,R* z,const R* stop,const int* valid,int B){
  cudaStream_t branchStream=borrowCaptureStream();
  using executionGraphDetail::appendCaptured;using executionGraphDetail::addIf;
  auto& l=*levels[k];GridSelectionGuard selectionGuard(*this,valid,l.flags.ptr+14,l.flags.ptr+15);VcPointer<const R> rhs=k?VcPointer<const R>(l.r.ptr):VcPointer<const R>(r);VcPointer<R> result=k?l.z.ptr:VcPointer<R>(z);const auto matrix=at(k,fine);
  cudaGraphConditionalHandle normal,direct,terminal;
  checked(cudaGraphConditionalHandleCreate(&normal,graph,0,cudaGraphCondAssignDefault));
  checked(cudaGraphConditionalHandleCreate(&direct,graph,0,cudaGraphCondAssignDefault));
  checked(cudaGraphConditionalHandleCreate(&terminal,graph,0,cudaGraphCondAssignDefault));
  auto selected=appendCaptured(graph,dependencies,branchStream,[&]{vcSelectLevel<R><<<1,1,0,branchStream>>>(l.flags.ptr,stop,valid,normal,direct,terminal);});
  cudaGraph_t normalBody=nullptr,directBody=nullptr,terminalBody=nullptr;
  auto first=addIf(graph,normal,{selected},normalBody);
  auto second=addIf(graph,direct,{first},directBody);
  auto third=addIf(graph,terminal,{second},terminalBody);
  if(k+1<levels.size()){
   auto down=sparseGraphDetail::appendCaptured(normalBody,{},branchStream,[&]{
    dispatch(vcLineApply<R>,B,0,l.n,branchStream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcLineApply<R><<<blocks,B,0,workStream>>>(l.seed.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,l.inverseOffsets.ptr,l.inverse.ptr,l.weight.ptr,rhs,l.smooth.ptr,stop,valid,false,l.flags.ptr,0);});
    dispatch(vcDown<R>,B,0,l.n,branchStream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcDown<R><<<blocks,B,0,workStream>>>(l.seed.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,matrix,l.smooth.ptr,rhs,levels[k+1]->r.ptr,stop,valid);});
   });
   auto coarse=appendLevel(normalBody,{down},k+1,fine,r,z,stop,valid,B);
   sparseGraphDetail::appendCaptured(normalBody,{coarse},branchStream,[&]{
    dispatch(vcUp<R>,B,0,l.n,branchStream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){vcUp<R><<<blocks,B,0,workStream>>>(l.n,l.map.ptr,matrix,l.smooth.ptr,rhs,levels[k+1]->z.ptr,result,l.residual.ptr,stop,valid,k==0);});
    dispatch(vcLineApply<R>,B,0,l.n,branchStream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcLineApply<R><<<blocks,B,0,workStream>>>(l.seed.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,l.inverseOffsets.ptr,l.inverse.ptr,l.weight.ptr,l.residual.ptr,result,stop,valid,true,l.flags.ptr,0);});
   });
  }
  appendCaptured(directBody,{},branchStream,[&]{vcLeafSolve<R><<<1,B,0,branchStream>>>(l.n,matrix,l.L.ptr,rhs,result,stop,valid);});
  sparseGraphDetail::appendCaptured(terminalBody,{},branchStream,[&]{
   for(int step=0;step<4;++step){
    dispatch(vcTerminalResidual<R>,B,0,l.n,branchStream,matrix.activeRows,B,0,[&](int blocks,cudaStream_t workStream){vcTerminalResidual<R><<<blocks,B,0,workStream>>>(matrix,rhs,result,l.residual.ptr,stop,valid,step==0);});
    dispatch(vcLineApply<R>,B,0,l.n,branchStream,l.seed.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcLineApply<R><<<blocks,B,0,workStream>>>(l.seed.ptr,l.count.ptr,l.members.ptr,l.memberOffsets.ptr,l.inverseOffsets.ptr,l.inverse.ptr,l.weight.ptr,l.residual.ptr,result,stop,valid,true,l.flags.ptr,1);});
   }
  });
  return third;
 }
 cudaGraphNode_t appendApply(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,MatrixOperator fine,const R* r,R* z,const R* stop,const int* valid,int B){
  if(!applyGridSelections)return withGridPublication(graph,dependencies,[&](const std::vector<cudaGraphNode_t>& before){return appendApply(graph,before,fine,r,z,stop,valid,B);});
  cudaStream_t branchStream=borrowCaptureStream();
  using executionGraphDetail::appendCaptured;using executionGraphDetail::addIf;
  cudaGraphConditionalHandle fallback,hierarchy;
  checked(cudaGraphConditionalHandleCreate(&fallback,graph,0,cudaGraphCondAssignDefault));
  checked(cudaGraphConditionalHandleCreate(&hierarchy,graph,0,cudaGraphCondAssignDefault));
  auto selected=appendCaptured(graph,dependencies,branchStream,[&]{vcSelectApply<R><<<1,1,0,branchStream>>>(stop,valid,fallback,hierarchy);});
  cudaGraph_t fallbackBody=nullptr,hierarchyBody=nullptr;
  auto a=addIf(graph,fallback,{selected},fallbackBody);auto b=addIf(graph,hierarchy,{a},hierarchyBody);
  sparseGraphDetail::appendCaptured(fallbackBody,{},branchStream,[&]{dispatch(vcApplyFallback<R>,B,0,levels[0]->n,branchStream,levels[0]->rows.ptr,B,0,[&](int blocks,cudaStream_t workStream){vcApplyFallback<R><<<blocks,B,0,workStream>>>(levels[0]->n,fine,r,z,stop,valid);});});
  appendLevel(hierarchyBody,{},0,fine,r,z,stop,valid,B);return b;
 }
 void apply(MatrixOperator fine,const R* r,R* z,const R* stop,const int* valid,cudaStream_t stream,int B){
  cudaStreamCaptureStatus state;cudaGraph_t graph=nullptr;const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
  checked(cudaStreamGetCaptureInfo(stream,&state,nullptr,&graph,&dependencies,nullptr,&count));
  if(state==cudaStreamCaptureStatusActive){
   std::vector<cudaGraphNode_t> copied(dependencies,dependencies+count);
   auto node=appendApply(graph,copied,fine,r,z,stop,valid,B);
   checked(cudaStreamUpdateCaptureDependencies(stream,&node,nullptr,1,cudaStreamSetCaptureDependencies));return;
  }
  refreshTask.wait();
  ExecutionGraphSignature signature;signature.add(fine);signature.add(r);signature.add(z);signature.add(stop);signature.add(valid);signature.add(B);
  if(!applyExecutable||!(signature==applySignature)){
   clearApply();applySignature=signature;checked(cudaGraphCreate(&applyGraph,0));appendApply(applyGraph,{},fine,r,z,stop,valid,B);
   checked(cudaGraphInstantiateWithFlags(&applyExecutable,applyGraph,0));
  }
  checked(cudaGraphLaunch(applyExecutable,stream));
 }

};

__global__ void multiply(int n,MatrixOperator matrix,const GpuReal* x,GpuReal* y,const GpuReal* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||(stop&&stop[7]))return;
    GpuReal value=matrix.product(i,x);
    y[i]=value;
}
__global__ void prepareLduRows(int n,const int* offsets,const int* entrySource,const int* entryKind,
    const int* boundaryOffsets,const int* boundaryIndices,const int* boundaryNeighbour,
    const GpuReal* diagonal,const GpuReal* upper,const GpuReal* lower,const GpuReal* source,
    const GpuReal* boundaryDiagonal,const GpuReal* boundarySource,GpuReal* completedDiagonal,GpuReal* rhs,GpuReal* rowSum,
    const GpuReal* boundaryJump=nullptr,GpuReal* affine=nullptr,SparsePhysicalDiffusionValues diffusion={}){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    GpuReal d=diffusion.active?physicalDiffusionDetail::diagonal(diffusion,i):diagonal[i],b=source[i],jump=0;
    for(int j=boundaryOffsets[i];j<boundaryOffsets[i+1];++j){
        const int f=boundaryIndices[j];
        const float bd=diffusion.active?physicalDiffusionDetail::boundaryDiagonal(diffusion,f):boundaryDiagonal[f];
        const float bs=diffusion.active?physicalDiffusionDetail::boundarySource(diffusion,f):boundarySource[f];
        d+=bd;
        if(boundaryNeighbour[f]==-1)b+=bs;
        if(boundaryJump)jump+=bs*boundaryJump[f];
    }
    completedDiagonal[i]=d;rhs[i]=b;GpuReal sum=d;
    for(int j=offsets[i]+1;j<offsets[i+1];++j){
        const int f=entrySource[j],kind=entryKind[j];
        const GpuReal value=diffusion.active?(kind==3?-physicalDiffusionDetail::boundarySource(diffusion,f):physicalDiffusionDetail::internal(diffusion,f)):(kind==1?lower[f]:kind==2?upper[f]:-boundarySource[f]);
        if(kind!=3)sum+=value;
    }
    for(int j=boundaryOffsets[i];j<boundaryOffsets[i+1];++j){
        const int f=boundaryIndices[j];if(boundaryNeighbour[f]!=-1)sum-=diffusion.active?physicalDiffusionDetail::boundarySource(diffusion,f):boundarySource[f];
    }
    rowSum[i]=sum;if(affine)affine[i]=jump;
}



__global__ void addInitialAffine(int n,const GpuReal* affine,GpuReal* product){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)product[i]+=affine[i];
}
__global__ void initialResidual(int n,const GpuReal* source,const GpuReal* ax,GpuReal* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=source[i]-ax[i];
}
__global__ void precondition(int n,MatrixOperator matrix,const GpuReal* r,GpuReal* z,const GpuReal* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!(stop&&stop[7]))z[i]=r[i]/matrix.diagonal(i);
}


__global__ void exportPreconditioner(int n,MatrixOperator matrix,const GpuReal* z,GpuReal* result){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)result[i]=matrix.diagonal(i)==GpuReal(0)?GpuReal(0):z[i];
}

__global__ void direction(int n,const GpuReal* z,const GpuReal* recurrence,GpuReal* p){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[7])p[i]=(recurrence[6]==0||recurrence[9]!=0)?z[i]:z[i]+(recurrence[0]/recurrence[2])*p[i];
}
struct LocalSolveParameters {GpuReal norm,initial,mean;ConjugateGradientControls controls;};
__device__ bool confirmTrueResidual(LocalSolveParameters* parameters,GpuReal* recurrence){
    const auto& c=parameters->controls;
    const GpuReal residual=recurrence[4]/parameters->norm;
    const bool converged=residual<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&residual<c.relativeTolerance*parameters->initial);
    const bool exhausted=recurrence[6]>=c.maximumIterations&&recurrence[6]>=c.minimumIterations;


    recurrence[11]=!converged&&!exhausted&&!recurrence[5]&&recurrence[8]==0;
    recurrence[7]=converged||exhausted||recurrence[5]||recurrence[11];
    if(!recurrence[7]){recurrence[9]=1;recurrence[10]+=1;}
    return recurrence[7]!=0;
}
__global__ void setConfirmationCondition(const GpuReal* recurrence,cudaGraphConditionalHandle handle){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(handle,recurrence[7]!=0);
}
__global__ void completeTrueResidualConfirmation(LocalSolveParameters* parameters,GpuReal* recurrence,
    cudaGraphConditionalHandle loop){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(loop,confirmTrueResidual(parameters,recurrence)?0:1);
}
__global__ void initializeLocalParameters(int count,ConjugateGradientControls controls,
    LocalSolveParameters* parameters,GpuReal* recurrence,const ConjugateGradientControls* deviceControls){
    if(blockIdx.x||threadIdx.x)return;
    *parameters={0,0,0,deviceControls?*deviceControls:controls};
    for(int i=0;i<12;++i)recurrence[i]=0;
}
__global__ void normalizeLocalMean(int count,GpuReal* mean){
    if(!blockIdx.x&&!threadIdx.x)*mean/=count?count:1;
}
__global__ void initializeLocalCondition(LocalSolveParameters* parameters,GpuReal* recurrence,
    cudaGraphConditionalHandle handle){
    if(blockIdx.x||threadIdx.x)return;
    parameters->norm+=parameters->controls.normalizationSmall;
    parameters->initial=recurrence[4]/parameters->norm;
    const auto& c=parameters->controls;
    const bool converged=parameters->initial<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&parameters->initial<c.relativeTolerance*parameters->initial);
    recurrence[7]=c.minimumIterations<=0&&converged;
    cudaGraphSetConditional(handle,recurrence[7]?0:1);
}
__global__ void publishLocalReport(const LocalSolveParameters* parameters,const GpuReal* recurrence,
    ConjugateGradientReport* report,const int* blockValid=nullptr,int hierarchyCount=0){
    if(blockIdx.x||threadIdx.x)return;
    const auto& c=parameters->controls;const GpuReal residual=recurrence[4]/parameters->norm;
    report->preconditionerFallbackBlocks=0;
    for(int b=0;b<hierarchyCount;++b)report->preconditionerFallbackBlocks+=blockValid[b]==0;
    report->iterations=int(recurrence[6]);report->initialResidual=parameters->initial;report->finalResidual=residual;
    report->singular=recurrence[5]!=0;
    report->converged=residual<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&residual<c.relativeTolerance*parameters->initial);
    report->residualReplacements=int(recurrence[10]);
    report->stagnated=recurrence[11]!=0;
    report->budgetExhausted=!report->converged&&report->iterations>=c.maximumIterations;
    report->targetResidual=c.tolerance;
    if(c.relativeTolerance>c.normalizationSmall)
        report->targetResidual=fmaxf(report->targetResidual,c.relativeTolerance*parameters->initial);
}
__global__ void prepareStep(GpuReal* recurrence,GpuReal norm,GpuReal singularSmall,const LocalSolveParameters* parameters=nullptr){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
    if(parameters){norm=parameters->norm;singularSmall=parameters->controls.singularSmall;}
    recurrence[5]=fabsf(recurrence[1])/norm<singularSmall;
    if(!recurrence[5]){recurrence[3]=recurrence[0]/recurrence[1];recurrence[6]+=1;}
    recurrence[2]=recurrence[0];recurrence[9]=0;
}
__global__ void update(int n,const GpuReal* recurrence,const GpuReal* p,const GpuReal* ap,GpuReal* x,GpuReal* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[5]&&!recurrence[7]){x[i]+=recurrence[3]*p[i];r[i]-=recurrence[3]*ap[i];}
}


__global__ void partialReduction(int n,int mode,const GpuReal* a,const GpuReal* b,
    const GpuReal* rowSum,GpuReal mean,GpuReal* partial,const GpuReal* stop,const GpuReal* residentMean=nullptr){
    if(residentMean)mean=*residentMean;
    if(stop&&stop[7])return;
    extern __shared__ GpuReal values[];GpuReal v=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        if(mode==0)v+=a[i];
        else if(mode==1)v+=fabsf(a[i]);
        else if(mode==2)v+=a[i]*b[i];
        else if(mode==4)v+=a[i]!=b[i]?1:0;
        else {const GpuReal reference=rowSum[i]*mean;v+=fabsf(a[i]-reference)+fabsf(b[i]-reference);}
    }
    values[threadIdx.x]=v;__syncthreads();
    for(int active=threads;active>1;active=(active+1)/2){if(threadIdx.x<active/2)values[threadIdx.x]+=values[threadIdx.x+(active+1)/2];__syncthreads();}
    if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}
__global__ void finishReduction(int n,const GpuReal* partial,GpuReal* result,const GpuReal* stop){
    if(stop&&stop[7])return;
    extern __shared__ GpuReal values[];GpuReal v=0;
    for(int i=threadIdx.x;i<n;i+=threads)v+=partial[i];
    values[threadIdx.x]=v;__syncthreads();
    for(int active=threads;active>1;active=(active+1)/2){if(threadIdx.x<active/2)values[threadIdx.x]+=values[threadIdx.x+(active+1)/2];__syncthreads();}
    if(threadIdx.x==0)*result=values[0];
}
__global__ void gatherInterface(int count,const int* cells,const GpuReal* x,const GpuReal* y,GpuReal* packed){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){packed[i]=x[cells[i]];packed[count+i]=y[cells[i]];}
}
__global__ void scatterInterface(int count,const int* cells,const GpuReal* packed,GpuReal* y){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[cells[i]]=packed[i];
}


__device__ GpuReal localBlockSum(GpuReal value,GpuReal* scratch){
    scratch[threadIdx.x]=value;__syncthreads();
    for(int active=threads;active>1;active=(active+1)/2){
        if(threadIdx.x<active/2)scratch[threadIdx.x]+=scratch[threadIdx.x+(active+1)/2];
        __syncthreads();
    }
    return scratch[0];
}
__global__ void preconditionAndDot(int n,MatrixOperator matrix,
    const GpuReal* r,GpuReal* z,GpuReal* partial,const GpuReal* recurrence){
    if(recurrence[7])return;
    extern __shared__ GpuReal scratch[];GpuReal sum=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        const GpuReal value=r[i]/matrix.diagonal(i);z[i]=value;sum+=value*r[i];
    }
    sum=localBlockSum(sum,scratch);if(!threadIdx.x)partial[blockIdx.x]=sum;
}
__global__ void multiplyAndDot(int n,MatrixOperator matrix,
    const GpuReal* p,GpuReal* ap,GpuReal* partial,const GpuReal* recurrence){
    if(recurrence[7])return;
    extern __shared__ GpuReal scratch[];GpuReal sum=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        GpuReal value=matrix.product(i,p);
        ap[i]=value;sum+=value*p[i];
    }
    sum=localBlockSum(sum,scratch);if(!threadIdx.x)partial[blockIdx.x]=sum;
}
__global__ void updatePreconditionAndNorm(int n,MatrixOperator matrix,
    const GpuReal* recurrence,const GpuReal* p,const GpuReal* ap,GpuReal* x,GpuReal* r,GpuReal* z,GpuReal* partial){
    if(recurrence[7])return;
    extern __shared__ GpuReal scratch[];GpuReal norm=0,rho=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        GpuReal value=r[i];
        if(!recurrence[5]){x[i]+=recurrence[3]*p[i];value-=recurrence[3]*ap[i];r[i]=value;}
        const GpuReal preconditioned=value/matrix.diagonal(i);z[i]=preconditioned;
        norm+=fabsf(value);rho+=preconditioned*value;
    }
    norm=localBlockSum(norm,scratch);if(!threadIdx.x)partial[blockIdx.x]=norm;
    __syncthreads();
    rho=localBlockSum(rho,scratch);if(!threadIdx.x)partial[gridDim.x+blockIdx.x]=rho;
}
__global__ void blockNormAndRho(int n,MatrixOperator matrix,
    const GpuReal* recurrence,const GpuReal* p,const GpuReal* ap,GpuReal* x,GpuReal* r,GpuReal* z,GpuReal* partial){
    if(recurrence[7])return;
    extern __shared__ GpuReal scratch[];GpuReal norm=0,rho=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        GpuReal value=r[i];

        const GpuReal preconditioned=z[i];
        norm+=fabsf(value);rho+=preconditioned*value;
    }
    norm=localBlockSum(norm,scratch);if(!threadIdx.x)partial[blockIdx.x]=norm;
    __syncthreads();
    rho=localBlockSum(rho,scratch);if(!threadIdx.x)partial[gridDim.x+blockIdx.x]=rho;
}

template<int Stage> __global__ void finishLocalReduction(int n,const GpuReal* partial,
    GpuReal* recurrence,const LocalSolveParameters* parameters,cudaGraphConditionalHandle handle){
    if(recurrence[7])return;
    extern __shared__ GpuReal scratch[];GpuReal sum=0;
    for(int i=threadIdx.x;i<n;i+=threads)sum+=partial[i];
    sum=localBlockSum(sum,scratch);
    GpuReal nextRho=0;
    if constexpr(Stage==2){
        for(int i=threadIdx.x;i<n;i+=threads)nextRho+=partial[n+i];
        __syncthreads();
        nextRho=localBlockSum(nextRho,scratch);
    }
    if(threadIdx.x)return;
    if constexpr(Stage==0)recurrence[0]=sum;
    else if constexpr(Stage==1){
        recurrence[1]=sum;
        recurrence[5]=fabsf(sum)/parameters->norm<parameters->controls.singularSmall;
        if(!recurrence[5]){recurrence[3]=recurrence[0]/sum;recurrence[6]+=1;}
        recurrence[2]=recurrence[0];recurrence[9]=0;
    }else{
        recurrence[4]=sum;recurrence[0]=nextRho;
        const auto& controls=parameters->controls;
        const GpuReal residual=sum/parameters->norm;
        const bool converged=residual<controls.tolerance ||
            (controls.relativeTolerance>controls.normalizationSmall&&residual<controls.relativeTolerance*parameters->initial);
        recurrence[7]=recurrence[5] ||
            ((recurrence[6]>=controls.maximumIterations||converged)&&recurrence[6]>=controls.minimumIterations);
        cudaGraphSetConditional(handle,recurrence[7]?0:1);
    }
}





__global__ void solveLocalBlock(int n,MatrixOperator matrix,
    const GpuReal* rowSum,const GpuReal* b,const GpuReal* affine,GpuReal* x,GpuReal* r,GpuReal* z,GpuReal* p,GpuReal* ap,GpuReal* confirmedX,
    ConjugateGradientControls settings,const ConjugateGradientControls* deviceSettings,
    LocalSolveParameters* parameters,GpuReal* recurrence,ConjugateGradientReport* report){
    extern __shared__ GpuReal scratch[];const int i=threadIdx.x;
    if(!i){*parameters={0,0,0,deviceSettings?*deviceSettings:settings};for(int j=0;j<12;++j)recurrence[j]=0;}
    __syncthreads();
    if(i<n)confirmedX[i]=x[i];
    GpuReal sum=localBlockSum(i<n?x[i]:0.f,scratch);
    if(!i)parameters->mean=sum/(n?n:1);
    __syncthreads();
    GpuReal product=0,normTerm=0,residual=0;
    if(i<n){
        product=matrix.product(i,x);
        if(affine)product+=affine[i];
        ap[i]=product;residual=b[i]-product;r[i]=residual;
        const GpuReal reference=rowSum[i]*parameters->mean;
        normTerm=fabsf(product-reference)+fabsf(b[i]-reference);
    }
    sum=localBlockSum(normTerm,scratch);
    if(!i)parameters->norm=sum+parameters->controls.normalizationSmall;
    __syncthreads();
    sum=localBlockSum(fabsf(residual),scratch);
    if(!i){
        recurrence[4]=sum;parameters->initial=sum/parameters->norm;
        const auto& c=parameters->controls;
        const bool converged=parameters->initial<c.tolerance ||
            (c.relativeTolerance>c.normalizationSmall&&parameters->initial<c.relativeTolerance*parameters->initial);
        recurrence[7]=c.minimumIterations<=0&&converged;
    }
    __syncthreads();
    while(!recurrence[7]){
        GpuReal preconditioned=0;if(i<n){preconditioned=r[i]/matrix.diagonal(i);z[i]=preconditioned;}
        sum=localBlockSum(i<n?preconditioned*r[i]:0.f,scratch);
        if(!i)recurrence[0]=sum;
        __syncthreads();
        if(i<n)p[i]=(recurrence[6]==0||recurrence[9]!=0)?preconditioned:preconditioned+(recurrence[0]/recurrence[2])*p[i];
        __syncthreads();
        product=0;
        if(i<n){
            product=matrix.product(i,p);
            ap[i]=product;
        }
        sum=localBlockSum(i<n?product*p[i]:0.f,scratch);
        if(!i){
            recurrence[1]=sum;recurrence[5]=fabsf(sum)/parameters->norm<parameters->controls.singularSmall;
            if(!recurrence[5]){recurrence[3]=recurrence[0]/sum;recurrence[6]+=1;}
            recurrence[2]=recurrence[0];recurrence[9]=0;
        }
        __syncthreads();
        if(i<n&&!recurrence[5]){x[i]+=recurrence[3]*p[i];r[i]-=recurrence[3]*ap[i];}
        sum=localBlockSum(i<n?fabsf(r[i]):0.f,scratch);
        if(!i){
            recurrence[4]=sum;const auto& c=parameters->controls;const GpuReal value=sum/parameters->norm;
            const bool converged=value<c.tolerance||(c.relativeTolerance>c.normalizationSmall&&value<c.relativeTolerance*parameters->initial);
            recurrence[7]=recurrence[5]||((recurrence[6]>=c.maximumIterations||converged)&&recurrence[6]>=c.minimumIterations);
        }
        __syncthreads();
        if(recurrence[7]){
            GpuReal changed=0;product=0;
            if(i<n){
                changed=x[i]!=confirmedX[i]?1:0;confirmedX[i]=x[i];
                product=matrix.product(i,x);
                if(affine)product+=affine[i];
                r[i]=b[i]-product;
            }
            sum=localBlockSum(changed,scratch);if(!i)recurrence[8]=sum;
            __syncthreads();
            sum=localBlockSum(i<n?fabsf(r[i]):GpuReal(0),scratch);
            if(!i){recurrence[4]=sum;confirmTrueResidual(parameters,recurrence);}
            __syncthreads();
        }
    }
    if(!i){
        const auto& c=parameters->controls;const GpuReal value=recurrence[4]/parameters->norm;
        report->preconditionerFallbackBlocks=0;
        report->iterations=int(recurrence[6]);report->initialResidual=parameters->initial;report->finalResidual=value;
        report->singular=recurrence[5]!=0;report->converged=value<c.tolerance||
            (c.relativeTolerance>c.normalizationSmall&&value<c.relativeTolerance*parameters->initial);
    report->residualReplacements=int(recurrence[10]);
    report->stagnated=recurrence[11]!=0;
    report->budgetExhausted=!report->converged&&report->iterations>=c.maximumIterations;
    report->targetResidual=c.tolerance;
    if(c.relativeTolerance>c.normalizationSmall)
        report->targetResidual=fmaxf(report->targetResidual,c.relativeTolerance*parameters->initial);
    }
}

}
struct SparseConjugateGradient::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    std::unique_ptr<SparseConjugateGradient> residualEvaluator;
    GpuReal* hostRecurrence=nullptr;
    DeviceArray<int> offsets,columns,entrySource,entryKind,boundaryOffsets,boundaryIndices,boundaryNeighbour;
    DeviceArray<GpuReal> diagonal,upper,lower,source,boundaryDiagonal,boundarySource,boundaryJump,affine;
    std::vector<int> cachedOwner,cachedNeighbour,cachedBoundaryOwner,cachedBoundaryNeighbour;
    int cachedRows=-1,cachedEntries=0;
    std::vector<double> cachedBoundaryJump;
    SparseDeviceLduValues residentValues{},operatorValues{};
    bool lduOperator=false;
    MatrixOperator matrix()const{
        return {offsets.ptr,columns.ptr,values.ptr,lduOperator?completedDiagonal.ptr:nullptr,
            entrySource.ptr,entryKind.ptr,operatorValues.upper,operatorValues.lower,operatorValues.boundarySource,lduOperator?operatorValues.diffusion:SparsePhysicalDiffusionValues{}};
    }
    int residentRows=0;bool residentConfigured=false,residentAffine=false;
    DeviceArray<GpuReal> completedDiagonal,values,rowSum,x,b,r,z,p,ap,confirmedX,partial,result,recurrence;
    std::vector<GpuReal> interfaceX,interfaceResult,interfacePacked;
    std::vector<int> cachedInterfaceCells;
    DeviceArray<int> interfaceCells;
    DeviceArray<GpuReal> interfaceDevice;
    void configureInterface(int n,const SparseCommunication& comm){
        if(!comm.addInterface||comm.interfaceCellCount<0)return;
        const int count=comm.interfaceCellCount;
        if(count&&!comm.interfaceCells)throw std::invalid_argument("missing compact interface cells");
        std::vector<int> cells;if(count)cells.assign(comm.interfaceCells,comm.interfaceCells+count);
        std::sort(cells.begin(),cells.end());
        if((count&&(cells.front()<0||cells.back()>=n))||std::adjacent_find(cells.begin(),cells.end())!=cells.end())
            throw std::invalid_argument("invalid compact interface cells");
        if(cells!=cachedInterfaceCells){interfaceCells.upload(cells.data(),count,transfers);cachedInterfaceCells=std::move(cells);}
        interfacePacked.resize(2*count);interfaceDevice.reserve(2*count);
        interfaceX.resize(n);interfaceResult.resize(n);
    }
    DeviceArray<LocalSolveParameters> localParameters;
    DeviceArray<ConjugateGradientReport> localReport;
    ConjugateGradientControls graphControls{};
    struct Graph {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);executable=nullptr;graph=nullptr;}
        ~Graph(){clear();}
    } localGraph,preparedGraph;
    GraphContinuationTask preparedTask;
    unsigned long long localRevision=0,preparedRevision=~0ull;
    bool preparedResident=false;void* preparedSolution=nullptr;
    int graphRows=-1,graphBlockThreads=-1,graphDiffusionInternal=-1;bool graphDiffusionTransient=false,graphDiffusionActive=false;std::vector<const void*> graphAddresses;
    cudaEvent_t externalReady=nullptr,externalDone=nullptr;
    void waitFor(void* producer){
        if(!externalReady)checked(cudaEventCreateWithFlags(&externalReady,cudaEventDisableTiming));
        checked(cudaEventRecord(externalReady,static_cast<cudaStream_t>(producer)));
        checked(cudaStreamWaitEvent(stream,externalReady,0));
    }
    void publishTo(void* consumer){
        if(!externalDone)checked(cudaEventCreateWithFlags(&externalDone,cudaEventDisableTiming));
        checked(cudaEventRecord(externalDone,stream));
        checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer),externalDone,0));
    }
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);const auto status=cudaMallocHost(reinterpret_cast<void**>(&hostRecurrence),12*sizeof(GpuReal));if(status!=cudaSuccess){cudaStreamDestroy(stream);stream=nullptr;checked(status);}}
    ~Implementation(){preparedTask.reset();transfers.discard();if(stream){cudaStreamSynchronize(stream);preparedGraph.clear();localGraph.clear();try{hierarchy.release();}catch(...){}cudaStreamDestroy(stream);}if(hostRecurrence)cudaFreeHost(hostRecurrence);if(externalReady)cudaEventDestroy(externalReady);if(externalDone)cudaEventDestroy(externalDone);}
    void reduce(int n,int mode,const GpuReal* a,const GpuReal* other,GpuReal* output,
        const SparseCommunication& comm,GpuReal mean=0,const GpuReal* stop=nullptr,const GpuReal* residentMean=nullptr){
        const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));partial.reserve(blocks);
        partialReduction<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,mode,a,other,rowSum.ptr,mean,partial.ptr,stop,residentMean);
        finishReduction<<<1,threads,threads*sizeof(GpuReal),stream>>>(blocks,partial.ptr,output,stop);
        if(comm.globalSum){
            GpuReal host=0;transfers.download(&host,output,sizeof(GpuReal));
            transfers.finish();comm.globalSum(comm.context,&host,1);
            transfers.upload(output,&host,sizeof(GpuReal));
        }
    }
    GpuReal sum(int n,int mode,const GpuReal* a,const GpuReal* other,const SparseCommunication& comm,GpuReal mean=0){
        result.reserve(1);reduce(n,mode,a,other,result.ptr,comm,mean);
        GpuReal host=0;transfers.download(&host,result.ptr,sizeof(GpuReal));transfers.finish();return host;
    }
    void matvec(int n,const GpuReal* input,GpuReal* output,const SparseCommunication& comm,const GpuReal* stop=nullptr){
        if(n)multiply<<<(n+threads-1)/threads,threads,0,stream>>>(n,matrix(),input,output,stop);
        if(comm.addInterface){
            if(comm.interfaceCellCount>=0){
                const int count=int(cachedInterfaceCells.size());
                if(count){gatherInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,input,output,interfaceDevice.ptr);
                    transfers.download(interfacePacked.data(),interfaceDevice.ptr,2*count*sizeof(GpuReal));}
                transfers.finish();
                for(int i=0;i<count;++i){const int c=cachedInterfaceCells[i];interfaceX[c]=interfacePacked[i];interfaceResult[c]=interfacePacked[count+i];}
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                for(int i=0;i<count;++i)interfacePacked[i]=interfaceResult[cachedInterfaceCells[i]];
                if(count){transfers.upload(interfaceDevice.ptr,interfacePacked.data(),count*sizeof(GpuReal));
                    scatterInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,interfaceDevice.ptr,output);}
            }else{
                interfaceX.resize(n);interfaceResult.resize(n);
                if(n){transfers.download(interfaceX.data(),input,n*sizeof(GpuReal));
                    transfers.download(interfaceResult.data(),output,n*sizeof(GpuReal));}
                transfers.finish();
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                if(n)transfers.upload(output,interfaceResult.data(),n*sizeof(GpuReal));
            }
        }
    }
    VCycle<GpuReal> hierarchy;
    DeviceArray<int> hierarchyValid;
    int hierarchyCount=0;
    unsigned long long hierarchyTopologyRevision=0;
    std::vector<int> hierarchyOffsetsCache,hierarchyColumnsCache;
    bool usingMultigrid(int n)const{return n>threads;}
    void refreshHierarchy(int n){
        if(usingMultigrid(n))hierarchy.refresh(matrix(),hierarchyValid.ptr,stream,threads);
    }
    void buildHierarchy(int n,const std::vector<int>& off,const std::vector<int>& col){
        if(hierarchyCount&&off==hierarchyOffsetsCache&&col==hierarchyColumnsCache)return;
        localGraph.clear();graphRows=-1;++hierarchyTopologyRevision;
        hierarchyOffsetsCache=off;hierarchyColumnsCache=col;
        hierarchy.build(off,col,transfers);hierarchyCount=1;hierarchyValid.reserve(1);
    }
    void applyVcycle(const GpuReal* stop){
        hierarchy.apply(matrix(),r.ptr,z.ptr,stop,hierarchyValid.ptr,stream,threads);

    }

    void iteration(int n,GpuReal norm,GpuReal singularSmall,const SparseCommunication& comm,bool local=false){
        const GpuReal* stop=local?recurrence.ptr:nullptr;
        const int blocks=(n+threads-1)/threads;
        if(usingMultigrid(n))applyVcycle(stop);else if(n)precondition<<<blocks,threads,0,stream>>>(n,matrix(),r.ptr,z.ptr,stop);
        reduce(n,2,z.ptr,r.ptr,recurrence.ptr,comm,0,stop);
        if(n)direction<<<blocks,threads,0,stream>>>(n,z.ptr,recurrence.ptr,p.ptr);
        matvec(n,p.ptr,ap.ptr,comm,stop);
        reduce(n,2,ap.ptr,p.ptr,recurrence.ptr+1,comm,0,stop);
        prepareStep<<<1,1,0,stream>>>(recurrence.ptr,norm,singularSmall,local?localParameters.ptr:nullptr);
        if(n)update<<<blocks,threads,0,stream>>>(n,recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr);
        reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,comm,0,stop);
    }



    cudaGraphNode_t appendLocal(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,
        int n,const ConjugateGradientControls& controls,const float* initialAffine,const ConjugateGradientControls* deviceControls=nullptr){
        if(n<=threads)return appendLocalBody(graph,dependencies,n,controls,initialAffine,deviceControls,false);
        cudaGraphConditionalHandle diagonal,multigrid;
        checked(cudaGraphConditionalHandleCreate(&diagonal,graph,0,cudaGraphCondAssignDefault));
        checked(cudaGraphConditionalHandleCreate(&multigrid,graph,0,cudaGraphCondAssignDefault));
        auto selected=sparseGraphDetail::appendCaptured(graph,dependencies,stream,[&]{
            selectDispatch<<<1,1,0,stream>>>(hierarchyValid.ptr,diagonal,multigrid);
        });
        cudaGraph_t diagonalBody=nullptr,multigridBody=nullptr;
        auto a=executionGraphDetail::addIf(graph,diagonal,{selected},diagonalBody);
        auto b=executionGraphDetail::addIf(graph,multigrid,{a},multigridBody);
        appendLocalBody(diagonalBody,{},n,controls,initialAffine,deviceControls,false);
        appendLocalBody(multigridBody,{},n,controls,initialAffine,deviceControls,true);
        return b;
    }
    cudaGraphNode_t appendLocalBody(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,
        int n,const ConjugateGradientControls& controls,const GpuReal* initialAffine,const ConjugateGradientControls* deviceControls,bool multigrid){
        using sparseGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
        if(n<=threads)return appendCaptured(graph,dependencies,stream,[&]{
            solveLocalBlock<<<1,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),rowSum.ptr,b.ptr,
                initialAffine,x.ptr,r.ptr,z.ptr,p.ptr,ap.ptr,confirmedX.ptr,controls,deviceControls,localParameters.ptr,recurrence.ptr,localReport.ptr);
        });
        if(multigrid&&!hierarchy.applyGridSelections)
            return hierarchy.withGridPublication(graph,dependencies,[&](const std::vector<cudaGraphNode_t>& before){return appendLocalBody(graph,before,n,controls,initialAffine,deviceControls,multigrid);});
        cudaGraphConditionalHandle handle;
        checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        const auto initialized=appendCaptured(graph,dependencies,stream,[&]{
            initializeLocalParameters<<<1,1,0,stream>>>(n,controls,localParameters.ptr,recurrence.ptr,deviceControls);
            if(n)checked(cudaMemcpyAsync(confirmedX.ptr,x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
            matvec(n,x.ptr,ap.ptr,{});
            if(n&&initialAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,stream>>>(n,initialAffine,ap.ptr);
            if(n)initialResidual<<<(n+threads-1)/threads,threads,0,stream>>>(n,b.ptr,ap.ptr,r.ptr);
            reduce(n,0,x.ptr,nullptr,&localParameters.ptr->mean,{});
            normalizeLocalMean<<<1,1,0,stream>>>(n,&localParameters.ptr->mean);
            reduce(n,3,ap.ptr,b.ptr,&localParameters.ptr->norm,{},0,nullptr,&localParameters.ptr->mean);
            reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,{});
            initializeLocalCondition<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,handle);
            const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));
            if(multigrid){
                applyVcycle(recurrence.ptr);
                partialReduction<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,2,z.ptr,r.ptr,rowSum.ptr,GpuReal(0),partial.ptr,recurrence.ptr);
            }else
            preconditionAndDot<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),r.ptr,z.ptr,partial.ptr,recurrence.ptr);
            finishLocalReduction<0><<<1,threads,threads*sizeof(GpuReal),stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
        });
        cudaGraph_t body=nullptr;const auto loop=addWhile(graph,handle,{initialized},body);
        cudaGraphConditionalHandle confirmation;
        checked(cudaGraphConditionalHandleCreate(&confirmation,body,0,cudaGraphCondAssignDefault));
        const auto iterated=appendCaptured(body,{},stream,[&]{


            const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));



            for(int batch=0;batch<4;++batch){
                if(n)direction<<<(n+threads-1)/threads,threads,0,stream>>>(n,z.ptr,recurrence.ptr,p.ptr);
                multiplyAndDot<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),p.ptr,ap.ptr,partial.ptr,recurrence.ptr);
                finishLocalReduction<1><<<1,threads,threads*sizeof(GpuReal),stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
                if(multigrid){
                    update<<<(n+threads-1)/threads,threads,0,stream>>>(n,recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr);
                    applyVcycle(recurrence.ptr);
                    blockNormAndRho<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr,z.ptr,partial.ptr);
                }else
                updatePreconditionAndNorm<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr,z.ptr,partial.ptr);
                finishLocalReduction<2><<<1,threads,threads*sizeof(GpuReal),stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
            }
            setConfirmationCondition<<<1,1,0,stream>>>(recurrence.ptr,confirmation);
        });
        cudaGraph_t confirmationBody=nullptr;
        executionGraphDetail::addIf(body,confirmation,{iterated},confirmationBody);
        appendCaptured(confirmationBody,{},stream,[&]{
            reduce(n,4,x.ptr,confirmedX.ptr,recurrence.ptr+8,{});
            if(n)checked(cudaMemcpyAsync(confirmedX.ptr,x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
            matvec(n,x.ptr,ap.ptr,{});
            if(n&&initialAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,stream>>>(n,initialAffine,ap.ptr);
            if(n)initialResidual<<<(n+threads-1)/threads,threads,0,stream>>>(n,b.ptr,ap.ptr,r.ptr);
            reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,{});
            completeTrueResidualConfirmation<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,handle);
            const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));
            if(multigrid){
                applyVcycle(recurrence.ptr);
                partialReduction<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,2,z.ptr,r.ptr,rowSum.ptr,GpuReal(0),partial.ptr,recurrence.ptr);
            }else preconditionAndDot<<<blocks,threads,threads*sizeof(GpuReal),stream>>>(n,matrix(),r.ptr,z.ptr,partial.ptr,recurrence.ptr);
            finishLocalReduction<0><<<1,threads,threads*sizeof(GpuReal),stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
        });
        return appendCaptured(graph,{loop},stream,[&]{
            publishLocalReport<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,localReport.ptr,hierarchyValid.ptr,multigrid?hierarchyCount:0);
        });
    }
    void prepareResidentCoefficients(bool factor=true){
        const int n=residentRows;const auto& values=residentValues;
        if(!n)return;
        auto assemble=[&](cudaStream_t target){
            prepareLduRows<<<(n+threads-1)/threads,threads,0,target>>>(n,offsets.ptr,entrySource.ptr,entryKind.ptr,
                boundaryOffsets.ptr,boundaryIndices.ptr,boundaryNeighbour.ptr,values.diagonal,values.upper,values.lower,
                values.source,values.boundaryDiagonal,values.boundarySource,completedDiagonal.ptr,b.ptr,rowSum.ptr,
                residentAffine?(values.boundaryJump?values.boundaryJump:boundaryJump.ptr):nullptr,residentAffine?affine.ptr:nullptr,values.diffusion);
        };
        cudaStreamCaptureStatus capture;checked(cudaStreamIsCapturing(stream,&capture));
        if(factor&&usingMultigrid(n)&&capture==cudaStreamCaptureStatusActive)hierarchy.unless(stream,hierarchy.progressing.ptr,assemble);
        else assemble(stream);
        if(factor)refreshHierarchy(n);
    }
    void prepareResidentRhs(float* solution,bool factor=true){
        prepareResidentCoefficients(factor);
        if(!residentRows)return;
        cudaStreamCaptureStatus capture;checked(cudaStreamIsCapturing(stream,&capture));
        if(factor&&usingMultigrid(residentRows)&&capture==cudaStreamCaptureStatusActive)
            hierarchy.ready(stream,[&](cudaStream_t target){checked(cudaMemcpyAsync(x.ptr,solution,residentRows*sizeof(float),cudaMemcpyDeviceToDevice,target));});
        else checked(cudaMemcpyAsync(x.ptr,solution,residentRows*sizeof(float),cudaMemcpyDeviceToDevice,stream));
    }
    void enqueueLocal(int n,const ConjugateGradientControls& controls,const GpuReal* initialAffine,bool launch=true){
        localParameters.reserve(1);localReport.reserve(1);recurrence.reserve(12);confirmedX.reserve(n);
        partial.reserve(2*std::max(1,std::min(1024,(n+threads-1)/threads)));
        const std::vector<const void*> addresses={hierarchyValid.ptr,offsets.ptr,columns.ptr,values.ptr,lduOperator?completedDiagonal.ptr:nullptr,
            entrySource.ptr,entryKind.ptr,operatorValues.upper,operatorValues.lower,operatorValues.boundarySource,rowSum.ptr,
            r.ptr,z.ptr,p.ptr,ap.ptr,confirmedX.ptr,x.ptr,b.ptr,recurrence.ptr,partial.ptr,localParameters.ptr,localReport.ptr,initialAffine,
            operatorValues.diffusion.cellOffsets,operatorValues.diffusion.cellFaces,operatorValues.diffusion.gamma,operatorValues.diffusion.area,
            operatorValues.diffusion.delta,operatorValues.diffusion.volume,operatorValues.diffusion.capacity,operatorValues.diffusion.interval,
            operatorValues.diffusion.gradientInternal,operatorValues.diffusion.gradientBoundary};
        const bool sameControls=graphControls.maximumIterations==controls.maximumIterations
            &&graphControls.minimumIterations==controls.minimumIterations&&graphControls.tolerance==controls.tolerance
            &&graphControls.relativeTolerance==controls.relativeTolerance&&graphControls.normalizationSmall==controls.normalizationSmall
            &&graphControls.singularSmall==controls.singularSmall;
        auto& graph=localGraph;
        if(graphRows!=n||graphBlockThreads!=threads||graphAddresses!=addresses||!sameControls||graphDiffusionInternal!=operatorValues.diffusion.internalFaces||graphDiffusionTransient!=operatorValues.diffusion.transient||graphDiffusionActive!=operatorValues.diffusion.active){
            graph.clear();++localRevision;graphRows=n;graphBlockThreads=threads;graphAddresses=addresses;graphControls=controls;graphDiffusionInternal=operatorValues.diffusion.internalFaces;graphDiffusionTransient=operatorValues.diffusion.transient;graphDiffusionActive=operatorValues.diffusion.active;
        }
        if(!graph.executable){
            checked(cudaGraphCreate(&graph.graph,0));
            appendLocal(graph.graph,{},n,controls,initialAffine);
            checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
        }
        if(launch)checked(cudaGraphLaunch(graph.executable,stream));
    }

    void enqueuePrepared(int n,const ConjugateGradientControls& controls,const GpuReal* initialAffine,bool resident,GpuReal* solution){
        preparedTask.wait();enqueueLocal(n,controls,initialAffine,false);
        if(!preparedGraph.executable||preparedRevision!=localRevision||preparedResident!=resident||preparedSolution!=solution){
            preparedGraph.clear();preparedRevision=localRevision;preparedResident=resident;preparedSolution=solution;
            checked(cudaGraphCreate(&preparedGraph.graph,0));auto graph=preparedGraph.graph;
            auto prepared=sparseGraphDetail::appendCaptured(graph,{},stream,[&]{if(resident)prepareResidentRhs(solution);else refreshHierarchy(n);});
            cudaGraphConditionalHandle ready;checked(cudaGraphConditionalHandleCreate(&ready,graph,0,cudaGraphCondAssignDefault));
            auto selected=sparseGraphDetail::appendCaptured(graph,{prepared},stream,[&]{vcSelectReady<<<1,1,0,stream>>>(usingMultigrid(n)?hierarchy.pending.ptr:nullptr,ready);});
            cudaGraph_t body=nullptr;executionGraphDetail::addIf(graph,ready,{selected},body);
            auto solved=appendLocal(body,{},n,controls,initialAffine);
            if(resident)sparseGraphDetail::appendCaptured(body,{solved},stream,[&]{if(n)checked(cudaMemcpyAsync(solution,x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));});
            checked(cudaGraphInstantiateWithFlags(&preparedGraph.executable,graph,0));
        }
        preparedTask.start(stream,preparedGraph.executable,[this]{return hierarchyCount&&hierarchy.servicePending();},[this,resident]{if(resident)publishTo(residentValues.readyStream);});
    }

};
SparseConjugateGradient::SparseConjugateGradient():data_(new Implementation){}
SparseConjugateGradient::~SparseConjugateGradient()=default;
ExecutionGraphSignature SparseConjugateGradient::residentExecutionSignature()const{
    const auto& d=*data_;ExecutionGraphSignature key;key.add(this);
    for(const void* p:{static_cast<const void*>(d.hierarchyValid.ptr),
        static_cast<const void*>(d.offsets.ptr),static_cast<const void*>(d.columns.ptr),static_cast<const void*>(d.entrySource.ptr),
        static_cast<const void*>(d.entryKind.ptr),static_cast<const void*>(d.boundaryOffsets.ptr),static_cast<const void*>(d.boundaryIndices.ptr),
        static_cast<const void*>(d.boundaryNeighbour.ptr),static_cast<const void*>(d.boundaryJump.ptr),static_cast<const void*>(d.affine.ptr),
        static_cast<const void*>(d.completedDiagonal.ptr),static_cast<const void*>(d.confirmedX.ptr),static_cast<const void*>(d.values.ptr),static_cast<const void*>(d.rowSum.ptr),static_cast<const void*>(d.x.ptr),
        static_cast<const void*>(d.b.ptr),static_cast<const void*>(d.r.ptr),static_cast<const void*>(d.z.ptr),
        static_cast<const void*>(d.p.ptr),static_cast<const void*>(d.ap.ptr),static_cast<const void*>(d.partial.ptr),
        static_cast<const void*>(d.recurrence.ptr),static_cast<const void*>(d.localParameters.ptr),static_cast<const void*>(d.localReport.ptr)
    })key.add(p);
    key.add(workPlan().sparseBlockThreads);key.add(d.residentRows);key.add(d.residentConfigured);key.add(d.residentAffine);
    const auto& v=d.residentValues;
    key.add(v.diagonal);key.add(v.upper);key.add(v.lower);key.add(v.source);
    key.add(v.boundaryDiagonal);key.add(v.boundarySource);key.add(v.boundaryJump);
    const auto& f=v.diffusion;key.add(f.internalFaces);key.add(f.transient);key.add(f.active);key.add(f.cellOffsets);key.add(f.cellFaces);
    key.add(f.gamma);key.add(f.area);key.add(f.delta);key.add(f.volume);key.add(f.capacity);key.add(f.interval);
    key.add(f.gradientInternal);key.add(f.gradientBoundary);
    const auto& c=d.graphControls;
    key.add(c.maximumIterations);key.add(c.minimumIterations);key.add(c.tolerance);
    key.add(d.hierarchyTopologyRevision);key.add(d.hierarchyCount);
    key.add(c.relativeTolerance);key.add(c.normalizationSmall);key.add(c.singularSmall);
    return key;
}
ConjugateGradientReport SparseConjugateGradient::solve(const SparseMatrixView& matrix,
    const double* source,double* solution,const ConjugateGradientControls& controls,const SparseCommunication& comm){
    auto& d=*data_;d.preparedTask.wait();PinnedTransferQueue::Operation operation(d.transfers);const int n=matrix.rows;

    d.offsets.upload(matrix.offsets,n+1,d.transfers);d.columns.upload(matrix.columns,matrix.entries,d.transfers);
    d.lduOperator=false;
    d.values.upload(matrix.values,matrix.entries,d.transfers);d.rowSum.upload(matrix.rowSum,n,d.transfers);
    d.b.upload(source,n,d.transfers);
    if(n>32)d.buildHierarchy(n,std::vector<int>(matrix.offsets,matrix.offsets+n+1),std::vector<int>(matrix.columns,matrix.columns+matrix.entries));

    d.cachedRows=-1;d.residentConfigured=false;
    return solvePrepared(n,solution,controls,comm);
}
ConjugateGradientReport SparseConjugateGradient::solvePrepared(int n,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm,const GpuReal* initialAffine,GpuReal* deviceSolution){
    auto& d=*data_;d.configureInterface(n,comm);PinnedTransferQueue::Operation operation(d.transfers);const int blocks=(n+threads-1)/threads;
    if(deviceSolution){d.x.reserve(n);if(n)checked(cudaMemcpyAsync(d.x.ptr,deviceSolution,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));}
    else d.x.upload(solution,n,d.transfers);
    d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
    if(!comm.addInterface&&!comm.globalSum){
        d.enqueuePrepared(n,controls,initialAffine,false,nullptr);d.preparedTask.wait();
        ConjugateGradientReport report;
        d.transfers.download(&report,d.localReport.ptr,sizeof(report));
        if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
            else d.transfers.downloadConverted(solution,d.x.ptr,n);}
        d.transfers.finish();return report;
    }
    d.refreshHierarchy(n);
    d.matvec(n,d.x.ptr,d.ap.ptr,comm);
    if(n&&initialAffine)addInitialAffine<<<blocks,threads,0,d.stream>>>(n,initialAffine,d.ap.ptr);
    if(n)initialResidual<<<blocks,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,d.r.ptr);
    GpuReal count=n;if(comm.globalSum)comm.globalSum(comm.context,&count,1);
    const GpuReal mean=d.sum(n,0,d.x.ptr,nullptr,comm)/(count?count:1);
    const GpuReal norm=d.sum(n,3,d.ap.ptr,d.b.ptr,comm,mean)+controls.normalizationSmall;
    ConjugateGradientReport report;
    report.initialResidual=report.finalResidual=d.sum(n,1,d.r.ptr,nullptr,comm)/norm;
    auto converged=[&]{return report.finalResidual<controls.tolerance ||
        (controls.relativeTolerance>controls.normalizationSmall && report.finalResidual<controls.relativeTolerance*report.initialResidual);};
    report.converged=converged();
    report.targetResidual=controls.tolerance;
    if(controls.relativeTolerance>controls.normalizationSmall)
        report.targetResidual=std::max(report.targetResidual,controls.relativeTolerance*report.initialResidual);
    d.hierarchy.refreshTask.wait();
    if(controls.minimumIterations>0||!report.converged){
        d.confirmedX.reserve(n);
        if(n)checked(cudaMemcpyAsync(d.confirmedX.ptr,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
        d.recurrence.reserve(12);checked(cudaMemsetAsync(d.recurrence.ptr,0,12*sizeof(GpuReal),d.stream));
        for(;;){
            d.iteration(n,norm,controls.singularSmall,comm);
            checked(cudaMemcpyAsync(d.hostRecurrence,d.recurrence.ptr,12*sizeof(GpuReal),cudaMemcpyDeviceToHost,d.stream));
            d.transfers.finish();report.iterations=int(d.hostRecurrence[6]);
            report.singular=d.hostRecurrence[5]!=0;
            report.finalResidual=d.hostRecurrence[4]/norm;report.converged=converged();
            const bool exhausted=report.iterations>=controls.maximumIterations&&report.iterations>=controls.minimumIterations;
            if(report.singular||exhausted||(report.converged&&report.iterations>=controls.minimumIterations)){
                const GpuReal changed=d.sum(n,4,d.x.ptr,d.confirmedX.ptr,comm);
                if(n)checked(cudaMemcpyAsync(d.confirmedX.ptr,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
                d.matvec(n,d.x.ptr,d.ap.ptr,comm);
                if(n&&initialAffine)addInitialAffine<<<blocks,threads,0,d.stream>>>(n,initialAffine,d.ap.ptr);
                if(n)initialResidual<<<blocks,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,d.r.ptr);
                report.finalResidual=d.sum(n,1,d.r.ptr,nullptr,comm)/norm;report.converged=converged();
                report.stagnated=!report.converged&&!report.singular&&!exhausted&&changed==0;
                if(report.converged||report.singular||exhausted||report.stagnated)break;
                ++report.residualReplacements;
                d.hostRecurrence[9]=1;
                checked(cudaMemcpyAsync(d.recurrence.ptr,d.hostRecurrence,12*sizeof(GpuReal),cudaMemcpyHostToDevice,d.stream));
            }
        }
    }
    report.budgetExhausted=!report.converged&&report.iterations>=controls.maximumIterations;
    if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
        else d.transfers.downloadConverted(solution,d.x.ptr,n);}
    if(d.usingMultigrid(n)){std::vector<int> valid(d.hierarchyCount);d.transfers.download(valid.data(),d.hierarchyValid.ptr,valid.size()*sizeof(int));d.transfers.finish();for(int v:valid)report.preconditionerFallbackBlocks+=v==0;}
    d.transfers.finish();return report;
}
void SparseConjugateGradient::prepareLdu(const SparseLduView& matrix,const SparseDeviceLduValues* resident,bool assemble){
    auto& d=*data_;d.preparedTask.wait();const int n=matrix.rows,nf=matrix.faces,nb=matrix.boundaryFaces;
    if(resident)d.waitFor(resident->readyStream);
    const auto same=[](const std::vector<int>& old,const int* values,int count){
        return int(old.size())==count&&(count==0||std::equal(old.begin(),old.end(),values));
    };
    if(d.cachedRows!=n||!same(d.cachedOwner,matrix.owner,nf)||!same(d.cachedNeighbour,matrix.neighbour,nf)
        ||!same(d.cachedBoundaryOwner,matrix.boundaryOwner,nb)||!same(d.cachedBoundaryNeighbour,matrix.boundaryNeighbour,nb)){
        struct Entry {int column,source,kind;};std::vector<std::vector<Entry>> rows(n);
        std::vector<std::vector<int>> boundaries(n);
        for(int f=0;f<nf;++f){rows[matrix.neighbour[f]].push_back({matrix.owner[f],f,1});rows[matrix.owner[f]].push_back({matrix.neighbour[f],f,2});}
        for(int f=0;f<nb;++f){
            const int own=matrix.boundaryOwner[f],neighbour=matrix.boundaryNeighbour[f];boundaries[own].push_back(f);
            if(neighbour>=0)rows[own].push_back({neighbour,f,3});
        }
        std::vector<int> offsets(1,0),columns,indices,kinds,bOffsets(1,0),bIndices;
        for(int c=0;c<n;++c){
            columns.push_back(c);indices.push_back(c);kinds.push_back(0);
            for(const auto& entry:rows[c]){columns.push_back(entry.column);indices.push_back(entry.source);kinds.push_back(entry.kind);}
            offsets.push_back(columns.size());bIndices.insert(bIndices.end(),boundaries[c].begin(),boundaries[c].end());bOffsets.push_back(bIndices.size());
        }
        d.offsets.upload(offsets.data(),offsets.size(),d.transfers);d.columns.upload(columns.data(),columns.size(),d.transfers);
        if(n>32)d.buildHierarchy(n,offsets,columns);
        d.entrySource.upload(indices.data(),indices.size(),d.transfers);d.entryKind.upload(kinds.data(),kinds.size(),d.transfers);
        d.boundaryOffsets.upload(bOffsets.data(),bOffsets.size(),d.transfers);d.boundaryIndices.upload(bIndices.data(),bIndices.size(),d.transfers);
        d.boundaryNeighbour.upload(matrix.boundaryNeighbour,nb,d.transfers);

        d.transfers.finish();
        const auto copy=[](std::vector<int>& out,const int* input,int count){out.clear();if(count)out.assign(input,input+count);};
        copy(d.cachedOwner,matrix.owner,nf);copy(d.cachedNeighbour,matrix.neighbour,nf);
        copy(d.cachedBoundaryOwner,matrix.boundaryOwner,nb);copy(d.cachedBoundaryNeighbour,matrix.boundaryNeighbour,nb);
        d.cachedRows=n;d.cachedEntries=columns.size();
    }
    if(!resident){
        d.diagonal.upload(matrix.diagonal,n,d.transfers);d.upper.upload(matrix.upper,nf,d.transfers);d.lower.upload(matrix.lower,nf,d.transfers);
        d.source.upload(matrix.source,n,d.transfers);d.boundaryDiagonal.upload(matrix.boundaryDiagonal,nb,d.transfers);d.boundarySource.upload(matrix.boundarySource,nb,d.transfers);
    }
    if(matrix.boundaryJump){
        if(d.cachedBoundaryJump.size()!=std::size_t(nb)||
            (nb&&std::memcmp(d.cachedBoundaryJump.data(),matrix.boundaryJump,nb*sizeof(double)))){
            d.boundaryJump.upload(matrix.boundaryJump,nb,d.transfers);
            d.cachedBoundaryJump.assign(matrix.boundaryJump,matrix.boundaryJump+nb);
        }
        d.affine.reserve(n);
    }
    d.completedDiagonal.reserve(n);d.b.reserve(n);d.rowSum.reserve(n);
    const SparseDeviceLduValues values=resident?*resident:SparseDeviceLduValues{
        d.diagonal.ptr,d.upper.ptr,d.lower.ptr,d.source.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.stream};
    d.lduOperator=true;d.operatorValues=values;
    if(n&&assemble)prepareLduRows<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.offsets.ptr,d.entrySource.ptr,d.entryKind.ptr,
        d.boundaryOffsets.ptr,d.boundaryIndices.ptr,d.boundaryNeighbour.ptr,values.diagonal,values.upper,values.lower,
        values.source,values.boundaryDiagonal,values.boundarySource,d.completedDiagonal.ptr,d.b.ptr,d.rowSum.ptr,
        matrix.boundaryJump?(values.boundaryJump?values.boundaryJump:d.boundaryJump.ptr):nullptr,matrix.boundaryJump?d.affine.ptr:nullptr,values.diffusion);
}
ConjugateGradientReport SparseConjugateGradient::solveLdu(const SparseLduView& matrix,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.residentConfigured=false;prepareLdu(matrix,nullptr);
    return solvePrepared(matrix.rows,solution,controls,comm,matrix.boundaryJump?d.affine.ptr:nullptr);
}
ConjugateGradientReport SparseConjugateGradient::solveResidentLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,GpuReal* solution,const ConjugateGradientControls& controls,const SparseCommunication& comm){
    if(!comm.addInterface&&!comm.globalSum){
        configureResidentLocalLdu(matrix,values,controls);enqueueResidentLocalLdu(solution);
        return collectResidentLocalReport();
    }
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.residentConfigured=false;prepareLdu(matrix,&values);
    const auto report=solvePrepared(matrix.rows,nullptr,controls,comm,matrix.boundaryJump?d.affine.ptr:nullptr,solution);
    d.publishTo(values.readyStream);return report;
}
void SparseConjugateGradient::configureResidentLocalLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,const ConjugateGradientControls& controls){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);prepareLdu(matrix,&values,false);
    const int n=matrix.rows;d.x.reserve(n);d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
    d.residentRows=n;d.residentValues=values;d.residentAffine=matrix.boundaryJump!=nullptr;
    d.enqueueLocal(n,controls,d.residentAffine?d.affine.ptr:nullptr,false);d.residentConfigured=true;
}
ConjugateGradientDeviceReport SparseConjugateGradient::enqueueResidentLocalLdu(GpuReal* solution){
    auto& d=*data_;d.preparedTask.wait();if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    const int n=d.residentRows;const auto& values=d.residentValues;d.waitFor(values.readyStream);
    d.enqueuePrepared(n,d.graphControls,d.residentAffine?d.affine.ptr:nullptr,true,solution);
    return {d.localReport.ptr,d.stream,d.preparedTask.result()};
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalLdu(ExecutionGraph graph,ExecutionGraphNode dependency,GpuReal* solution,const ConjugateGradientControls* deviceControls){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    d.enqueueLocal(d.residentRows,d.graphControls,d.residentAffine?d.affine.ptr:nullptr,false);
    using sparseGraphDetail::appendCaptured;
    std::vector<cudaGraphNode_t> dependencies;if(dependency)dependencies.push_back(dependency);
    const auto prepared=appendCaptured(graph,dependencies,d.stream,[&]{d.prepareResidentRhs(solution);});
    cudaGraphConditionalHandle ready;checked(cudaGraphConditionalHandleCreate(&ready,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendCaptured(graph,{prepared},d.stream,[&]{vcSelectReady<<<1,1,0,d.stream>>>(d.usingMultigrid(d.residentRows)?d.hierarchy.pending.ptr:nullptr,ready);});
    cudaGraph_t body=nullptr;const auto finished=executionGraphDetail::addIf(graph,ready,{selected},body);
    const auto solved=d.appendLocal(body,{},d.residentRows,d.graphControls,d.residentAffine?d.affine.ptr:nullptr,deviceControls);
    appendCaptured(body,{solved},d.stream,[&]{
        if(d.residentRows)checked(cudaMemcpyAsync(solution,d.x.ptr,d.residentRows*sizeof(float),cudaMemcpyDeviceToDevice,d.stream));
    });
    return finished;
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalPreparation(ExecutionGraph graph,ExecutionGraphNode dependency){
    auto& d=*data_;
    return sparseGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        d.prepareResidentCoefficients();
    });
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalPreconditioner(ExecutionGraph graph,ExecutionGraphNode dependency,const GpuReal* rhs,GpuReal* result){
    auto& d=*data_;const int n=d.residentRows;
    using sparseGraphDetail::appendCaptured;using executionGraphDetail::addIf;
    auto copied=appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        if(n)checked(cudaMemcpyAsync(d.r.ptr,rhs,n*sizeof(float),cudaMemcpyDeviceToDevice,d.stream));
    });
    cudaGraphNode_t finished=copied;
    if(d.usingMultigrid(n)){
        cudaGraphConditionalHandle diagonal,multigrid;
        checked(cudaGraphConditionalHandleCreate(&diagonal,graph,0,cudaGraphCondAssignDefault));
        checked(cudaGraphConditionalHandleCreate(&multigrid,graph,0,cudaGraphCondAssignDefault));
        auto selected=appendCaptured(graph,{copied},d.stream,[&]{selectDispatch<<<1,1,0,d.stream>>>(d.hierarchyValid.ptr,diagonal,multigrid);});
        cudaGraph_t diagonalBody=nullptr,multigridBody=nullptr;
        auto a=addIf(graph,diagonal,{selected},diagonalBody);
        finished=addIf(graph,multigrid,{a},multigridBody);
        appendCaptured(diagonalBody,{},d.stream,[&]{precondition<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.matrix(),d.r.ptr,d.z.ptr,nullptr);});
        appendCaptured(multigridBody,{},d.stream,[&]{d.applyVcycle(nullptr);});
    }else finished=appendCaptured(graph,{copied},d.stream,[&]{if(n)precondition<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.matrix(),d.r.ptr,d.z.ptr,nullptr);});
    return appendCaptured(graph,{finished},d.stream,[&]{if(n)exportPreconditioner<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.matrix(),d.z.ptr,result);});
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalResidual(ExecutionGraph graph,ExecutionGraphNode dependency,GpuReal* solution,GpuReal* residual){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse residual is not configured");
    return sparseGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        d.prepareResidentRhs(solution,false);const int n=d.residentRows;d.matvec(n,d.x.ptr,d.ap.ptr,{});
        if(n&&d.residentAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.affine.ptr,d.ap.ptr);
        if(n)initialResidual<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,residual);
    });
}
const int* SparseConjugateGradient::capacityPendingDevice() const{return data_->hierarchyCount?data_->hierarchy.pending.ptr:nullptr;}
bool SparseConjugateGradient::servicePendingCapacity(){return data_->hierarchyCount&&data_->hierarchy.servicePending();}
const ConjugateGradientReport* SparseConjugateGradient::deviceReport() const{return data_->localReport.ptr;}
ConjugateGradientReport SparseConjugateGradient::collectResidentLocalReport(){
    auto& d=*data_;d.preparedTask.wait();PinnedTransferQueue::Operation operation(d.transfers);ConjugateGradientReport report;
    if(!d.residentConfigured)throw std::logic_error("resident local sparse report is not configured");
    d.transfers.download(&report,d.localReport.ptr,sizeof(report));d.transfers.finish();return report;
}
void SparseConjugateGradient::residualResidentLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,const GpuReal* solution,GpuReal* residual,const SparseCommunication& comm){
    auto& d=*data_;
    if(d.residentConfigured){
        if(!d.residualEvaluator)d.residualEvaluator.reset(new SparseConjugateGradient);
        d.residualEvaluator->residualResidentLdu(matrix,values,solution,residual,comm);
        return;
    }
    PinnedTransferQueue::Operation operation(d.transfers);prepareLdu(matrix,&values);
    const int n=matrix.rows;d.configureInterface(n,comm);d.ap.reserve(n);
    d.matvec(n,solution,d.ap.ptr,comm);
    if(n&&matrix.boundaryJump)addInitialAffine<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.affine.ptr,d.ap.ptr);
    if(n)initialResidual<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,residual);
    d.publishTo(values.readyStream);
}

}}

#undef threads
