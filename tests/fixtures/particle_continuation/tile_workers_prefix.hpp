
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
using std::min;using std::max;
#define __global__
#define __device__
#define __forceinline__ inline
#define __syncthreads() ((void)0)
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) (x)
#define GPU_POOL_STATIC_DIRECTORY 0
#define GPU_POOL_SPLIT_REDUCTION_TOPOLOGY PoolReductionTopology::sharedTree
#define GPU_MOMENT_REAL double
#define GPU_MOMENT_R(x) (x)
#define GPU_MOMENT_THERMAL 0
#define GPU_MOMENT_GATHER 0
struct Dim{unsigned x=0;} blockIdx,threadIdx,blockDim;
std::vector<int> seen;int expectedBegin,expectedEnd;
void record(int i){if(i<expectedBegin||i>=expectedEnd)throw std::runtime_error("invalid worker position="+std::to_string(i));seen.push_back(i);}
struct IndexProxy{bool recordIndex=true;int operator[](int i){record(i);return recordIndex?0:i;}};
struct DeviceState {
 int nCells=1,csrHeavyReductionEnabled=0,particleCapacity=INT_MAX;
 int base[2]={},offset[2]={};int* preBaseCellOffset=base;int* cellParticleOffset=offset;
 IndexProxy sortedParticleIndex;
 int pStatus[1]={1},pCellId[1]={0};double pm[1]={1},pux[1]={1},puy[1]={1},puz[1]={1},pTheta[1]={1},pd[1]={1},pT[1]={1};
 double particleDiameterFallback=1,TpMin=0,TpMax=2;
};
double particleHeatFactorDevice(DeviceState&){return 1;}
double finiteOr(double a,double){return a;}
double clampMin(double a,double b){return max(a,b);}
double clampRange(double a,double lo,double hi){return min(max(a,lo),hi);}
double sqr3(double a,double b,double c){return a*a+b*b+c*c;}
template<bool P>void accumulateOnePoolParticle(DeviceState&,int,int i,double,double&,double&,double&,double&,double&,double&,double&,double&){if(i)record(i);}
template<bool P>void accumulateCsrSplitLogicalPoolParticle(DeviceState&,int,int i,double,double&,double&,double&,double&,double&,double&,double&,double&){record(i);}
template<int N>void blockReduceComponentSums(double(&)[N],double*){}
enum class PoolReductionTopology{sharedTree,warpComponents};
template<PoolReductionTopology T>void reducePoolMoments(double(&)[8],double*){}
template<bool A>void publishPoolCell(DeviceState&,int,double(&)[8]){}
double poissonCollisionProbabilityForCell(DeviceState&,int,double){return 1;}
