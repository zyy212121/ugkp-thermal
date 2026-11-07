// Included by PolyhedralGeometry.cu.
#include "SweepTrial.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
struct SweepTrialPartial {int invalid=INT_MAX;bool changed=false;};
__global__ void sweepTrialPoints(int count,const SweepTrialProblem* input,Point* output,SweepTrialPartial* partial){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;SweepTrialPartial result;
    const int backtrack=input->backtrack;
    const double alpha=backtrack>=0?::ldexp(1.,-backtrack):0;
    if(p<count){
        const Point old=input->origin[p],correction=input->correction[p];
        const auto x=sweepTrialCoordinate(old.x,correction.x,alpha);
        const auto y=sweepTrialCoordinate(old.y,correction.y,alpha);
        const auto z=sweepTrialCoordinate(old.z,correction.z,alpha);
        const bool valid=backtrack>=0&&x.representable&&y.representable&&z.representable;
        output[p]=valid?Point{x.value,y.value,z.value}:old;
        if(!valid)result.invalid=p;
        result.changed=x.changed||y.changed||z.changed;
    }
    __shared__ SweepTrialPartial values[128];values[threadIdx.x]=result;__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride){
        const auto other=values[threadIdx.x+stride];
        if(other.invalid<values[threadIdx.x].invalid)values[threadIdx.x].invalid=other.invalid;
        values[threadIdx.x].changed=values[threadIdx.x].changed||other.changed;
    }__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void finishSweepTrial(int blocks,const SweepTrialProblem* input,const SweepTrialPartial* partial,SweepTrialStatus* output){
    int invalid=INT_MAX;bool changed=false;
    for(int b=threadIdx.x;b<blocks;b+=128){if(partial[b].invalid<invalid)invalid=partial[b].invalid;changed=changed||partial[b].changed;}
    __shared__ SweepTrialPartial values[128];values[threadIdx.x]={invalid,changed};__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride){const auto other=values[threadIdx.x+stride];
        if(other.invalid<values[threadIdx.x].invalid)values[threadIdx.x].invalid=other.invalid;
        values[threadIdx.x].changed=values[threadIdx.x].changed||other.changed;
    }__syncthreads();}
    if(threadIdx.x)return;SweepTrialStatus result;
    result.alpha=input->backtrack>=0?::ldexp(1.,-input->backtrack):0;
    result.representable=input->backtrack>=0&&values[0].invalid==INT_MAX;
    result.firstInvalidPoint=values[0].invalid==INT_MAX?-1:values[0].invalid;
    result.changed=result.representable&&values[0].changed;
    *output=result;
}
}
struct ResidentSweepTrial::Implementation {
    int points=0,blocks=0;Array<Point> endpoint;Array<SweepTrialPartial> partial;Array<SweepTrialStatus> status;
};
ResidentSweepTrial::ResidentSweepTrial(int points):data_(new Implementation){
    if(points<0)throw std::invalid_argument("negative sweep trial point count");auto& d=*data_;d.points=points;d.blocks=points?(points-1)/128+1:0;
    d.endpoint.reserve(points);d.partial.reserve(d.blocks);d.status.reserve(1);
}
ResidentSweepTrial::~ResidentSweepTrial()=default;
SweepTrialView ResidentSweepTrial::view()const{const auto& d=*data_;return {d.endpoint.pointer,d.status.pointer};}
void ResidentSweepTrial::evaluate(const SweepTrialProblem* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    if(d.blocks)sweepTrialPoints<<<d.blocks,128,0,stream>>>(d.points,input,d.endpoint.pointer,d.partial.pointer);
    finishSweepTrial<<<1,128,0,stream>>>(d.blocks,input,d.partial.pointer,d.status.pointer);checked(cudaGetLastError());
}
}}}
