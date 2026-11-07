#include "applications/CHMT/gpu/CoupledGasInterval.H"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
using namespace chmt;
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T> T* device(const std::vector<T>& x){T* p=nullptr;check(cudaMalloc(&p,x.size()*sizeof(T)));check(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void queuedIntervals(ResidentWallClock clock,const ResidentWallKnotTime* knots,int count,
    const double* steps,int n,CoupledGasInterval* output){
    if(blockIdx.x||threadIdx.x)return;
    for(int i=0;i<n;++i)output[i]=prepareCoupledWallInterval(clock,knots,count,steps[i],17);
}
int main(){
    int failures=0,comparisons=0,roundedStamps=0;double maxAlpha=0;
    for(int configuration=0;configuration<6;++configuration){
        const double origins[]={0.,.13,1.e12,0.,1.e12,1.e-200};
        const double increments[]={.125,1.e-7,1.e-12,1.e-200,3.e-7,1.e-202};
        const int n=4096;const double step=increments[configuration];
        std::vector<double> steps(n);std::vector<WindowTime> elapsed(n+1);
        for(int i=0;i<n;++i){steps[i]=step*(configuration%2?1.+double(i%7)*.125:1.);elapsed[i+1]=elapsed[i]+steps[i];}
        WallProgram program;program.interval.begin=origins[configuration];
        for(int i=0;i<=n;i+=256){WallKnot knot;knot.localTime=elapsed[i];knot.time=windowAbsolute(program.interval.begin,knot.localTime);program.knots.push_back(knot);}
        std::vector<ResidentWallKnotTime> times;for(const auto& k:program.knots)times.push_back({residentWindowTime(k.localTime),k.time});
        auto* td=device(times);auto* dt=device(steps);CoupledGasInterval* out=nullptr;check(cudaMalloc(&out,n*sizeof(*out)));
        ResidentWallClock clock;clock.origin=program.interval.begin;
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        queuedIntervals<<<1,1,0,stream>>>(clock,td,int(times.size()),dt,n,out);
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        // Replay resets only its local clock; all 4096 decisions remain on device.
        for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
        std::vector<CoupledGasInterval> result(n);check(cudaMemcpyAsync(result.data(),out,n*sizeof(*out),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        for(int i=0;i<n;++i){const auto& got=result[i];if(got.sequence!=std::uint64_t(i+1)||got.geometry!=17)++failures;
            const auto previous=i?result[i-1].localEnd:ugkp::CompensatedTime{};
            if(!(got.localBegin==previous)||!(got.localEnd==got.localBegin+steps[i])||got.actualDt!=steps[i])++failures;
            for(const auto stamp:{std::make_pair(got.localBegin,elapsed[i]),std::make_pair(got.localEnd,elapsed[i+1])}){
                const auto expected=windowDuration(stamp.second,WindowTime{}),actual=stamp.first.value();
                if(actual!=expected&&actual!=std::nextafter(expected,actual))++failures;
            }
            for(int stage=0;stage<4;++stage){const auto time=elapsed[i]+(stage==0?0:stage==2?.5*steps[i]:steps[i]);
                GasWallProgramPosition expected;std::string error;if(!locateGasWallProgramLocal(program,time,expected,error))return 3;
                const auto actual=stage==3?got.endpoint:got.position[stage];
                const double difference=std::abs(actual.alpha-expected.alpha);maxAlpha=std::max(maxAlpha,difference);
                if(actual.lower!=expected.lower||actual.upper!=expected.upper||difference>8*std::numeric_limits<double>::epsilon()){
                    if(failures<8)std::printf("mismatch config=%d step=%d stage=%d expected=%d/%d %.17g actual=%d/%d %.17g\n",configuration,i,stage,expected.lower,expected.upper,expected.alpha,actual.lower,actual.upper,actual.alpha);++failures;}
                if(stage<3){const double expectedStamp=windowAbsolute(program.interval.begin,time),actualStamp=got.time[stage];
                    // The old x87 sum rounds to extended precision before its
                    // FP64 cast. A directly rounded FP64 pair can differ by one
                    // ULP at this publication boundary, without moving a knot.
                    if(actualStamp!=expectedStamp){++roundedStamps;
                        if(actualStamp!=std::nextafter(expectedStamp,actualStamp))++failures;}}
                ++comparisons;
            }
        }
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));check(cudaFree(td));check(cudaFree(dt));check(cudaFree(out));
    }
    std::printf("wall clock: %d stage/endpoint comparisons, maximum alpha error %.17g, %d one-ULP output stamps, %d failures\n",comparisons,maxAlpha,roundedStamps,failures);
    return failures?1:0;
}
