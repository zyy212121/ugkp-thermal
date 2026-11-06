#define main unused_polyhedral_resident_main
#include "polyhedral_resident.cu"
#undef main
__host__ __device__ Point motionPoint(int i,int phase){return {double(i)*(1+phase*.03125),.125*(i%3),.0625*(i%5)};}
__host__ __device__ Point boundaryValue(int i,int phase){return {.001953125*(phase+1),.00390625*(i%7+phase),-.0078125};}
__host__ __device__ MotionPenalty penaltyValue(int i,int phase){MotionPenalty p;addNormalHeight(p,{0,1,0},.001953125*(i%5+phase),.5);return p;}
__global__ void prepareMotionInputs(int n,const int* phase,Point* points,Point* boundary,MotionPenalty* penalty,double* tolerance,bool bad){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!i)*tolerance=*phase%2?1e-9:1e-11;
    if(i<n){points[i]=motionPoint(bad&&*phase==2&&i==2?1:i,*phase);boundary[i]=boundaryValue(i,*phase);penalty[i]=penaltyValue(i,*phase);}
}
__global__ void archiveMotion(int n,const int* phase,HarmonicDeviceView input,Point* points,HarmonicDeviceStatus* report){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!i)report[*phase]=*input.status;
    if(i<n&&input.status->errorPoint<0)points[*phase*n+i]=input.points[i];
}
HarmonicResult referenceMotion(const std::vector<Point>& points,const std::vector<Point>& boundary,
    const std::vector<int>& offsets,const std::vector<int>& neighbour,const std::vector<unsigned char>& fixed,
    const std::vector<MotionPenalty>& penalties,double tolerance,int maximum,bool represented){
    const int n=points.size();HarmonicResult result;std::vector<double> weights(neighbour.size()),total(n);
    std::vector<Point> displacement(n),next(n);
    for(int i=0;i<n;++i){if(fixed[i]){displacement[i]=boundary[i];continue;}
        for(int k=offsets[i];k<offsets[i+1];++k){const double length=magnitude(points[i]-points[neighbour[k]]);
            if(!(length>0)){result.errorPoint=i;result.errorKind=1;return result;}weights[k]=1/length;total[i]+=weights[k];}}
    bool active=n>0&&maximum>0;
    while(active){double residual=0;
        for(int i=0;i<n;++i){if(fixed[i]){next[i]=displacement[i];continue;}
            Point sum;for(int k=offsets[i];k<offsets[i+1];++k)sum+=displacement[neighbour[k]]*weights[k];
            next[i]=penalties.empty()?sum/total[i]:motionUpdate(sum,total[i],penalties[i]);
            residual=std::max(residual,magnitude(next[i]-displacement[i]));}
        displacement.swap(next);result.residual=residual;++result.iterations;
        active=!(residual<=tolerance)&&result.iterations<maximum;
    }
    for(int i=0;i<n;++i){Point output=points[i]+displacement[i];
        if(represented){output=output-points[i];if(fixed[i])output=boundary[i];}result.points.push_back(output);}
    return result;
}
void motionFixture(int n,bool penalty,bool represented,bool bad){
    const int replays=4,limit=400;std::vector<Point> initial;std::vector<int> offsets(1,0),neighbour;std::vector<unsigned char> fixed(n,0);
    for(int i=0;i<n;++i){initial.push_back(motionPoint(i,0));if(i)neighbour.push_back(i-1);if(i+1<n)neighbour.push_back(i+1);offsets.push_back(neighbour.size());}
    if(n){fixed.front()=1;fixed.back()=1;}
    HarmonicPointMotion motion(initial,offsets,neighbour,fixed,1e-12,limit);
    Buffer<Point> points(n),boundary(n),savedPoints(n*replays);Buffer<MotionPenalty> penalties(n);
    Buffer<int> phase(1);phase.upload({0});Buffer<double> tolerance(1);Buffer<HarmonicDeviceStatus> savedStatus(replays);
    // Error samples do not publish a position payload; initialize archive so
    // its unused entries are not uninitialized device reads at final output.
    if(n)check(cudaMemset(savedPoints.p,0,n*replays*sizeof(Point)));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    prepareMotionInputs<<<std::max(1,(n+127)/128),128,0,stream>>>(n,phase.p,points.p,boundary.p,penalties.p,tolerance.p,bad);
    const auto view=motion.solveDevice(points.p,boundary.p,penalty?penalties.p:nullptr,tolerance.p,represented,stream);
    archiveMotion<<<std::max(1,(n+127)/128),128,0,stream>>>(n,phase.p,view,savedPoints.p,savedStatus.p);
    advance<<<1,1,0,stream>>>(phase.p);check(cudaGetLastError());
    cudaGraph_t graph;check(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int i=0;i<replays;++i)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    const auto actual=savedPoints.read();const auto reports=savedStatus.read();
    for(int step=0;step<replays;++step){std::vector<Point> geometry,bc;std::vector<MotionPenalty> terms;
        for(int i=0;i<n;++i){geometry.push_back(motionPoint(bad&&step==2&&i==2?1:i,step));bc.push_back(boundaryValue(i,step));if(penalty)terms.push_back(penaltyValue(i,step));}
        const auto expected=referenceMotion(geometry,bc,offsets,neighbour,fixed,terms,step%2?1e-9:1e-11,limit,represented);
        assert(reports[step].errorPoint==expected.errorPoint&&reports[step].errorKind==expected.errorKind);
        assert(reports[step].iterations==expected.iterations);close(reports[step].residual,expected.residual);
        if(expected.errorPoint<0)for(int i=0;i<n;++i)close(actual[step*n+i],expected.points[i]);
    }
    // Reusing the resident entry must not corrupt the host publication path's
    // cached geometry weights. Invoke it after the final moving-coordinate solve.
    std::vector<Point> originalBoundary;for(int i=0;i<n;++i)originalBoundary.push_back(boundaryValue(i,0));
    const auto original=referenceMotion(initial,originalBoundary,offsets,neighbour,fixed,{},1e-12,limit,represented);
    for(int publication=0;publication<2;++publication){
        const auto published=motion.solve(originalBoundary,represented);
        assert(published.errorPoint==original.errorPoint&&published.iterations==original.iterations);
        for(int i=0;i<n;++i)close(published.points[i],original.points[i]);
        if(!publication){
            // Replaying a previously captured solve cannot update host cache
            // metadata. A publication between replays must not stale weights.
            phase.upload({0});
            for(int i=0;i<replays;++i)check(cudaGraphLaunch(executable,stream));
            check(cudaStreamSynchronize(stream));
        }
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
int main(){int configurations=0;
    for(int n:{0,1,17,257})for(bool penalty:{false,true})for(bool represented:{false,true}){
        motionFixture(n,penalty,represented,n>2);++configurations;
    }
    std::printf("PASS %d resident harmonic configurations, %d captured replays: device coordinates, boundary, penalty, tolerance, error recovery and publication reuse\n",configurations,8*configurations);
    return 0;
}
