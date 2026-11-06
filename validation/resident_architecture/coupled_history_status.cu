#include "applications/CHMT/gpu/CoupledSurfaceHistory.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace chmt;
using namespace ugkp::diagnostics;
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
struct Fixture {CoupledSurfaceDownload records[2]{};CoupledWallDownload walls[2]{};int faces[2]{11,29};CoupledRecordFailure first{2,2};DeviceContext context{};};
__global__ void validateFixtures(Fixture* data,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto& f=data[i];
    beginOperation(f.context,7,19,.25);
    reportCoupledRecordFailure(f.first,2,f.records,f.walls,f.faces,3,5,&f.context);
    // A later operation, including another failing transaction, cannot replace
    // the original provenance. No CPU observation occurs between these calls.
    if(hasFatal(&f.context)){beginOperation(f.context,8,20,.5);
        reportCoupledRecordFailure(f.first,2,f.records,f.walls,f.faces,3,5,&f.context);}
}
int main(){
    constexpr int n=8;Fixture input[n];
    for(auto& f:input)for(auto& r:f.records){r.primary.kind=ExchangeKind::GasSolid;r.primary.solidCell=0;}
    input[1].first={1,0};input[1].records[1].failed=1;input[1].records[0].primary.solidCell=-1;
    input[2].first={0,2};input[2].records[0].failed=2;input[2].walls[0].status.node=6;
    input[2].walls[0].status.iteration=17;input[2].walls[0].status.residual=.125;input[2].walls[0].transportCode=9;
    input[3].first={2,1};input[3].records[1].primary.solidCell=3;
    input[4].first={2,0};input[4].records[0].primary.kind=ExchangeKind::GasFilm;input[4].records[0].primary.filmFace=-1;
    input[5].first={2,1};input[5].records[1].pore.kind=ExchangeKind::PoreGas;input[5].records[1].pore.mass=1;input[5].records[1].pore.solidCell=8;
    input[6].first={2,0};input[6].records[0].primary.kind=ExchangeKind::ParticleGas;input[6].records[0].primary.solidCell=-1;input[6].records[0].particleWallEnergy=1;
    input[7].first={0,2};input[7].records[0].failed=1;
    beginOperation(input[7].context,3,4,.1);recordFatal(&input[7].context,makeRecord(Code::NonFinite,123));
    Fixture* data=nullptr;check(cudaMalloc(&data,sizeof(input)));check(cudaMemcpy(data,input,sizeof(input),cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));validateFixtures<<<1,32,0,stream>>>(data,n);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,stream));
    Fixture result[n];check(cudaMemcpyAsync(result,data,sizeof(result),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    const unsigned sites[n]={0,2631,2632,2633,2634,2633,2633,123};const int faces[n]={0,29,11,29,11,29,11,-1};int failures=0;
    for(int i=0;i<n;++i){const auto& c=result[i].context;
        if(bool(c.fatalClaim)!=(i!=0))++failures;
        if(i){const auto& f=c.firstFatal;if(f.sourceSiteId!=sites[i]||f.entityIndex!=faces[i]
            ||f.operationSequence!=(i==7?3:7)||f.step!=(i==7?4:19)||f.simulationTime!=(i==7?.1:.25))++failures;}}
    const auto& wall=result[2].context.firstFatal;
    if(wall.component!=6||wall.observed!=.125||wall.reference!=17||wall.corrected!=9)++failures;
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));check(cudaFree(data));
    std::printf("coupled history status: eight priority/owner/sticky scenarios, %d failures\n",failures);return failures?1:0;
}
