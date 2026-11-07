#include "gpu/PolyhedralGeometry.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <vector>
namespace {
namespace gpu=ugkp::gpu::geometry;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void selectProblem(const gpu::MinimumNormDeviceProblem* initialized,gpu::MinimumNormDeviceProblem* selected){*selected=*initialized;}
struct Fixture {
    gpu::ResidentDenseMinimumNorm solver;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    gpu::MinimumNormDeviceProblem *seed=nullptr,*selected=nullptr;
    double *matrix=nullptr,*rhs=nullptr;
    int rows=0,columns=0;
    Fixture(int nr,int nc):solver(nr,nc),rows(nr),columns(nc){
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMalloc(&seed,sizeof(*seed)));check(cudaMalloc(&selected,sizeof(*selected)));
        if(nr*nc)check(cudaMalloc(&matrix,std::size_t(nr)*nc*sizeof(double)));
        if(nr)check(cudaMalloc(&rhs,nr*sizeof(double)));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        selectProblem<<<1,1,0,stream>>>(seed,selected);solver.solve(selected,stream);
        check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("resident QR graph contains host/transfer node");}
        check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    }
    ~Fixture(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(seed)cudaFree(seed);if(selected)cudaFree(selected);if(matrix)cudaFree(matrix);if(rhs)cudaFree(rhs);if(stream)cudaStreamDestroy(stream);}
    gpu::MinimumNormResult run(int nr,int nc,const std::vector<double>& a,const std::vector<double>& b,double tolerance){
        // Only test-fixture initialization and final observation use host data.
        // The same captured graph consumes device-selected dimensions/inputs.
        if(!a.empty())check(cudaMemcpy(matrix,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice));
        if(!b.empty())check(cudaMemcpy(rhs,b.data(),b.size()*sizeof(double),cudaMemcpyHostToDevice));
        const gpu::MinimumNormDeviceProblem problem{nr,nc,tolerance,matrix,rhs};
        check(cudaMemcpy(seed,&problem,sizeof(problem),cudaMemcpyHostToDevice));
        for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        const auto view=solver.view();gpu::MinimumNormDeviceStatus state;
        check(cudaMemcpy(&state,view.status,sizeof(state),cudaMemcpyDeviceToHost));
        gpu::MinimumNormResult result;result.rank=state.rank;result.initialized=state.error!=1;result.valid=state.error==0;
        if(result.initialized){result.delta.resize(nc);result.rhs.resize(nr);
            if(nc)check(cudaMemcpy(result.delta.data(),view.delta,nc*sizeof(double),cudaMemcpyDeviceToHost));
            if(nr)check(cudaMemcpy(result.rhs.data(),view.rhs,nr*sizeof(double),cudaMemcpyDeviceToHost));}
        std::vector<double> retained(a.size());if(!a.empty())check(cudaMemcpy(retained.data(),matrix,a.size()*sizeof(double),cudaMemcpyDeviceToHost));
        if(!a.empty()&&std::memcmp(a.data(),retained.data(),a.size()*sizeof(double)))throw std::runtime_error("resident QR mutated borrowed matrix");
        retained.resize(b.size());if(!b.empty())check(cudaMemcpy(retained.data(),rhs,b.size()*sizeof(double),cudaMemcpyDeviceToHost));
        if(!b.empty()&&std::memcmp(b.data(),retained.data(),b.size()*sizeof(double)))throw std::runtime_error("resident QR mutated borrowed RHS");
        return result;
    }
};
}
ugkp::gpu::geometry::MinimumNormResult residentMinimumNorm(int nr,int nc,const std::vector<double>& a,
    const std::vector<double>& b,double tolerance){
    static Fixture fixture(129,73);return fixture.run(nr,nc,a,b,tolerance);
}
void residentMinimumNormShapeChecks(){
    Fixture empty(0,0);if(!empty.run(0,0,{}, {},1e-12).valid)throw std::runtime_error("empty QR failed");
    if(empty.run(1,0,{}, {},1e-12).initialized)throw std::runtime_error("invalid QR capacity accepted");
    if(!empty.run(0,0,{}, {},1e-12).valid)throw std::runtime_error("QR did not recover after invalid shape");
}
