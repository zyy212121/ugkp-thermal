#define main legacyHarmonicControlMain
#include "harmonic_control_cuda.cpp"
#undef main
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void selectMotionInput(const g::HarmonicMotionDeviceInput* seed,g::HarmonicMotionDeviceInput* out){*out=*seed;}
g::HarmonicResult selected(g::HarmonicPointMotion& owner,const std::vector<g::Point>& boundary,
    double tolerance,bool represented,bool alias=false){
    g::Point* b=nullptr;double* tol=nullptr;g::HarmonicMotionDeviceInput *seed=nullptr,*input=nullptr;
    if(!boundary.empty()){check(cudaMalloc(&b,boundary.size()*sizeof(g::Point)));
        check(cudaMemcpy(b,boundary.data(),boundary.size()*sizeof(g::Point),cudaMemcpyHostToDevice));}
    check(cudaMalloc(&tol,sizeof(double)));check(cudaMemcpy(tol,&tolerance,sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMalloc(&seed,sizeof(*seed)));check(cudaMalloc(&input,sizeof(*input)));
    const g::HarmonicMotionDeviceInput description{alias?owner.deviceView().points:owner.initialPoints(),b,nullptr,tol};
    check(cudaMemcpy(seed,&description,sizeof(description),cudaMemcpyHostToDevice));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    selectMotionInput<<<1,1,0,stream>>>(seed,input);const auto view=owner.solveSelectedDevice(input,represented,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("selected harmonic graph has host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    g::HarmonicDeviceStatus status;check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    g::HarmonicResult out;out.residual=status.residual;out.iterations=status.iterations;out.errorPoint=status.errorPoint;out.errorKind=status.errorKind;
    if(status.errorPoint<0){out.points.resize(boundary.size());if(!boundary.empty())check(cudaMemcpy(out.points.data(),view.points,out.points.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    if(b)check(cudaFree(b));check(cudaFree(tol));check(cudaFree(seed));check(cudaFree(input));return out;
}
void compare(const g::HarmonicResult& expected,const g::HarmonicResult& got){
    assert(expected.errorPoint==got.errorPoint&&expected.errorKind==got.errorKind&&expected.iterations==got.iterations);
    exact(expected.residual,got.residual);assert(expected.points.size()==got.points.size());
    for(std::size_t p=0;p<expected.points.size();++p){exact(expected.points[p].x,got.points[p].x);exact(expected.points[p].y,got.points[p].y);exact(expected.points[p].z,got.points[p].z);}
}
}
int main(){int cases=0;
    for(int n:{3,17,1025,50001}){
        std::vector<g::Point> points(n),boundary(n);std::vector<int> offsets{0},adjacent;std::vector<unsigned char> fixed(n);
        for(int p=0;p<n;++p){points[p]={.125*p,.0625*(p%7),-.03125*(p%3)};fixed[p]=p%11==0||p==n-1;
            boundary[p]={.01*(p%3),-.03*(p%7),.002*(p%2)};if(p)adjacent.push_back(p-1);if(p+1<n)adjacent.push_back(p+1);offsets.push_back(adjacent.size());}
        for(int cap:{0,1,7,37})for(double tolerance:{0.,1e-12,std::numeric_limits<double>::quiet_NaN()})for(bool represented:{false,true}){
            g::HarmonicPointMotion owner(points,offsets,adjacent,fixed,tolerance,cap);
            compare(reference(points,offsets,adjacent,fixed,boundary,tolerance,cap,represented),selected(owner,boundary,tolerance,represented));++cases;}
    }
    const std::vector<g::Point> points={{1.25,.2,.3},{2.5,.4,.7},{4.1,.1,1}},boundary={{.03,.02,-.01},{},{-.02,.07,.09}};
    const std::vector<int> rows={0,1,3,4},adjacent={1,0,2,1};const std::vector<unsigned char> fixed={1,0,1};
    for(bool represented:{false,true}){
        g::HarmonicPointMotion owner(points,rows,adjacent,fixed,1e-12,37);
        const auto first=reference(points,rows,adjacent,fixed,boundary,1e-12,37,represented);
        compare(first,selected(owner,boundary,1e-12,represented));
        compare(reference(first.points,rows,adjacent,fixed,boundary,1e-12,37,represented),selected(owner,boundary,1e-12,represented,true));
        compare(first,owner.solve(boundary,represented));++cases;
    }
    std::cout<<"PASS "<<cases<<" device-selected harmonic controls, grid stride, represented in-place publication and legacy restoration\n";
}
