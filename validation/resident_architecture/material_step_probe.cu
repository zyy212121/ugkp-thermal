#include "material_step_probe.H"
#include "applications/CHMT/gpu/ResidentMaterialStep.cuh"
#include "applications/CHMT/gpu/ResidentMaterialDrive.H"
#include "common/gasTransport/GasDriveSelection.cuh"
#include <stdexcept>
namespace materialStepProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
};
std::vector<chmt::ResidentMaterialStep> evaluate(const std::vector<Test>& tests){
    using namespace chmt;const auto n=tests.size();
    std::vector<ugkp::IntervalSliceCursor> cursors;std::vector<ugkp::CompensatedTime> totals;
    std::vector<ResidentWallKnotTime> knots;std::vector<MaterialTransportDeviceResult> transports;
    std::vector<CoupledSurfaceDownload> samples(2*n);
    std::vector<ResidentMaterialDriveInterval> times;
    for(std::size_t i=0;i<n;++i){const auto& t=tests[i];cursors.push_back({t.begin,0});totals.push_back(t.total);knots.insert(knots.end(),t.knots.begin(),t.knots.end());
        MaterialTransportDeviceResult transport;transport.dtLimit=t.transportLimit;transport.errorCode=t.transportError;transports.push_back(transport);
        samples[2*i].bulk.temperature=300;samples[2*i+1]=samples[2*i];if(t.changed)samples[2*i+1].bulk.temperature=600;
        times.push_back({ugkp::CompensatedTime(0),ugkp::CompensatedTime(1),1});times.push_back({ugkp::CompensatedTime(1),ugkp::CompensatedTime(2),1});}
    Device<ugkp::IntervalSliceCursor> cursor(cursors);Device<ugkp::CompensatedTime> total(totals);
    Device<ResidentWallKnotTime> dk(knots);Device<MaterialTransportDeviceResult> transport(transports);
    Device<CoupledSurfaceDownload> surface(samples);Device<ResidentMaterialDriveInterval> interval(times);
    Device<ResidentMaterialStep> step{std::vector<ResidentMaterialStep>(n)};
    Device<ugkwp::GasDriveControls> control{std::vector<ugkwp::GasDriveControls>(n)};
    Device<ugkwp::GasDriveSelection> selected{std::vector<ugkwp::GasDriveSelection>(n)};Device<int> scratch{std::vector<int>(2*n)};
    check(cudaDeviceSynchronize());cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));std::size_t offset=0;
    for(std::size_t i=0;i<n;++i){const auto& t=tests[i];
        requestResidentMaterialStep<<<1,1,0,stream>>>(cursor.p+i,total.p+i,t.controls,step.p+i,control.p+i);
        ResidentMaterialDriveHistory history{surface.p+2*i,nullptr,interval.p+2*i,2,1,nullptr};
        check(ugkwp::selectGasDriveInterval<Ns>(history,control.p+i,selected.p+i,scratch.p+2*i,stream));
        alignResidentMaterialStep<<<1,1,0,stream>>>(step.p+i,selected.p+i,dk.p?dk.p+offset:nullptr,t.knots.size());offset+=t.knots.size();
        limitResidentMaterialStep<<<1,1,0,stream>>>(step.p+i,total.p+i,transport.p+i);
    }
    check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("material timing graph contains a host/copy node");}
    check(cudaGraphInstantiateWithFlags(&exec,graph,0));for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
    std::vector<ResidentMaterialStep> result(n);if(n)check(cudaMemcpy(result.data(),step.p,n*sizeof(*step.p),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
