// Compile this translation unit with the production backend's normal includes.
// Including production code exercises real kernels/callbacks, including private
// projection views and the public API; it does not duplicate their algorithms.
#include "../../applications/gasUGKP/private_backend/GpuResidentStrict.cu"
#include <cassert>
#include <iostream>

namespace {
void checked(cudaError_t error) {
    if(error!=cudaSuccess) {
        std::cerr<<cudaGetErrorString(error)<<'\n';std::abort();
    }
}
struct DeviceAllocations {
    std::vector<void*> items;
    template<class T> T* make(std::size_t count=1) {
        T* result=nullptr;checked(cudaMalloc(&result,count*sizeof(T)));
        items.push_back(result);return result;
    }
    ~DeviceAllocations(){for(auto* item:items)checked(cudaFree(item));}
};
void checkSelection(cudaStream_t stream) {
    using namespace ugkp::diagnostics;
    constexpr int nc=513,nf=1031;
    DeviceAllocations memory;
    auto* context=memory.make<DeviceContext>();auto* scratch=memory.make<unsigned long long>();
    auto* cells=memory.make<int>(nc);auto* faces=memory.make<int>(nf);
    std::vector<int> hc(nc,int(ugkwp::GasTransportCode::InvalidComposition)),hf(nf,0);
    DeviceContext initial{},actual{};reset(initial,31,42,7,0.125);
    const auto check=[&]() {
        checked(cudaMemcpyAsync(context,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream));
        checked(cudaMemcpyAsync(cells,hc.data(),nc*sizeof(int),cudaMemcpyHostToDevice,stream));
        checked(cudaMemcpyAsync(faces,hf.data(),nf*sizeof(int),cudaMemcpyHostToDevice,stream));
        checked(ugkwp::enqueueGasStatusCheck(cells,nc,faces,nf,context,scratch,2509,23,stream));
        checked(cudaMemcpyAsync(&actual,context,sizeof(actual),cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
    };
    check();assert(!actual.fatalClaim); // Only unavailable storage is escalated.
    hc[301]=hc[257]=hf[3]=int(ugkwp::GasTransportCode::InvalidStorage);
    check();assert(actual.fatalClaim&&actual.firstFatal.entityKind==EntityKind::Cell);
    assert(actual.firstFatal.entityIndex==257&&actual.firstFatal.sourceSiteId==2509);
    assert(actual.firstFatal.operatorId==23&&actual.firstFatal.operationSequence==31);
    assert(actual.firstFatal.step==7&&actual.firstFatal.simulationTime==0.125);
    hc[301]=hc[257]=0;hf[1028]=int(ugkwp::GasTransportCode::InvalidStorage);
    check();assert(actual.firstFatal.entityKind==EntityKind::Face&&actual.firstFatal.entityIndex==3);
    recordFatal(&initial,makeRecord(Code::InvalidPhysicalState,2201,22,2,EntityKind::Cell,501));
    check();assert(actual.firstFatal.sourceSiteId==2201&&actual.firstFatal.entityIndex==501);
    assert(ugkwp::enqueueGasStatusCheck(nullptr,nc,faces,nf,context,scratch,2509,23,stream)==cudaErrorInvalidValue);
    reset(initial,32,42,8,0.25);
    checked(cudaMemcpyAsync(context,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream));
    checked(ugkwp::enqueueGasStatusCheck(nullptr,0,nullptr,0,context,scratch,2509,23,stream));
    checked(cudaMemcpyAsync(&actual,context,sizeof(actual),cudaMemcpyDeviceToHost,stream));
    checked(cudaStreamSynchronize(stream));assert(!actual.fatalClaim);
}
void checkProductionPipeline(cudaStream_t stream) {
    using namespace ugkp::diagnostics;
    DeviceAllocations memory;GpuOwner owner;checked(owner.initialise());
    DeviceState state{};state.nCells=257;state.nFaces=521;state.nInternalFaces=1;
    state.diagnosticOwner=&owner;state.diagnostics=owner.device();state.gasCaptureStream=stream;
    state.deviceState=memory.make<DeviceState>();state.gasStatusFirstUnavailable=memory.make<unsigned long long>();
    state.gasSpecies.cellStatus=memory.make<int>(state.nCells);
    state.gasSpecies.faceStatus=memory.make<int>(state.nFaces);
    state.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;state.gasSpeciesUploaded=true;
    state.hostGasTimeIntegrator=3;state.hostTurbulenceModel=3;
    state.hasPeriodicFaces=1;state.hostGravityActive=true;
    state.gasBoundaryLayer.enabled=true;state.gasBoundaryLayer.count=1;
    state.gasBoundaryLayerModel.config.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
    state.gasBoundaryLayerModel.config.nodes=8;state.gasBoundaryLayerModel.workspaceCount=1;
    state.gasBoundaryLayerModel.workspaceCapacity=24;
    auto* projected=memory.make<WallProjectionView>();
    state.gasBoundaryLayerModel.conservative.enabled=true;
    state.gasBoundaryLayerModel.conservative.projectedState=projected;
    state.gasBoundaryLayerModel.acceptedHistory=memory.make<double>(48);
    state.gasFluxPositivityScale=memory.make<double>(state.nCells);
    // All numerical arrays are deliberately null. Only diagnostic/status,
    // projected-view metadata, reduction input and a history sentinel are valid.
    std::vector<int> cells(state.nCells,0),faces(state.nFaces,0);cells[193]=1;
    std::vector<double> history(48,42),courant(state.nCells,19);
    checked(cudaMemcpy(state.gasSpecies.cellStatus,cells.data(),cells.size()*sizeof(int),cudaMemcpyHostToDevice));
    checked(cudaMemcpy(state.gasSpecies.faceStatus,faces.data(),faces.size()*sizeof(int),cudaMemcpyHostToDevice));
    checked(cudaMemcpy(state.gasBoundaryLayerModel.acceptedHistory,history.data(),history.size()*sizeof(double),cudaMemcpyHostToDevice));
    checked(cudaMemcpy(state.gasFluxPositivityScale,courant.data(),courant.size()*sizeof(double),cudaMemcpyHostToDevice));
    checked(cudaMemset(projected,0xa5,sizeof(*projected)));
    checked(cudaMemcpy(state.deviceState,&state,sizeof(state),cudaMemcpyHostToDevice));

    // Capturing the actual requested-interval protocol proves that stage
    // validation no longer blocks the host. Device checks gate the real kernels.
    cudaGraph_t graph{};cudaGraphExec_t executable{};
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    assert(SharedGasTrialPolicy::validate(&state)==0);
    const GasRequestedIntervalControls<double> controls{};
    assert(advanceGasRequestedInterval<SharedGasTrialPolicy>(&state,1e-6,0.,controls)==0);
    evaluateGasBoundaryLayerKernel<false><<<2,32,0,stream>>>(state.deviceState);
    evaluateGasBoundaryLayerTile8Kernel<false><<<2,32,0,stream>>>(state.deviceState);
    computeGasHllcAdcSensorKernel<<<3,128,0,stream>>>(state.deviceState);
    checked(cudaGetLastError());checked(cudaStreamEndCapture(stream,&graph));
    checked(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    checked(cudaGraphLaunch(executable,stream));assert(SharedGasTrialPolicy::complete(&state)==1);
    Snapshot snapshot;checked(owner.finish(snapshot));
    assert(snapshot.hasFatal&&snapshot.fatal.entityIndex==193&&snapshot.fatal.sourceSiteId==2509);
    WallProjectionView view{};checked(cudaMemcpy(&view,projected,sizeof(view),cudaMemcpyDeviceToHost));
    assert(view.diagnostics==state.diagnostics);
    std::vector<double> observed(48);checked(cudaMemcpy(observed.data(),state.gasBoundaryLayerModel.acceptedHistory,observed.size()*sizeof(double),cudaMemcpyDeviceToHost));
    assert(observed==history);

    // The real public API drains the same pipeline and rejects the interval;
    // the Courant API must not publish a stale reduction as a usable candidate.
    assert(ugkwpGpuResidentStrictAdvanceGasOnly(&state,1e-6,0.)==1);
    double candidate=-123.;assert(ugkwpGpuResidentStrictComputeGasCourant(&state,1e-6,0.,&candidate)==1);
    assert(candidate==-123.);
    checked(cudaGraphExecDestroy(executable));checked(cudaGraphDestroy(graph));
}
}
int main() {
    checked(cudaDeviceSetLimit(cudaLimitStackSize,65536));
    cudaStream_t stream{};checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    checkSelection(stream);checkProductionPipeline(stream);
    checked(cudaStreamDestroy(stream));std::cout<<"asynchronous gas completion PASS\n";
}
