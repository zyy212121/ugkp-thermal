#pragma once

namespace chmt {
namespace gas_window_detail {
template<class T> inline T* writable(const T* value){return const_cast<T*>(value);}
__global__ void reserve(GasWindowProgress* progress,std::size_t capacity){
    if(threadIdx.x||blockIdx.x)return;
    progress->batchFrames=0;progress->batchCapacity=capacity;
}
__global__ void selectGeometry(const GasWindowProgress* progress,const ResidentWallClock* clock,
    ResidentWallClock* geometryClock,bool selected,ugkp::diagnostics::DeviceContext* diagnostics,
    cudaGraphConditionalHandle condition){
    if(threadIdx.x||blockIdx.x)return;


    if(!selected)*geometryClock=*clock;
    cudaGraphSetConditional(condition,progress->stop==GasWindowStop::Running
        &&!ugkp::diagnostics::hasFatal(diagnostics)&&(!selected||progress->dt!=progress->reference));
}
inline WindowTime hostTime(ugkp::CompensatedTime value){
    WindowTime time(value.sum);time.add(value.correction);return time;
}
}

namespace gas_window_detail {
__global__ void selectRadiation(const GasWindowProgress* progress,const int* stage,int expected,
    const int* sourceStage,ugkp::diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle allowed){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(allowed,progress->stop==GasWindowStop::Complete
        &&*stage==expected&&(!sourceStage||*sourceStage==4)&&!ugkp::diagnostics::hasFatal(diagnostics));
}
__global__ void setRadiationStage(int* stage,int value){if(!threadIdx.x&&!blockIdx.x)*stage=value;}
}
static ugkp::gpu::ExecutionGraphNode appendGasWindowRadiation(Backend& b,ugkp::gpu::ExecutionGraph graph,
    ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.windowTransport)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    auto& radiation=*b.windowTransport;auto& material=*b.windowRadiationMaterial;
    const auto stream=b.gasCaptureStream;
    const auto branch=[&](cudaGraphNode_t before,int expected,const int* source,auto build){
        cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
        const auto selected=appendRecorded(graph,{before},stream,[&]{
            gas_window_detail::selectRadiation<<<1,1,0,stream>>>(b.windowProgress.data(),b.windowRadiationStage.data(),expected,
                source,b.diagnostics.device(),condition);
        });
        cudaGraph_t body=nullptr;const auto done=addIf(graph,condition,{selected},body);build(body);return done;
    };
    auto node=branch(dependency,0,nullptr,[&](cudaGraph_t body){
        auto response=ugkp::thermal::appendThermalRadiationInputs(b.particles.owner(),body,nullptr,stream);
        response=material.appendWindowStartTo(body,response);response=radiation.appendInputsTo(body,response);
        appendRecorded(body,{response},stream,[&]{gas_window_detail::setRadiationStage<<<1,1,0,stream>>>(b.windowRadiationStage.data(),1);});
    });
    node=branch(node,1,nullptr,[&](cudaGraph_t body){radiation.appendSourceTo(body);});
    node=branch(node,1,radiation.continuationStage(),[&](cudaGraph_t body){
        auto complete=radiation.appendDiagnosticsTo(body,nullptr);
        complete=material.appendWindowFinishTo(body,complete,radiation.deviceView());
        appendRecorded(body,{complete},stream,[&]{gas_window_detail::setRadiationStage<<<1,1,0,stream>>>(b.windowRadiationStage.data(),2);});
    });
    return branch(node,2,nullptr,[&](cudaGraph_t body){
        const auto heat=material.deviceView();
        const auto published=appendRecorded(body,{},stream,[&]{
            if(ugkp::thermal::enqueueThermalParticleEnergy(b.particles.owner(),heat,stream))throw std::runtime_error("publish CHMT particle radiation energy");
            const auto count=b.interfaceFaces.size();
            if(count)mapWindowRadiation<<<(count+255)/256,256,0,stream>>>(int(count),b.interfaceFaces.data(),b.nInternalFaces,heat.boundaryEnergy,b.windowRadiation.data());
        });
        const auto moments=ugkp::thermal::appendThermalParticleMoments(b.particles.owner(),body,published,stream);
        appendRecorded(body,{moments},stream,[&]{gas_window_detail::setRadiationStage<<<1,1,0,stream>>>(b.windowRadiationStage.data(),3);});
    });
}
bool prepareGasWindowRadiationClock(Backend& b,double elapsed,const double*& deviceElapsed,std::string& error){
    if(b.windowBatchPrepared||b.windowBatchPending||!b.particles.active()){
        error="prepare CHMT radiation before submitting the resident gas window";return false;
    }
    if(!std::isfinite(elapsed)||!(elapsed>0)){error="invalid CHMT radiation interval";return false;}
    if(!b.windowRadiationElapsed.resize(1,b.executionFault)||!b.windowRadiationStage.resize(1,b.executionFault)
        ||!b.windowRadiation.resize(b.interfaceFaces.size(),b.executionFault)
        ||!b.storage.write(b.windowRadiationElapsed.data(),std::vector<double>{elapsed})
        ||!b.storage.write(b.windowRadiationStage.data(),std::vector<int>{0})
        ||!b.particles.result(ugkp::thermal::prepareThermalParticleEnergy(b.particles.owner()),error)){
        if(error.empty())error=b.executionFault.message.empty()?b.storage.error():b.executionFault.message;return false;
    }
    deviceElapsed=b.windowRadiationElapsed.data();error.clear();return true;
}
bool bindGasWindowRadiation(Backend& b,ugkp::gpu::DiscreteOrdinateRadiation& radiation,
    ugkp::gpu::ParticleRadiationMaterial& material,std::string& error){
    const auto heat=material.deviceView();const int boundary=b.nFaces-b.nInternalFaces;
    if(!b.windowRadiationStage.data()||!b.windowRadiationElapsed.data()||b.windowBatchPrepared||b.windowBatchPending
        ||!heat.enabled||heat.cells!=b.nCells||heat.boundaryFaces!=boundary||(boundary&&!heat.boundaryEnergy)
        ||heat.diagnostics!=b.diagnostics.device()){
        error="CHMT parent radiation storage or shared diagnostic dependency is missing";return false;
    }
    for(int face:b.accepted.surface.gasFace)if(face>=0&&(face<b.nInternalFaces||face>=b.nFaces)){
        error="material radiation face is outside the gas boundary";return false;
    }
    b.windowTransport=&radiation;b.windowRadiationMaterial=&material;
    b.history.setExternalRadiationDevice({b.windowRadiation.data(),b.interfaceFaces.size(),b.gasCaptureStream});
    error.clear();return true;
}
static void orderGasRadiationInputs(Backend& b){
    if(!b.windowTransport)return;
    using ugkp::gpu::executionGraphDetail::checked;
    cudaEvent_t ready=nullptr;checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    try{
        for(void* source:{b.windowTransport->producerStream(),b.windowRadiationMaterial->producerStream()}){
            checked(cudaEventRecord(ready,static_cast<cudaStream_t>(source)));checked(cudaStreamWaitEvent(b.gasCaptureStream,ready,0));
        }
    }catch(...){cudaEventDestroy(ready);throw;}
    checked(cudaEventDestroy(ready));
}
static void acceptGasRadiationExecution(Backend& b,void* stream){
    if(!b.windowTransport)return;
    b.windowTransport->acceptExternalExecution(stream);b.windowRadiationMaterial->acceptExternalExecution(stream);
}

namespace gas_window_detail {
__global__ void selectMaterial(const GasWindowProgress* progress,const int* radiationStage,bool historyReady,
    ugkp::diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle condition){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(condition,progress->stop==GasWindowStop::Complete
        &&(historyReady||progress->microSteps==progress->batchFrames)&&(!radiationStage||*radiationStage==3)
        &&!ugkp::diagnostics::hasFatal(diagnostics));
}
__global__ void propagateMaterialFatal(const ugkp::diagnostics::DeviceContext* source,
    ugkp::diagnostics::DeviceContext* destination){
    if(!threadIdx.x&&!blockIdx.x&&ugkp::diagnostics::hasFatal(source))ugkp::diagnostics::recordFatal(destination,source->firstFatal);
}
}
static ugkp::gpu::ExecutionGraphNode appendGasWindowMaterial(Backend& b,ugkp::gpu::ExecutionGraph graph,
    ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.windowMaterial)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    const auto view=residentMaterialLocalDeviceView(*b.windowMaterial);const auto stream=b.gasCaptureStream;
    cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,{dependency},stream,[&]{
        gas_window_detail::selectMaterial<<<1,1,0,stream>>>(b.windowProgress.data(),
            b.windowTransport?b.windowRadiationStage.data():nullptr,b.windowMaterialHistoryReady,b.diagnostics.device(),condition);
    });
    cudaGraph_t body=nullptr;const auto completed=addIf(graph,condition,{selected},body);
    appendResidentMaterialLocalWindow(*b.windowMaterial,body,nullptr);
    return appendRecorded(graph,{completed},stream,[&]{gas_window_detail::propagateMaterialFatal<<<1,1,0,stream>>>(view.diagnostics,b.diagnostics.device());});
}
ResidentMaterialDriveHistory pendingGasWindowDriveHistory(const Backend& b){
    if(!b.windowBatchPrepared||b.windowBatchPending)throw std::logic_error("prepare gas storage before borrowing pending history");
    const auto frame=b.windowBatch.storage;
    return {frame.surface,frame.wall,frame.interval,b.windowBatch.capacity,b.interfaceFaces.size(),b.gasCaptureStream,&b.windowProgress.data()->batchFrames};
}
bool bindGasWindowMaterial(Backend& b,MaterialBackend& material,std::string& error){
    if(!b.windowBatchPrepared||b.windowBatchPending||b.windowMaterial){error="gas window has no unused material binding";return false;}
    const auto view=residentMaterialLocalDeviceView(material);
    if(!view.state||!view.report||!view.diagnostics){error="prepared material parent storage missing";return false;}
    if(!bindResidentMaterialEndpointGasFraction(material,b.storage.hostView().gasVolumeFraction,b.gasCaptureStream,error))return false;
    b.windowMaterial=&material;error.clear();return true;
}
bool gasWindowHasResidentRadiation(const Backend& b){return b.windowTransport!=nullptr;}
bool bindImplicitGasWindowMaterial(Backend& b,MaterialBackend& material,bool appendSolve,std::string& error){
    if((appendSolve&&!b.windowBatchPrepared)||b.windowBatchPending||b.windowMaterial||b.windowImplicit){
        error="gas window has no unused implicit material binding";return false;
    }
    if(!b.windowBatchPrepared)orderImplicitWindow(material,b.gasCaptureStream);
    b.windowImplicit=&material;b.windowImplicitSolve=appendSolve;error.clear();return true;
}
static void orderGasMaterialInputs(Backend& b){
    if(b.windowImplicit)orderImplicitWindow(*b.windowImplicit,b.gasCaptureStream);
    if(!b.windowMaterial)return;
    using ugkp::gpu::executionGraphDetail::checked;
    cudaEvent_t ready=nullptr;checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    try{checked(cudaEventRecord(ready,static_cast<cudaStream_t>(residentMaterialLocalDeviceView(*b.windowMaterial).producerStream)));
        checked(cudaStreamWaitEvent(b.gasCaptureStream,ready,0));}
    catch(...){cudaEventDestroy(ready);throw;}
    checked(cudaEventDestroy(ready));
}

static cudaGraphNode_t appendGasWindowGeometry(Backend& b,cudaGraph_t graph,cudaGraphNode_t dependency,bool selected){
    if(!b.movingWindow)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    const auto stream=b.gasCaptureStream;
    cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
    const auto chosen=appendRecorded(graph,{dependency},stream,[&]{
        gas_window_detail::selectGeometry<<<1,1,0,stream>>>(b.windowProgress.data(),b.wallClock.data(),
            b.windowGeometryClock.data(),selected,b.diagnostics.device(),condition);
    });
    cudaGraph_t body=nullptr;const auto complete=addIf(graph,condition,{chosen},body);
    appendRecorded(body,{},stream,[&]{
        const auto* interval=selected?&b.windowProgress.data()->dt:&b.windowProgress.data()->reference;
        if(!b.residentGeometry->prepare(DeviceValue<Real>{interval},b.interfaceKnotTime.data(),int(b.interfaceKnotTime.size()),
            b.windowGeometryClock.data(),b.interfaceFaces.data(),b.interfaceInputs.data(),b.storage,stream,b.error))
            throw std::runtime_error(b.error);
    });
    return complete;
}

namespace gas_window_detail {
__global__ void selectImplicitMaterial(const GasWindowProgress* progress,const int* radiationStage,
    const ugkp::diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle condition){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(condition,progress->stop==GasWindowStop::Complete
        &&(!radiationStage||*radiationStage==3)&&!ugkp::diagnostics::hasFatal(diagnostics));
}
}

static ugkp::gpu::ExecutionGraphNode appendGasWindowImplicitSolve(Backend& b,ugkp::gpu::ExecutionGraph graph,
    ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.windowImplicitSolve)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    auto& material=*b.windowImplicit;const auto stream=b.gasCaptureStream;
    const auto stage=b.windowTransport?b.windowRadiationStage.data():nullptr;
    cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,{dependency},stream,[&]{
        gas_window_detail::selectImplicitMaterial<<<1,1,0,stream>>>(b.windowProgress.data(),stage,b.diagnostics.device(),condition);
    });
    cudaGraph_t body=nullptr;const auto done=addIf(graph,condition,{selected},body);
    appendImplicitMaterialSolve(material,stage?b.windowRadiation.data():nullptr,body,nullptr,stream);
    return appendRecorded(graph,{done},stream,[&]{
        gas_window_detail::propagateMaterialFatal<<<1,1,0,stream>>>(implicitWindowDiagnostics(material),b.diagnostics.device());
    });
}

ugkp::gpu::ExecutionGraphNode appendGasWindowBatch(Backend& b,ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency){
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    if(!b.windowBatchPrepared)throw std::logic_error("prepare gas window storage before appending its graph");
    const auto stream=b.gasCaptureStream;const auto frame=b.windowBatch.storage;
    const int count=int(b.interfaceRecords.size());
    const int reductionBlock=256,parts=(b.nCells+reductionBlock-1)/reductionBlock;
    cudaGraphConditionalHandle loop,refresh,physical;
    for(auto handle:{&loop,&refresh,&physical})checked(cudaGraphConditionalHandleCreate(handle,graph,0,cudaGraphCondAssignDefault));
    const auto initialized=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        gas_window_detail::begin<<<1,1,0,stream>>>(b.deviceState,b.windowProgress.data(),b.windowControls.data(),loop);
    });
    cudaGraph_t body=nullptr;const auto completed=addWhile(graph,loop,{initialized},body);
    const auto requested=appendRecorded(body,{},stream,[&]{
        gas_window_detail::reference<<<1,1,0,stream>>>(b.deviceState,b.windowProgress.data(),b.windowControls.data(),
            b.wallClock.data(),b.interfaceKnotTime.data(),int(b.interfaceKnotTime.size()),refresh);
    });
    const auto referenceGeometry=appendGasWindowGeometry(b,body,requested,false);
    cudaGraph_t cfl=nullptr;const auto refreshed=addIf(body,refresh,{referenceGeometry},cfl);
    appendRecorded(cfl,{},stream,[&]{
        if(transport::computeGasCurrentCourantField(&b,Real(1),Real(0),b.fixedCellBlockThreads,b.fixedFaceBlockThreads,stream))
            throw std::runtime_error(transport::lastError);
        gas_window_detail::courantParts<<<parts,reductionBlock,reductionBlock*sizeof(GasWindowCourant),stream>>>(
            b.storage.hostView().gasFluxPositivityScale,b.gasSpecies.cellStatus,b.nCells,b.windowCourant.data()+(parts==1?parts:0));
        if(parts>1)gas_window_detail::courantFinish<<<1,1,0,stream>>>(b.windowCourant.data(),parts,b.windowCourant.data()+parts);
        if(!b.storage.checkStatus(b.error))throw std::runtime_error(b.error);
    });
    const auto selected=appendRecorded(body,{refreshed},stream,[&]{
        gas_window_detail::select<<<1,1,0,stream>>>(b.deviceState,b.windowProgress.data(),b.windowControls.data(),
            b.windowCourant.data()+parts,b.wallClock.data(),b.interfaceKnotTime.data(),int(b.interfaceKnotTime.size()),
            b.coupledInterval.data(),b.windowParticleInterval.data(),physical);
    });
    const auto selectedGeometry=appendGasWindowGeometry(b,body,selected,true);
    cudaGraph_t physicalBody=nullptr;const auto advanced=addIf(body,physical,{selectedGeometry},physicalBody);
    const int oldMode=b.gasGraphMode;b.gasGraphMode=0;b.gasStageIndex=0;b.preparedWallReady=false;
    ugkwp::invalidateGasBoundaryLayerPreparation(b);
    try{
        const auto gas=appendRecorded(physicalBody,{},stream,[&]{
            if(!b.storage.clearTrialStatus())throw std::runtime_error(b.storage.error());
            if(b.particles.active()&&ugkp::thermal::prepareParticleGasInterval(b.particles.owner(),stream))
                throw std::runtime_error("prepare bound particle contact geometry");
            const bool courantPrimitives=b.options.adjustTimeStep&&b.options.courantUpdateInterval==1
                &&!b.movingWindow&&!b.particles.active()&&!ugkwp::gasBoundaryLayerEnabled(b)
                &&b.gasSpecies.mode!=ugkwp::GasMode::MixtureChemistry;
            if(transport::advancePreparedGasTransport<transport::CoupledTrialPolicy>(&b,Real(1),Real(0),false,courantPrimitives)
                ||transport::CoupledTrialPolicy::validate(&b))throw std::runtime_error(b.error.empty()?transport::lastError:b.error);
            if(b.particles.active()&&count){const auto surface=b.particles.surface();
                publishParticleMaterialSurfaceKernel<<<(count+b.fixedFaceBlockThreads-1)/b.fixedFaceBlockThreads,b.fixedFaceBlockThreads,0,stream>>>(count,
                    b.interfaceFaces.data(),b.interfaceInputs.data(),b.interfaceWallProgram.data(),GasWallProgramPosition{},
                    gas_window_detail::writable(surface.temperature),gas_window_detail::writable(surface.velocityX),
                    gas_window_detail::writable(surface.velocityY),gas_window_detail::writable(surface.velocityZ),
                    gas_window_detail::writable(surface.effusivity),b.coupledInterval.data());
            }
        });
        auto particles=gas;
        if(b.movingWindow){
            particles=appendRecorded(physicalBody,{particles},stream,[&]{
                auto& geometry=*b.residentGeometry;
                if(!b.particles.mapGeometry(geometry.particleGeometry(b.storage,false),geometry.particleGeometry(b.storage,true),stream,b.error)
                    ||!geometry.finish(b.storage,stream,b.error)||!b.particles.refreshTrackingGeometry(b.storage,stream,b.error))
                    throw std::runtime_error(b.error);
            });
            if(b.particles.active())particles=ugkp::thermal::appendParticleGeometryMoments(b.particles.owner(),physicalBody,particles,stream);
        }
        if(b.particles.active())particles=ugkp::thermal::appendBoundParticlePhase(b.particles.owner(),physicalBody,particles,
            b.windowParticleInterval.data(),stream);
        appendRecorded(physicalBody,{particles},stream,[&]{
            if(transport::finaliseGasTrialSources<transport::CoupledTrialPolicy>(&b,Real(1))
                ||transport::finaliseGasBoundaryStage(&b,Real(1),Real(1))
                ||transport::CoupledTrialPolicy::validate(&b))throw std::runtime_error(b.error.empty()?transport::lastError:b.error);
            enqueueCoupledSurfaceArchive(b.interfaceRecords.data(),count,b.interfaceFaces.data(),b.particles.wallExchange(),
                {frame.surface,frame.wall,&b.windowProgress.data()->batchFrames},b.interfaceFailure.data(),
                int(b.accepted.solid.size()),int(b.accepted.film.size()),b.diagnostics.device(),b.fixedFaceBlockThreads,stream);
            const auto& model=b.storage.hostView().gasBoundaryLayerModel;
            if(model.acceptedHistory&&b.gasBoundaryLayer.count)
                ::commitAcceptedWallHistory<<<(b.gasBoundaryLayer.count+b.fixedFaceBlockThreads-1)/b.fixedFaceBlockThreads,b.fixedFaceBlockThreads,0,stream>>>(b.deviceState);
        });
    }catch(...){b.gasGraphMode=oldMode;b.preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(b);throw;}
    b.gasGraphMode=oldMode;b.preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(b);
    appendRecorded(body,{advanced},stream,[&]{
        gas_window_detail::accept<<<1,1,0,stream>>>(b.deviceState,b.windowProgress.data(),b.windowControls.data(),
            b.coupledInterval.data(),frame.interval,loop);
    });
    auto exchanged=completed;
    if(b.windowImplicit){
        exchanged=appendImplicitMaterialExchange(*b.windowImplicit,ResidentMaterialDriveHistory{frame.surface,frame.wall,frame.interval,b.windowBatch.capacity,b.interfaceFaces.size(),
                stream,&b.windowProgress.data()->batchFrames},b.windowProgress.data(),
            b.particles.active(),graph,exchanged,stream);
        exchanged=appendRecorded(graph,{exchanged},stream,[&]{
            gas_window_detail::propagateMaterialFatal<<<1,1,0,stream>>>(implicitWindowDiagnostics(*b.windowImplicit),b.diagnostics.device());
        });
    }
    const auto radiated=appendGasWindowRadiation(b,graph,exchanged);
    const auto material=b.windowImplicit?appendGasWindowImplicitSolve(b,graph,radiated):appendGasWindowMaterial(b,graph,radiated);
    if(!b.particles.active())return material;
    return appendRecorded(graph,{material},stream,[&]{
        if(ugkp::thermal::enqueueBoundParticleMirror(b.particles.owner(),b.windowParticleMirror.data(),stream))
            throw std::runtime_error("failed to queue bound particle mirror");
    });
}
static void buildGasWindowBatch(Backend& b,Backend::WindowGraph& graph){
    using ugkp::gpu::executionGraphDetail::checked;
    Backend::WindowGraph next;
    checked(cudaGraphCreate(&next.graph,0));
    appendGasWindowBatch(b,next.graph,nullptr);
    bool updated=false;
    if(graph.executable){
        cudaGraphExecUpdateResultInfo info{};
        const auto status=cudaGraphExecUpdate(graph.executable,next.graph,&info);
        if(status==cudaSuccess)updated=info.result==cudaGraphExecUpdateSuccess;
        else if(status==cudaErrorGraphExecUpdateFailure){
            if(cudaPeekAtLastError()==status)(void)cudaGetLastError();
        }else checked(status);
    }
    if(!updated){
        checked(cudaGraphInstantiateWithFlags(&next.executable,next.graph,0));
        std::swap(graph.executable,next.executable);
    }
    std::swap(graph.graph,next.graph);
    graph.storage=b.windowBatch.storage.interval;
    graph.radiationOwner=b.windowTransport;
    graph.radiationRevision=b.windowTransport?b.windowTransport->executionRevision():0;
    graph.implicitOwner=b.windowImplicit;graph.implicitSolve=b.windowImplicitSolve;
    graph.implicitRevision=b.windowImplicit?implicitWindowRevision(*b.windowImplicit):0;
}

static bool submitGasWindowBatch(Backend& b,std::string& error){
    using ugkp::gpu::executionGraphDetail::checked;
    const auto slot=(b.windowBatch.first/CoupledHistoryArchive::slabFrames)%2;
    auto& graph=b.windowGraph[slot];
    if(!graph.executable||graph.storage!=b.windowBatch.storage.interval||graph.radiationOwner!=b.windowTransport
        ||graph.radiationRevision!=(b.windowTransport?b.windowTransport->executionRevision():0)
        ||graph.implicitOwner!=b.windowImplicit||graph.implicitSolve!=b.windowImplicitSolve
        ||graph.implicitRevision!=(b.windowImplicit?implicitWindowRevision(*b.windowImplicit):0))buildGasWindowBatch(b,graph);
    orderGasRadiationInputs(b);orderGasMaterialInputs(b);checked(cudaGraphLaunch(graph.executable,b.gasCaptureStream));
    acceptGasRadiationExecution(b,b.gasCaptureStream);
    if(b.windowImplicit)acceptImplicitWindowExecution(*b.windowImplicit,b.gasCaptureStream);
    if(b.windowMaterial)acceptResidentMaterialLocalExecution(*b.windowMaterial,b.gasCaptureStream);
    b.windowBatchPending=true;return true;
}

bool prepareGasWindow(Backend& b,Real cap,std::string& error){
    if(b.windowBatchPending||b.windowBatchPrepared){error="gas window has already been submitted";return false;}
    if(b.poisoned){error="gas window has an earlier fatal device error";return false;}
    try{
        if(b.movingWindow){b.residentGeometry->bind(b.storage);b.gasGeometry=b.storage.hostView().gasGeometry;}
        else if(prepareCoupledGeometry(&b)){error=b.error;return false;}
        const auto count=b.interfaceRecords.size();
        if(!b.historyArchive)b.historyArchive=std::make_shared<CoupledHistoryArchive>(count,b.gasBoundaryLayer.enabled,&b.historyTransfers);
        GasWindowControls controls;
        controls.cap=cap;controls.maximum=b.model.physics.maxDt;controls.fixedDt=Real(b.options.fixedDeltaT);
        controls.cfl=b.model.physics.cfl;controls.growth=Real(b.options.maximumTimeStepGrowth);
        controls.moving=b.movingWindow;controls.adaptive=b.options.adjustTimeStep;controls.courantPeriod=b.options.courantUpdateInterval;
        controls.total=residentWindowTime(intervalDuration(b.program.interval));
        controls.gasGeometry=b.trial.gasMesh.geometryVersion;controls.solidGeometry=b.trial.solidMesh.geometryVersion;
        controls.acceptedStep=b.trial.acceptedSteps;
        GasWindowProgress progress;progress.elapsed=residentWindowTime(b.elapsed);progress.gasSteps=b.gasSteps;
        progress.microSteps=b.history.records().size();progress.seed=b.nextGasSeed;
        progress.rate=b.cachedCourantRate;progress.courantReady=b.courantReady;
        b.windowBatch=b.historyArchive->reserveBatch(b.gasCaptureStream);
        progress.batchCapacity=b.windowBatch.capacity;
        if(!b.windowProgress.resize(1,b.executionFault)||!b.windowControls.resize(1,b.executionFault)
            ||!b.windowGeometryClock.resize(1,b.executionFault)
            ||!b.windowCourant.resize((b.nCells+255)/256+1,b.executionFault)
            ||!b.windowParticleInterval.resize(1,b.executionFault)||!b.windowParticleMirror.resize(1,b.executionFault)
            ||!b.storage.write(b.windowControls.data(),std::vector<GasWindowControls>{controls})
            ||!b.storage.write(b.windowProgress.data(),std::vector<GasWindowProgress>{progress})){
            error=b.executionFault.message.empty()?b.storage.error():b.executionFault.message;return false;
        }
        b.windowBatchPrepared=true;b.windowBatchRetired=false;
        error.clear();return true;
    }catch(const std::exception& exception){b.poisoned=true;error=exception.what();return false;}
}

bool enqueuePreparedGasWindow(Backend& b,std::string& error){
    if(!b.windowBatchPrepared||b.windowBatchPending){error="gas window has no unsubmitted preparation";return false;}
    try{return submitGasWindowBatch(b,error);}
    catch(const std::exception& exception){b.poisoned=true;error=exception.what();return false;}
}
bool enqueueGasWindow(Backend& b,Real cap,std::string& error){
    return prepareGasWindow(b,cap,error)&&enqueuePreparedGasWindow(b,error);
}
GasWindowDeviceView gasWindowDeviceView(const Backend& b){return {b.windowProgress.data(),b.gasCaptureStream};}
bool acceptGasWindowExecution(Backend& b,void* producer,std::string& error){
    if(!b.windowBatchPrepared||b.windowBatchPending){error="gas window has no unsubmitted preparation";return false;}
    cudaEvent_t ready=nullptr;
    try{
        using ugkp::gpu::executionGraphDetail::checked;
        checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
        checked(cudaEventRecord(ready,static_cast<cudaStream_t>(producer)));
        checked(cudaStreamWaitEvent(b.gasCaptureStream,ready,0));
        checked(cudaEventDestroy(ready));ready=nullptr;
        acceptGasRadiationExecution(b,producer);
        if(b.windowImplicit)acceptImplicitWindowExecution(*b.windowImplicit,producer);
        if(b.windowMaterial)acceptResidentMaterialLocalExecution(*b.windowMaterial,producer);
        b.windowBatchPending=true;error.clear();return true;
    }catch(const std::exception& exception){if(ready)cudaEventDestroy(ready);b.poisoned=true;error=exception.what();return false;}
}

static bool retireGasWindowBatch(Backend& b,const GasWindowProgress& progress,
    const std::vector<ResidentMaterialDriveInterval>& intervals,const std::vector<ugkp::thermal::BoundParticleMirror>& mirror,
    std::string& error){
    if(b.windowBatchRetired)return true;
    if(progress.batchFrames>b.windowBatch.capacity||progress.stop==GasWindowStop::Fatal){
        b.poisoned=true;error="gas window batch has an invalid completion state";return false;
    }
            if(b.particles.active()&&ugkp::thermal::acceptBoundParticleMirror(b.particles.owner(),mirror.front())){
                b.poisoned=true;error="cannot accept completed particle buffer roles";return false;
            }
            for(std::size_t i=0;i<progress.batchFrames;++i){
                const auto& item=intervals[i];GasIntervalRecord record;
                record.microSequence=b.history.records().size()+1;
                record.localBegin=gas_window_detail::hostTime(item.begin);record.localEnd=gas_window_detail::hostTime(item.end);
                record.begin=windowAbsolute(b.program.interval.begin,record.localBegin);
                record.end=windowAbsolute(b.program.interval.begin,record.localEnd);record.actualDt=item.actualDt;
                if(b.movingWindow){++b.trial.gasMesh.geometryVersion;++b.trial.solidMesh.geometryVersion;}
                record.gasGeometry=b.trial.gasMesh.geometryVersion;record.solidGeometry=b.trial.solidMesh.geometryVersion;
                const Real stage=b.hostGasTimeIntegrator==1?Real(0):b.hostGasTimeIntegrator==2?item.actualDt:Real(.5)*item.actualDt;
                record.gasTraceTime=windowAbsolute(b.program.interval.begin,record.localBegin+stage);
                if(!b.history.appendAccepted(record,error)){b.poisoned=true;return false;}
                b.record=record;
            }
            b.historyArchive->acceptedBatch(progress.batchFrames,b.gasCaptureStream);
            b.historyFrameCount+=progress.batchFrames;b.elapsed=gas_window_detail::hostTime(progress.elapsed);
            b.gasSteps=progress.gasSteps;b.nextGasSeed=progress.seed;b.cachedCourantRate=progress.rate;b.courantReady=progress.courantReady;
            b.microDt=progress.dt;b.microTime=b.record.begin;b.trial.time=b.record.end;
            b.trial.lastAcceptedDt=progress.dt;b.trial.nextDt=progress.seed;b.stageTime=b.record.gasTraceTime;
            if(b.gasBoundaryLayer.enabled){b.wallDiagnosticsReady=true;b.wallDiagnosticTime=b.stageTime;}
    b.windowBatchRetired=true;return true;
}

bool collectGasWindow(Backend& b,GasMicroReport& report,std::string& error){
    if(!b.windowBatchPending){error="no submitted gas window to collect";return false;}
    try{
        for(;;){
            std::vector<GasWindowProgress> result;std::vector<ResidentMaterialDriveInterval> intervals;
            std::vector<ugkp::thermal::BoundParticleMirror> mirror;
            std::vector<int> radiationStage,materialStage;
            std::vector<MaterialLocalProgress> materialResult;
            const auto material=b.windowMaterial?residentMaterialLocalDeviceView(*b.windowMaterial):MaterialLocalDeviceView{};
            if(!b.storage.enqueueRead(b.windowProgress.data(),1,result)
                ||!b.storage.enqueueRead(b.windowBatch.storage.interval,b.windowBatch.capacity,intervals)
                ||(b.particles.active()&&!b.storage.enqueueRead(b.windowParticleMirror.data(),1,mirror))
                ||(b.windowTransport&&!b.storage.enqueueRead(b.windowRadiationStage.data(),1,radiationStage))
                ||(b.windowMaterial&&!b.storage.enqueueRead(material.report,1,materialResult))
                ||(material.windowStage&&!b.storage.enqueueRead(material.windowStage,1,materialStage))
                ||!b.storage.observeDiagnostics()){
                b.poisoned=true;error=b.storage.error();return false;
            }
            const auto& progress=result.front();
            if(b.windowTransport&&progress.stop==GasWindowStop::Complete){
                if(radiationStage.front()!=3){
                    if(!b.windowTransport->servicePendingCapacity()){
                        b.poisoned=true;error="CHMT radiation stopped without completion or a capacity request";return false;
                    }


                    const auto slot=(b.windowBatch.first/CoupledHistoryArchive::slabFrames)%2;
                    auto& graph=b.windowGraph[slot];buildGasWindowBatch(b,graph);
                    orderGasRadiationInputs(b);orderGasMaterialInputs(b);
                    ugkp::gpu::executionGraphDetail::checked(cudaGraphLaunch(graph.executable,b.gasCaptureStream));
                    acceptGasRadiationExecution(b,b.gasCaptureStream);
                    if(b.windowImplicit)acceptImplicitWindowExecution(*b.windowImplicit,b.gasCaptureStream);
                    if(b.windowMaterial)acceptResidentMaterialLocalExecution(*b.windowMaterial,b.gasCaptureStream);continue;
                }
            }
            if(b.windowMaterial&&progress.stop==GasWindowStop::Complete&&!b.windowMaterialHistoryReady
                &&progress.microSteps>progress.batchFrames){
                if(!retireGasWindowBatch(b,progress,intervals,mirror,error))return false;
                b.historyArchive->seal(b.gasCaptureStream);



                for(auto& graph:b.windowGraph)graph.clear();
                if(!completePendingMaterialHistory(*b.windowMaterial,b.history,b.historyArchive,error))return false;
                b.windowMaterialHistoryReady=true;
                const auto slot=(b.windowBatch.first/CoupledHistoryArchive::slabFrames)%2;
                auto& graph=b.windowGraph[slot];buildGasWindowBatch(b,graph);
                orderGasRadiationInputs(b);orderGasMaterialInputs(b);
                ugkp::gpu::executionGraphDetail::checked(cudaGraphLaunch(graph.executable,b.gasCaptureStream));
                acceptGasRadiationExecution(b,b.gasCaptureStream);
                acceptResidentMaterialLocalExecution(*b.windowMaterial,b.gasCaptureStream);continue;
            }
            if(b.windowMaterial&&progress.stop==GasWindowStop::Complete){
                if(!materialResult.front().complete||(material.windowStage&&materialStage.front()!=4)){
                    if(!serviceResidentMaterialLocalCapacity(*b.windowMaterial,error)){
                        b.poisoned=true;return false;
                    }
                    const auto slot=(b.windowBatch.first/CoupledHistoryArchive::slabFrames)%2;
                    auto& graph=b.windowGraph[slot];buildGasWindowBatch(b,graph);
                    orderGasRadiationInputs(b);orderGasMaterialInputs(b);
                    ugkp::gpu::executionGraphDetail::checked(cudaGraphLaunch(graph.executable,b.gasCaptureStream));
                    acceptGasRadiationExecution(b,b.gasCaptureStream);
                    acceptResidentMaterialLocalExecution(*b.windowMaterial,b.gasCaptureStream);continue;
                }
            }
            b.windowBatchPending=false;
            if(progress.batchFrames>b.windowBatch.capacity||progress.stop==GasWindowStop::Fatal){
                b.poisoned=true;error="gas window batch has an invalid completion state";return false;
            }
            if(!retireGasWindowBatch(b,progress,intervals,mirror,error))return false;
            if(progress.stop==GasWindowStop::Complete){


                if(b.windowRadiationMaterial)b.windowRadiationMaterial->acceptExternalCompletion({});
                report={};report.acceptedDt=progress.dt;report.nextGasDt=progress.seed;report.time=b.trial.time;
                report.localTime=b.elapsed;report.microSequence=progress.microSteps;report.recoverable=false;
                b.windowBatchPrepared=false;error.clear();return true;
            }
            if(progress.stop!=GasWindowStop::HistoryCapacity||!progress.batchFrames){
                b.poisoned=true;error="gas window stopped without completion or a storage request";return false;
            }


            b.windowBatch=b.historyArchive->reserveBatch(b.gasCaptureStream);
            b.windowBatchRetired=false;
            gas_window_detail::reserve<<<1,1,0,b.gasCaptureStream>>>(b.windowProgress.data(),b.windowBatch.capacity);
            ugkp::gpu::executionGraphDetail::checked(cudaGetLastError());
            if(!submitGasWindowBatch(b,error))return false;
        }
    }catch(const std::exception& exception){b.poisoned=true;error=exception.what();return false;}
}
}
