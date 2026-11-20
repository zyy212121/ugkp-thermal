#include "gpu/ImplicitMaterialExchange.cuh"
#include "gpu/ImplicitMaterialIteration.cuh"

namespace implicitWindowGpu {
bool configureColors(MaterialBackend& b,const HostMesh& m){
    auto& s=b.implicit;
    std::vector<Face> faces;
    std::vector<std::vector<int>> incidence(b.cells),colors,used(b.cells);
    if(s.transport)for(int f=0;f<b.faces;++f){
        const auto kind=m.boundaryKind[f];int right=m.neighbour[f];
        const int left=m.owner[f];
        if(kind==BoundaryKind::Periodic){
            const int pair=m.periodicPartner[f];
            if(f>=pair)continue;
            right=m.owner[pair];
        }
        if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty||right==left
            ||(right<0&&!s.hasSweepStorage))continue;
        const int index=faces.size();
        faces.push_back({left,right,f});incidence[left].push_back(index);
        if(right>=0)incidence[right].push_back(index);
        const auto inUse=[&](int cell,int color){
            return cell>=0&&std::find(used[cell].begin(),used[cell].end(),color)!=used[cell].end();
        };
        int color=0;
        while(inUse(left,color)||inUse(right,color))++color;
        if(color==int(colors.size()))colors.emplace_back();
        colors[color].push_back(index);used[left].push_back(color);
        if(right>=0)used[right].push_back(color);
    }
    std::vector<int> offsets{0},incident,ordered;
    for(const auto& row:incidence){
        incident.insert(incident.end(),row.begin(),row.end());offsets.push_back(incident.size());
    }
    s.colorOffsets.assign(1,0);
    for(const auto& color:colors){
        ordered.insert(ordered.end(),color.begin(),color.end());s.colorOffsets.push_back(ordered.size());
    }
    return b.upload(s.face,faces)&&b.upload(s.incidentOffsets,offsets)
        &&b.upload(s.incident,incident)&&b.upload(s.orderedFaces,ordered);
}

IterationView view(MaterialBackend& b){
    auto& s=b.implicit;auto transport=b.workspace();transport.splitFlux=s.splitFlux.data();
    return {b.mesh(),transport,b.devicePhysics.data(),b.initial.data(),b.next.data(),b.initialVolumes.data(),s.swept.data(),
        s.nativeVolume.data(),s.exchange.data(),s.incidentOffsets.data(),s.incident.data(),s.orderedFaces.data(),s.face.data(),
        int(s.face.size()),s.fixedMass.data(),s.mass.data(),s.extents.data(),s.fixedEnergy.data(),s.target.data(),s.conducted.data(),
        s.temperature.data(),s.caloric.data(),s.thermal.data(),s.capacity.data(),s.source.data(),s.conductivity.data(),
        s.cellMetrics.data(),s.faceMetrics.data(),s.controls.data(),s.report.data(),b.diagnostics.device()};
}
}

bool implicitWindowMeshConfigured(const MaterialBackend& b,const HostMesh& mesh,const std::vector<Real>& swept){
    const auto& s=b.implicit;
    return s.diffusion&&b.meshGeometry.matches(mesh)&&s.sweepValues==swept&&s.temperature.size()==mesh.volumes.size();
}

bool configureImplicitWindowMesh(MaterialBackend& b,const HostMesh& mesh,const std::vector<Real>& swept,
    const std::vector<Real>& nativeVolumes,std::string& error){
    auto& s=b.implicit;
    if((!swept.empty()&&swept.size()!=mesh.owner.size())||nativeVolumes.size()!=mesh.volumes.size()){
        error="implicit material endpoint geometry extent mismatch";return false;
    }
    const bool topologyChanged=s.colorTopology!=mesh.topologyHash||s.colorCells!=int(mesh.volumes.size())
        ||s.colorFaces!=int(mesh.owner.size())||!b.meshGeometry.topologyMatches(mesh);
    if(!b.configureMesh(mesh,error))return false;
    const bool darcy=b.physics.permeability>0;
    const bool diffusion=std::any_of(std::begin(b.physics.poreEffectiveDiffusivity),std::end(b.physics.poreEffectiveDiffusivity),
        [](PhysicalReal x){return x>0;});
    const bool ale=std::any_of(swept.begin(),swept.end(),[](Real x){return x!=0;});
    const bool transport=darcy||diffusion||ale;
    const auto cells=b.cells;
    for(const auto value:nativeVolumes)if(!finite(PhysicalReal(value))||PhysicalReal(value)<=0){
        error="invalid implicit material native cell volume";return false;
    }
    const bool hasSweepStorage=!swept.empty();
    if(!s.diffusion||topologyChanged||transport!=s.transport||hasSweepStorage!=s.hasSweepStorage){
        s.transport=transport;s.hasSweepStorage=hasSweepStorage;
        if(!implicitWindowGpu::configureColors(b,mesh))return b.failure(error);
        s.colorTopology=mesh.topologyHash;s.colorCells=b.cells;s.colorFaces=b.faces;
    }
    if(s.sweepValues!=swept){
        if(!b.upload(s.swept,swept))return b.failure(error);
        s.sweepValues=swept;
    }
    s.darcy=darcy;s.diffusionActive=diffusion;
    const std::vector<PhysicalReal> volumes(nativeVolumes.begin(),nativeVolumes.end());
    const int threads=ugkp::gpu::workPlan().materialBlockThreads;
    const int count=std::max(cells,int(s.face.size()));
    if(!b.upload(s.nativeVolume,volumes)||!b.allocate(s.fixedMass,cells)||!b.allocate(s.mass,cells)
        ||!b.allocate(s.extents,cells)||!b.allocate(s.fixedEnergy,cells)||!b.allocate(s.target,cells)
        ||!b.allocate(s.conducted,cells)||!b.allocate(s.temperature,cells)||!b.allocate(s.caloric,cells)
        ||!b.allocate(s.thermal,cells)||!b.allocate(s.splitFlux,transport?b.faces:0)
        ||!b.allocate(s.capacity,cells)||!s.capacity.zero(b.stream.get())
        ||!b.allocate(s.source,cells)||!s.source.zero(b.stream.get())
        ||!b.allocate(s.conductivity,cells)||!s.conductivity.zero(b.stream.get())
        ||!b.allocate(s.cellMetrics,cells)||!b.allocate(s.faceMetrics,s.face.size())
        ||!b.allocate(s.parts,implicitWindowGpu::blocks(count,threads)+1)||!b.allocate(s.linear,1))
        return b.failure(error);
    error.clear();return true;
}

ImplicitWindowThermalFields implicitWindowThermalFields(MaterialBackend& b){
    auto& s=b.implicit;return {s.capacity.data(),s.source.data(),s.conductivity.data(),b.stream.get()};
}
bool configureImplicitWindowConduction(MaterialBackend& b,ugkp::gpu::ImplicitDiffusion& diffusion,
    ugkp::gpu::SparseConjugateGradient& sparse,std::string& error){
    auto& s=b.implicit;
    s.diffusion=&diffusion;s.sparse=&sparse;
    if(!b.waitConduction(diffusion.deviceValues().readyStream))return b.failure(error);
    auto signature=diffusion.residentExecutionSignature();signature.append(sparse.residentExecutionSignature());
    signature.add(b.cells);signature.add(s.transport);signature.add(s.darcy);signature.add(s.diffusionActive);
    signature.add(b.physics.enableReactions);
    signature.add(b.initial.data());signature.add(b.next.data());signature.add(b.initialVolumes.data());
    signature.add(b.devicePhysics.data());signature.add(s.controls.data());signature.add(s.exchange.data());
    signature.add(s.face.data());signature.add(s.face.size());signature.add(s.swept.data());
    signature.add(s.incident.data());signature.add(s.incidentOffsets.data());signature.add(s.orderedFaces.data());
    for(const int offset:s.colorOffsets)signature.add(offset);
    const auto m=b.mesh();const auto w=b.workspace();
    signature.add(m.volumes);signature.add(m.cellCentres);signature.add(m.owner);signature.add(m.neighbour);
    signature.add(m.boundaryKind);signature.add(m.periodicPartner);signature.add(m.areaVectors);signature.add(m.faceCentres);
    signature.add(m.cellFaceOffsets);signature.add(m.cellFaces);signature.add(m.cellFaceSigns);
    signature.add(w.primitive);signature.add(w.pressureGradient);signature.add(w.poreVelocity);
    signature.add(w.gradientSensitivity);signature.add(w.faceFlux);signature.add(w.faceDiffusionLossLeft);signature.add(w.faceDiffusionLossRight);
    signature.add(s.nativeVolume.data());signature.add(s.fixedMass.data());signature.add(s.mass.data());signature.add(s.extents.data());
    signature.add(s.fixedEnergy.data());signature.add(s.target.data());signature.add(s.conducted.data());signature.add(s.temperature.data());
    signature.add(s.caloric.data());signature.add(s.thermal.data());signature.add(s.splitFlux.data());signature.add(s.capacity.data());
    signature.add(s.source.data());signature.add(s.conductivity.data());signature.add(s.cellMetrics.data());signature.add(s.faceMetrics.data());
    signature.add(s.parts.data());signature.add(s.linear.data());signature.add(s.report.data());
    if(!(signature==s.signature)){s.signature=std::move(signature);++s.revision;}
    error.clear();return true;
}

ugkp::gpu::ExecutionGraphNode appendImplicitMaterialSolve(MaterialBackend& b,const EnergyReal* radiation,
    ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency,void* executionStream){
    using namespace ugkp::gpu;using namespace executionGraphDetail;using namespace implicitWindowGpu;
    auto& s=b.implicit;auto& diffusion=*s.diffusion;auto& sparse=*s.sparse;
    const auto stream=static_cast<cudaStream_t>(executionStream);
    const auto v=view(b);
    const int threads=workPlan().materialBlockThreads,owners=s.surfaceCells.size();
    const int cells=blocks(b.cells,threads),faces=blocks(s.face.size(),threads),parts=std::max(cells,faces);
    const auto total=s.parts.data()+parts;
    const std::size_t shared=threads*sizeof(ImplicitWindowMetrics);
    cudaGraphConditionalHandle loop,prepare,solve,recover;
    checked(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
    const auto initialized=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        if(radiation&&owners)addExternalRadiation<<<blocks(owners,threads),threads,0,stream>>>(radiation,s.surfaceCells.data(),
            owners,s.surfaceOffsets.data(),s.surfaceFaces.data(),s.receivers.data(),s.exchange.data(),v.report,v.diagnostics);
        startCells<<<cells,threads,0,stream>>>(v);
        if(faces)clearFaces<<<faces,threads,0,stream>>>(v.faces,v.faceCount,v.report);
        startLoop<<<1,1,0,stream>>>(v.report,v.diagnostics,loop);
    });
    cudaGraph_t iteration=nullptr;const auto complete=addWhile(graph,loop,{initialized},iteration);
    checked(cudaGraphConditionalHandleCreate(&prepare,iteration,0,cudaGraphCondAssignDefault));
    const auto preparing=appendRecorded(iteration,{},stream,[&]{
        selectIteration<<<1,1,0,stream>>>(v.report,v.diagnostics,true,prepare);
    });
    cudaGraph_t preparation=nullptr;const auto prepared=addIf(iteration,prepare,{preparing},preparation);
    const auto physical=appendRecorded(preparation,{},stream,[&]{
        const bool reactions=b.physics.enableReactions;
        reconstructCells<<<cells,threads,0,stream>>>(v,s.transport,reactions);
        if(s.transport){
            prepareState<<<cells,threads,0,stream>>>(v);
            porePressure<<<cells,threads,0,stream>>>(v);
            if(faces)transportRates<<<faces,threads,0,stream>>>(v);
            for(std::size_t i=1;i<s.colorOffsets.size();++i){
                const int first=s.colorOffsets[i-1],count=s.colorOffsets[i]-first;
                massColor<<<blocks(count,threads),threads,0,stream>>>(v,first,count,s.darcy,s.diffusionActive);
            }
        }
        if(reactions)reactCells<<<cells,threads,0,stream>>>(v);
        prepareState<<<cells,threads,0,stream>>>(v);
        thermalCells<<<cells,threads,0,stream>>>(v,s.transport);
        for(std::size_t i=1;i<s.colorOffsets.size();++i){
            const int first=s.colorOffsets[i-1],count=s.colorOffsets[i]-first;
            energyColor<<<blocks(count,threads),threads,0,stream>>>(v,first,count);
        }
        coefficients<<<cells,threads,0,stream>>>(v,diffusion.deviceSolution());
        diffusion.enqueueResidentAssembly(stream,&s.controls.data()->interval);
    });
    const auto residual=sparse.appendResidentLocalResidual(preparation,physical,diffusion.deviceSolution(),diffusion.deviceResidual());
    appendRecorded(preparation,{residual},stream,[&]{
        linearParts<<<cells,threads,shared,stream>>>(v,diffusion.deviceResidual(),s.parts.data());
        prepareLinear<<<1,threads,shared,stream>>>(v,s.parts.data(),cells,total,s.linear.data());
    });
    checked(cudaGraphConditionalHandleCreate(&solve,iteration,0,cudaGraphCondAssignDefault));
    const auto solving=appendRecorded(iteration,{prepared},stream,[&]{
        selectIteration<<<1,1,0,stream>>>(v.report,v.diagnostics,false,solve);
    });
    cudaGraph_t solveBody=nullptr;const auto solved=addIf(iteration,solve,{solving},solveBody);
    sparse.appendResidentLocalLdu(solveBody,nullptr,diffusion.deviceSolution(),s.linear.data());
    checked(cudaGraphConditionalHandleCreate(&recover,iteration,0,cudaGraphCondAssignDefault));
    const auto recovery=appendRecorded(iteration,{solved},stream,[&]{
        selectRecovery<<<1,1,0,stream>>>(v.report,sparse.capacityPendingDevice(),v.diagnostics,recover);
    });
    cudaGraph_t recoverBody=nullptr;const auto recovered=addIf(iteration,recover,{recovery},recoverBody);
    appendRecorded(recoverBody,{},stream,[&]{
        diffusion.enqueueResidentFlux(stream,false);
        recoverConservativeEnergy<<<cells,threads,0,stream>>>(v,diffusion.deviceSolution(),diffusion.deviceBoundaryFlux());
        if(faces)checkEnergyFaces<<<faces,threads,0,stream>>>(v);
        resultParts<<<parts,threads,shared,stream>>>(v,s.parts.data());
        finishIteration<<<1,threads,shared,stream>>>(v,s.parts.data(),parts,total,sparse.deviceReport());
    });
    appendRecorded(iteration,{recovered},stream,[&]{
        continueLoop<<<1,1,0,stream>>>(v.report,sparse.capacityPendingDevice(),v.diagnostics,loop);
    });
    return complete;
}

namespace {
bool readImplicitWindowReport(MaterialBackend& b,ImplicitWindowReport& report,std::string& error){
    if(!b.transfers.download(&report,b.implicit.report.data(),1))return b.failure(error);
    return b.finish(error);
}
bool enqueueImplicitContinuation(MaterialBackend& b,const EnergyReal* radiation,std::string& error){
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    auto& s=b.implicit;
    auto signature=s.signature;signature.add(radiation);
    if(!s.continuation||!(signature==s.continuationSignature)){
        cudaGraph_t graph=nullptr;
        try{
            checked(cudaGraphCreate(&graph,0));
            appendImplicitMaterialSolve(b,radiation,graph,nullptr,b.stream.get());
            cudaGraphExec_t executable=nullptr;checked(cudaGraphInstantiateWithFlags(&executable,graph,0));
            if(s.continuation)cudaGraphExecDestroy(s.continuation);
            s.continuation=executable;s.continuationSignature=std::move(signature);
            checked(cudaGraphDestroy(graph));graph=nullptr;
        }catch(const std::exception& exception){
            if(graph)cudaGraphDestroy(graph);
            error=exception.what();return false;
        }
    }
    if(!b.fault.check(cudaGraphLaunch(s.continuation,b.stream.get()),"submit implicit material window"))return b.failure(error);
    return true;
}
}

bool runImplicitMaterialSolve(MaterialBackend& b,const IntervalRadiationDeviceView& radiation,std::string& error){
    if(radiation.faces!=0&&(radiation.faces!=b.implicit.surfaceFaces.size()||!radiation.energy)){
        error="implicit material window radiation extent mismatch";return false;
    }
    if(radiation.faces&&radiation.producerStream!=b.stream.get())acceptImplicitWindowExecution(b,radiation.producerStream);
    return enqueueImplicitContinuation(b,radiation.energy,error);
}

bool collectImplicitMaterialSolve(MaterialBackend& b,std::vector<SolidQ>& candidate,ImplicitWindowReport& report,std::string& error){
    auto& s=b.implicit;
    if(!readImplicitWindowReport(b,report,error))return false;
    while(report.capacity){
        if(!s.sparse->servicePendingCapacity()){
            error="unable to provide implicit material preconditioner storage";return false;
        }
        if(!configureImplicitWindowConduction(b,*s.diffusion,*s.sparse,error)
            ||!enqueueImplicitContinuation(b,nullptr,error)||!readImplicitWindowReport(b,report,error))return false;
    }
    if(!report.complete){error="implicit material graph stopped before the endpoint";return false;}
    if(!b.read(b.next,candidate)||!b.transfers.finish())return b.failure(error);
    return true;
}

bool commitImplicitMaterialInventory(MaterialBackend& b,const std::vector<SolidQ>& accepted,std::string& error){
    if(accepted.size()!=b.next.size()){error="implicit material accepted inventory extent mismatch";return false;}
    if(!b.current.copy(b.next,b.fault,b.stream.get()))return b.failure(error);
    b.publishedCurrent=accepted;b.publishedCurrentValid=true;
    return true;
}
