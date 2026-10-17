#pragma once
#include <new>
#include "gasTransport/GasBoundaryLayerWorkspace.H"
extern "C" int ugkwpGpuResidentStrictConfigureBoundaryLayerV1
(void* handle,const ugkwpGpuIpc::BoundaryLayerConfigV1* config,const int* faces,
 const int* qOffsets,const int* mOffsets,const int* mCells,const double* geometry,
 const double* qDistance,const double* qWeight,const double* mWeight,const double* speciesFlux)
try
{
    DeviceState* s=asState(handle);
    if(validateState(s,"configure boundaryLayer") || !config)
    {setLastErrorText("missing boundaryLayer resident or configuration");return 1;}
    const auto& a=*config;
    if(!a.wallCount)
    {
        if(releaseSharedGasTrialStorage(s->gasTrial))return 1;
        const int released=releaseBoundaryLayerStorage(*s);
        if(released){s->gasModelPoisoned=true;return 1;}
        s->sstWallTreatment=2;
        invalidateSharedGasConfiguration(s);
        if(syncBoundaryLayerConfiguration(s)){s->gasModelPoisoned=true;return 1;}
        return 0;
    }
    if(!faces || !qOffsets || !mOffsets || !mCells || !geometry || !mWeight || !speciesFlux
        || (a.quadratureCount&&(!qDistance||!qWeight)) || !s->faceOwner || !s->magSf
        || !s->gasSpecies.diffusivity)
    {setLastErrorText("missing consumed boundaryLayer arrays or species values");return 1;}
    if(a.model>1){setLastErrorText("missing selected boundaryLayer model");return 1;}
    std::vector<int> owners(s->nFaces);
    std::vector<double> areas(s->nFaces);
    if(copyToHost(owners.data(),s->faceOwner,owners.size(),"read wall owners")
        || copyToHost(areas.data(),s->magSf,areas.size(),"read wall areas"))return 1;
    if(s->diagnosticOwner->completeReadback(0,!owners.empty()))return 1;
    std::vector<int> faceSlot(s->nFaces,-1),ownerSlot(s->nCells,-1);
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        const int f=faces[slot];
        faceSlot[f]=slot;ownerSlot[owners[f]]=slot;
    }
    ugkwp::BoundaryLayerWorkspaceSizing sizing;
    if(a.model!=0)
    {
        const auto workspaceBytes=ugkwp::gasBoundaryLayerWorkspaceBytes<double,ugkwp::compiledGasSpecies>(a.nodes);
        if(!workspaceBytes)
        {setLastErrorText("boundaryLayer nodes have no compiled workspace/equation storage");return 1;}
        int device=0,multiprocessors=0;
        if(cudaGetDevice(&device)!=cudaSuccess
            || cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess)
        {setLastErrorText("cannot query boundaryLayer scratch device");return 1;}
        sizing=ugkwp::resolveBoundaryLayerWorkspace(a.wallCount,a.workspaceSlots,multiprocessors,
            0,workspaceBytes);
        if(sizing.slots<=0){setLastErrorText("boundaryLayer scratch size is not representable");return 1;}
    }


    std::vector<ugkwp::gaswall::WallInput<double,ugkwp::compiledGasSpecies>> inputs(a.wallCount);
    std::vector<ugkwp::GasBoundaryLayerExchange<double>> exchanges(a.wallCount);
    BoundaryLayerAllocation candidate;auto& w=candidate.gasBoundaryLayer;auto& model=candidate.gasBoundaryLayerModel;
    w.count=a.wallCount;int rc=0;
#define WALL_ALLOC(field,count) rc|=allocateSharedGasZero(candidate.field,count)
    WALL_ALLOC(gasBoundaryLayer.exchange,a.wallCount);WALL_ALLOC(gasBoundaryLayer.sst,a.wallCount);
    WALL_ALLOC(gasBoundaryLayer.status,a.wallCount);WALL_ALLOC(gasBoundaryLayer.speciesFlux,std::size_t(a.wallCount)*ugkwp::compiledGasSpecies);
    WALL_ALLOC(gasBoundaryLayerModel.input,a.wallCount);WALL_ALLOC(gasBoundaryLayerModel.output,a.wallCount);
    WALL_ALLOC(gasBoundaryLayerModel.status,a.wallCount);
    model.workspaceCount=sizing.slots;
    model.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(a.nodes);
    unsigned char* scratch=nullptr;
    rc|=allocate(scratch,sizing.bytes,"allocate bounded wall scratch");
    model.workspace=scratch;
    if(a.quadratureCount){WALL_ALLOC(gasWallQuadratureDistance,a.quadratureCount);WALL_ALLOC(gasWallQuadratureWeight,a.quadratureCount);}

#undef WALL_ALLOC
    rc|=uploadSharedGasTable(w.faceSlot,faceSlot.data(),faceSlot.size());
    rc|=uploadSharedGasTable(w.ownerSlot,ownerSlot.data(),ownerSlot.size());
    rc|=uploadSharedGasTable(model.faces,faces,a.wallCount);
    rc|=uploadSharedGasTable(model.matchingOffsets,mOffsets,a.wallCount+1);
    rc|=uploadSharedGasTable(model.matchingCells,mCells,a.matchingCount);
    rc|=uploadSharedGasTable(model.matchingWeights,mWeight,a.matchingCount);
    if(rc){discardBoundaryLayerCandidate(s,candidate);return 1;}
    if(a.quadratureCount){
        rc|=copyToDevice(candidate.gasWallQuadratureDistance,qDistance,a.quadratureCount,"upload wall quadrature distances");
        rc|=copyToDevice(candidate.gasWallQuadratureWeight,qWeight,a.quadratureCount,"upload wall quadrature weights");
    }
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        auto& in=inputs[slot];const double*g=geometry+std::size_t(7)*slot;
        for(int d=0;d<3;++d)in.normal[d]=g[d];
        in.ownerDistance=g[3];in.matchingDistance=g[4];
        in.quadrature={a.quadratureCount?candidate.gasWallQuadratureDistance+qOffsets[slot]:nullptr,
            a.quadratureCount?candidate.gasWallQuadratureWeight+qOffsets[slot]:nullptr,
            qOffsets[slot+1]-qOffsets[slot],g[5],g[6]};
        for(int species=0;species<ugkwp::compiledGasSpecies;++species)
        {
            in.massFlux[species]=speciesFlux[std::size_t(species)*a.wallCount+slot];
            exchanges[slot].mass-=areas[faces[slot]]*in.massFlux[species];
        }
        exchanges[slot].massReady=true;
    }
    rc|=copyToDevice(model.input,inputs.data(),inputs.size(),"upload wall input descriptors");
    rc|=copyToDevice(w.exchange,exchanges.data(),exchanges.size(),"upload known wall mass flux");
    if(rc){discardBoundaryLayerCandidate(s,candidate);return 1;}
    model.config.model=static_cast<ugkwp::gaswall::BoundaryLayerModel>(a.model);
    model.config.nodes=a.nodes;model.config.maxIterations=a.maxIterations;
    model.config.relativeTolerance=a.relativeTolerance;model.config.absoluteTolerance=a.absoluteTolerance;
    model.config.stretch=a.stretch;model.config.turbulentPrandtl=a.turbulentPrandtl;
    model.config.turbulentSchmidt=a.turbulentSchmidt;model.config.enableSst=s->hostTurbulenceModel==3;


    if(releaseSharedGasTrialStorage(s->gasTrial)){discardBoundaryLayerCandidate(s,candidate);return 1;}
    if(releaseBoundaryLayerStorage(*s)){
        discardBoundaryLayerCandidate(s,candidate);s->gasModelPoisoned=true;return 1;
    }
    w.enabled=true;s->gasBoundaryLayer=w;s->gasBoundaryLayerModel=model;
    s->gasWallQuadratureDistance=candidate.gasWallQuadratureDistance;s->gasWallQuadratureWeight=candidate.gasWallQuadratureWeight;
    s->sstWallTreatment=2;
    invalidateSharedGasConfiguration(s);
    if(syncBoundaryLayerConfiguration(s))
    {s->gasModelPoisoned=true;return 1;}
    std::fprintf(stderr,"boundaryLayer scratch: nodes=%u capacity=%d slots=%d bytes=%zu (auto=%s)\n",
        a.nodes,model.workspaceCapacity,model.workspaceCount,sizing.bytes,a.workspaceSlots==0?"yes":"no");
    return 0;
}
catch(const std::bad_alloc&)
{
    setLastErrorText("boundaryLayer host allocation failed");return 1;
}
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1
(void* handle,std::uint32_t walls,std::uint32_t species,double* diagnostics)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"download boundaryLayer"))return 1;
    if(!walls&&!s->gasBoundaryLayer.count)return 0;
    if(!diagnostics || !s->gasBoundaryLayer.exchange || !s->gasBoundaryLayerModel.input || !s->gasBoundaryLayerModel.output)
    {setLastErrorText("invalid boundaryLayer diagnostic dimensions");return 1;}
    std::vector<ugkwp::gaswall::WallInput<double,ugkwp::compiledGasSpecies>> input(walls);
    std::vector<ugkwp::gaswall::WallOutput<double,ugkwp::compiledGasSpecies>> output(walls);
    std::vector<ugkwp::GasBoundaryLayerExchange<double>> exchange(walls);
    std::vector<int> faces(walls),owners(s->nFaces);
    if(copyToHost(input.data(),s->gasBoundaryLayerModel.input,walls,"read wall input")
        || copyToHost(output.data(),s->gasBoundaryLayerModel.output,walls,"read wall output")
        || copyToHost(exchange.data(),s->gasBoundaryLayer.exchange,walls,"read wall readiness")
        || copyToHost(faces.data(),s->gasBoundaryLayerModel.faces,walls,"read wall face map")
        || copyToHost(owners.data(),s->faceOwner,owners.size(),"read wall owner map"))return 1;
    if(s->diagnosticOwner->completeReadback(0,walls!=0||!owners.empty()))return 1;
    const unsigned columns=ugkwpGpuIpc::boundaryLayerDiagnosticScalars+2*species;
    std::vector<double> result(std::size_t(walls)*columns);
    for(unsigned slot=0;slot<walls;++slot)
    {
        if(!exchange[slot].ready){setLastErrorText("wall diagnostics requested before an accepted profile");return 1;}
        auto*r=result.data()+slot*columns;const auto&i=input[slot];const auto&o=output[slot];
        r[0]=faces[slot];r[1]=owners[faces[slot]];r[2]=i.matchingDistance;r[3]=i.ownerDistance;r[4]=o.conductiveHeatFlux;
        r[5]=o.traction[0];r[6]=o.traction[1];r[7]=o.traction[2];r[8]=o.wallKFlux;r[9]=o.integratedKSource;
        r[10]=o.ownerOmega;r[11]=o.residual;r[12]=o.iterations;r[13]=o.traceDensity;r[14]=i.model.viscosity;r[15]=s->gasBoundaryLayerStageTime;

        for(unsigned k=0;k<species;++k){r[25+k]=o.wallSpeciesFlux[k];r[25+species+k]=o.reactionIntegral[k];}
    }
    std::copy(result.begin(),result.end(),diagnostics);return 0;
}
