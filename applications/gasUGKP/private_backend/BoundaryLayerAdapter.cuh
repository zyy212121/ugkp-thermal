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
        releaseSharedGasTrialStorage(s->gasTrial);
        releaseBoundaryLayerStorage(*s);s->sstWallTreatment=2;
        if(syncBoundaryLayerConfiguration(s)){s->gasModelPoisoned=true;return 1;}
        return 0;
    }
    if(!faces || !qOffsets || !mOffsets || !mCells || !geometry || !mWeight || !speciesFlux
        || (a.quadratureCount&&(!qDistance||!qWeight)) || !s->faceOwner || !s->magSf
        || !s->gasSpecies.diffusivity || a.speciesCount<ugkwp::compiledGasSpecies)
    {setLastErrorText("missing consumed boundaryLayer arrays or species values");return 1;}
    if(a.model>1){setLastErrorText("missing selected boundaryLayer model");return 1;}
    if(a.wallCount>unsigned(std::numeric_limits<int>::max())
        || a.quadratureCount>unsigned(std::numeric_limits<int>::max())
        || a.matchingCount>unsigned(std::numeric_limits<int>::max()))
    {setLastErrorText("boundaryLayer count exceeds actual compact index storage");return 1;}
    std::vector<int> owners(s->nFaces);
    std::vector<double> areas(s->nFaces);
    if(copyToHost(owners.data(),s->faceOwner,owners.size(),"read wall owners")
        || copyToHost(areas.data(),s->magSf,areas.size(),"read wall areas"))return 1;
    std::vector<int> faceSlot(s->nFaces,-1),ownerSlot(s->nCells,-1);
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        const int f=faces[slot];
        if(f<0 || f>=s->nFaces || owners[f]<0 || owners[f]>=s->nCells
            || qOffsets[slot]<0 || qOffsets[slot+1]<qOffsets[slot] || qOffsets[slot+1]>int(a.quadratureCount)
            || mOffsets[slot]<0 || mOffsets[slot+1]<=mOffsets[slot] || mOffsets[slot+1]>int(a.matchingCount))
        {setLastErrorText("missing boundaryLayer face, owner, quadrature or matching range");return 1;}
        faceSlot[f]=slot;ownerSlot[owners[f]]=slot;
        for(int m=mOffsets[slot];m<mOffsets[slot+1];++m)
            if(mCells[m]<0 || mCells[m]>=s->nCells)
            {setLastErrorText("missing boundaryLayer matching cell");return 1;}
    }
    ugkwp::BoundaryLayerWorkspaceSizing sizing;
    if(a.model!=0)
    {
        const auto workspaceBytes=ugkwp::gasBoundaryLayerWorkspaceBytes<double,ugkwp::compiledGasSpecies>(a.nodes);
        if(a.nodes<2 || !workspaceBytes)
        {setLastErrorText("boundaryLayer nodes have no compiled workspace/equation storage");return 1;}
        int device=0,multiprocessors=0;std::size_t freeBytes=0,totalBytes=0;
        if(cudaGetDevice(&device)!=cudaSuccess
            || cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess
            || cudaMemGetInfo(&freeBytes,&totalBytes)!=cudaSuccess)
        {setLastErrorText("cannot query boundaryLayer scratch memory/device");return 1;}
        sizing=ugkwp::resolveBoundaryLayerWorkspace(a.wallCount,a.workspaceSlots,multiprocessors,
            freeBytes,workspaceBytes);
        if(sizing.slots<=0){setLastErrorText("boundaryLayer scratch size is not representable");return 1;}
    }
    // Finish all throwing host staging allocations before acquiring any
    // candidate device ownership or replacing the published configuration.
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
    if(a.budgetAudit)
    {
        WALL_ALLOC(gasSstAudit.transportK,s->nCells);WALL_ALLOC(gasSstAudit.transportOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.sourceK,s->nCells);WALL_ALLOC(gasSstAudit.sourceOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.constraintK,s->nCells);WALL_ALLOC(gasSstAudit.constraintOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.initialTransportK,s->nCells);WALL_ALLOC(gasSstAudit.initialTransportOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.initialSourceK,s->nCells);WALL_ALLOC(gasSstAudit.initialSourceOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.initialConstraintK,s->nCells);WALL_ALLOC(gasSstAudit.initialConstraintOmega,s->nCells);
        WALL_ALLOC(gasSstAudit.volume,s->nCells);
        candidate.gasSstAudit.enabled=true;
    }
#undef WALL_ALLOC
    rc|=uploadSharedGasTable(w.faceSlot,faceSlot.data(),faceSlot.size());
    rc|=uploadSharedGasTable(w.ownerSlot,ownerSlot.data(),ownerSlot.size());
    rc|=uploadSharedGasTable(model.faces,faces,a.wallCount);
    rc|=uploadSharedGasTable(model.matchingOffsets,mOffsets,a.wallCount+1);
    rc|=uploadSharedGasTable(model.matchingCells,mCells,a.matchingCount);
    rc|=uploadSharedGasTable(model.matchingWeights,mWeight,a.matchingCount);
    if(rc){releaseBoundaryLayerStorage(candidate);return 1;}
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
            if(!std::isfinite(in.massFlux[species]))
            {setLastErrorText("nonfinite physical species wall flux");releaseBoundaryLayerStorage(candidate);return 1;}
            exchanges[slot].mass-=areas[faces[slot]]*in.massFlux[species];
        }
        exchanges[slot].massReady=true;
    }
    rc|=copyToDevice(model.input,inputs.data(),inputs.size(),"upload wall input descriptors");
    rc|=copyToDevice(w.exchange,exchanges.data(),exchanges.size(),"upload known wall mass flux");
    if(rc){releaseBoundaryLayerStorage(candidate);return 1;}
    model.config.model=static_cast<ugkwp::gaswall::BoundaryLayerModel>(a.model);
    model.config.nodes=a.nodes;model.config.maxIterations=a.maxIterations;
    model.config.relativeTolerance=a.relativeTolerance;model.config.absoluteTolerance=a.absoluteTolerance;
    model.config.stretch=a.stretch;model.config.turbulentPrandtl=a.turbulentPrandtl;
    model.config.turbulentSchmidt=a.turbulentSchmidt;model.config.enableSst=s->hostTurbulenceModel==3;
    // All real allocations and uploads succeeded. Replace old ownership and
    // discard snapshots of the previous wall/audit layout before publication.
    releaseSharedGasTrialStorage(s->gasTrial);
    releaseBoundaryLayerStorage(*s);
    w.enabled=true;s->gasBoundaryLayer=w;s->gasBoundaryLayerModel=model;s->gasSstAudit=candidate.gasSstAudit;
    s->gasWallQuadratureDistance=candidate.gasWallQuadratureDistance;s->gasWallQuadratureWeight=candidate.gasWallQuadratureWeight;
    s->sstWallTreatment=2;
    if(syncBoundaryLayerConfiguration(s))
    {s->gasModelPoisoned=true;return 1;}
    std::fprintf(stderr,"boundaryLayer scratch: nodes=%u capacity=%d slots=%d bytes=%zu budget=%zu (auto=%s)\n",
        a.nodes,model.workspaceCapacity,model.workspaceCount,sizing.bytes,sizing.budget,a.workspaceSlots==0?"yes":"no");
    std::fprintf(stderr,"boundaryLayer full-cell SST budget audit: %s\n",a.budgetAudit?"enabled":"disabled");
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
    if(!diagnostics || !s->gasBoundaryLayer.enabled
        || walls>unsigned(s->gasBoundaryLayer.count) || species>ugkwp::compiledGasSpecies)
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
    const unsigned columns=ugkwpGpuIpc::boundaryLayerDiagnosticScalars+2*species;
    std::vector<double> result(std::size_t(walls)*columns);
    std::vector<double> budget[6];
    if(s->gasSstAudit.enabled)
    {
        const double* channels[]={s->gasSstAudit.transportK,s->gasSstAudit.transportOmega,
            s->gasSstAudit.sourceK,s->gasSstAudit.sourceOmega,s->gasSstAudit.constraintK,s->gasSstAudit.constraintOmega};
        for(int channel=0;channel<6;++channel)
        {
            budget[channel].resize(s->nCells);
            if(copyToHost(budget[channel].data(),channels[channel],s->nCells,"read opt-in SST budget channel"))return 1;
        }
    }
    for(unsigned slot=0;slot<walls;++slot)
    {
        if(!exchange[slot].ready){setLastErrorText("wall diagnostics requested before an accepted profile");return 1;}
        auto*r=result.data()+slot*columns;const auto&i=input[slot];const auto&o=output[slot];
        r[0]=faces[slot];r[1]=owners[faces[slot]];r[2]=i.matchingDistance;r[3]=i.ownerDistance;r[4]=o.conductiveHeatFlux;
        r[5]=o.traction[0];r[6]=o.traction[1];r[7]=o.traction[2];r[8]=o.wallKFlux;r[9]=o.integratedKSource;
        r[10]=o.ownerOmega;r[11]=o.residual;r[12]=o.iterations;r[13]=o.traceDensity;r[14]=i.model.viscosity;r[15]=s->gasBoundaryLayerStageTime;
        // These are unique-owner inventory increments (k: J, omega: kg/s),
        // for the last accepted microstep, not face fluxes or an output-interval sum.
        r[16]=s->gasSstAudit.enabled?1:0;
        r[17]=s->gasSstAudit.enabled?s->gasBoundaryLayer.preparedTime:std::numeric_limits<double>::quiet_NaN();
        r[18]=s->gasSstAudit.enabled?s->gasBoundaryLayer.preparedInterval:std::numeric_limits<double>::quiet_NaN();
        for(int channel=0;channel<6;++channel)r[19+channel]=s->gasSstAudit.enabled
            ?budget[channel][owners[faces[slot]]]:std::numeric_limits<double>::quiet_NaN();
        for(unsigned k=0;k<species;++k){r[25+k]=o.wallSpeciesFlux[k];r[25+species+k]=o.reactionIntegral[k];}
    }
    std::copy(result.begin(),result.end(),diagnostics);return 0;
}
