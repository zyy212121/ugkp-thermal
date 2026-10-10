#pragma once
#include "gasTransport/GasBoundaryLayerWorkspace.H"
extern "C" int ugkwpGpuResidentStrictConfigureBoundaryLayerV1
(void* handle,const ugkwpGpuIpc::BoundaryLayerConfigV1* config,const int* faces,
 const int* qOffsets,const int* mOffsets,const int* mCells,const double* geometry,
 const double* qDistance,const double* qWeight,const double* mWeight,const double* speciesFlux)
{
    DeviceState* s=asState(handle);std::uint64_t bytes=0;
    if(validateState(s,"configure boundaryLayer") || !config || !faces || !qOffsets || !mOffsets || !mCells
        || !geometry || !qDistance || !qWeight || !mWeight || !speciesFlux
        || !ugkwpGpuIpc::boundaryLayerPayloadBytes(*config,s->nFaces,ugkwp::compiledGasSpecies,bytes)
        || s->gasBoundaryLayer.enabled || !s->gasSpeciesUploaded || !s->gasSpecies.diffusivity)
    {setLastErrorText("invalid or repeated boundaryLayer configuration");return 1;}
    // Initial-only wall configuration binds phase caches and snapshot layout;
    // optional SST audit channels also change the full-cell snapshot size.
    if(s->gasTrial.trial.cells || s->gasTrial.interval.cells)
    {setLastErrorText("boundaryLayer must be configured before the first gas trial");return 1;}
    const auto& a=*config;
    if(a.budgetAudit && s->hostTurbulenceModel!=3)
    {setLastErrorText("boundaryLayer budgetAudit requires SST");return 1;}
    auto request=sharedGasCapabilityRequest(s,s->gasSpecies.mode,2);request.boundaryLayerModel=a.model;
    const auto capability=ugkwp::validateGasCapabilities(request);
    if(!capability){setLastErrorText(capability.message);return 1;}
    if(!std::isfinite(a.relativeTolerance) || a.relativeTolerance<=0 || !std::isfinite(a.absoluteTolerance)
        || a.absoluteTolerance<0 || !std::isfinite(a.stretch) || a.stretch<1
        || !std::isfinite(a.turbulentPrandtl) || a.turbulentPrandtl<=0
        || !std::isfinite(a.turbulentSchmidt) || a.turbulentSchmidt<=0
        || qOffsets[0]!=0 || mOffsets[0]!=0 || qOffsets[a.wallCount]!=int(a.quadratureCount)
        || mOffsets[a.wallCount]!=int(a.matchingCount))
    {setLastErrorText("invalid boundaryLayer controls or compact offsets");return 1;}
    std::vector<int> owners(s->nFaces),neighbours(s->nFaces),kinds(s->nFaces);
    std::vector<double> volumes(s->nCells),areas(s->nFaces);
    if(copyToHost(owners.data(),s->faceOwner,owners.size(),"read wall owners")
        || copyToHost(neighbours.data(),s->faceNeighbour,neighbours.size(),"read wall neighbours")
        || copyToHost(kinds.data(),s->gasBoundaryKind,kinds.size(),"read wall classifications")
        || copyToHost(volumes.data(),s->V,volumes.size(),"read wall volumes")
        || copyToHost(areas.data(),s->magSf,areas.size(),"read wall areas"))return 1;
    std::vector<int> faceSlot(s->nFaces,-1),ownerSlot(s->nCells,-1);
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        const int f=faces[slot];
        if(f<0 || f>=s->nFaces || owners[f]<0 || owners[f]>=s->nCells || neighbours[f]>=0
            || kinds[f]!=2 || faceSlot[f]>=0 || ownerSlot[owners[f]]>=0
            || qOffsets[slot]<0 || qOffsets[slot+1]<=qOffsets[slot] || qOffsets[slot+1]>int(a.quadratureCount)
            || mOffsets[slot]<0 || mOffsets[slot+1]<=mOffsets[slot] || mOffsets[slot+1]>int(a.matchingCount)
            || mOffsets[slot+1]-mOffsets[slot]>32)
        {setLastErrorText("invalid boundaryLayer wall map or CSR range");return 1;}
        faceSlot[f]=slot;ownerSlot[owners[f]]=slot;
        const double* g=geometry+7*slot;double norm=0,sum=0,moment=0;
        for(int d=0;d<7;++d)if(!std::isfinite(g[d]))
        {setLastErrorText("nonfinite boundaryLayer geometry");return 1;}
        for(int d=0;d<3;++d)norm+=g[d]*g[d];
        if(std::abs(norm-1)>1e-9 || g[3]<=0 || g[4]<=g[3] || g[5]<=0 || g[6]<=0
            || !std::isfinite(volumes[owners[f]]) || std::abs(g[5]-volumes[owners[f]])>1e-9*g[5]
            || !std::isfinite(areas[f]) || areas[f]<=0)
        {setLastErrorText("boundaryLayer geometry does not match owner volume");return 1;}
        for(int q=qOffsets[slot];q<qOffsets[slot+1];++q)
        {
            if(!std::isfinite(qDistance[q]) || qDistance[q]<0 || qDistance[q]>=g[4]
                || !std::isfinite(qWeight[q]) || qWeight[q]<=0)
            {setLastErrorText("invalid positive owner quadrature");return 1;}
            sum+=qWeight[q];moment+=qWeight[q]*qDistance[q];
        }
        if(std::abs(sum-g[5])>1e-9*g[5] || std::abs(moment-g[6])>1e-9*g[6])
        {setLastErrorText("owner quadrature does not close its volume/moment");return 1;}
        double mass=0;
        for(int species=0;species<ugkwp::compiledGasSpecies;++species)
        {
            const double flux=speciesFlux[species*a.wallCount+slot];
            if(!std::isfinite(flux)){setLastErrorText("nonfinite physical species wall flux");return 1;}
            mass+=flux;
        }
        if(mass<0){setLastErrorText("boundaryLayer suction is outside the supported model");return 1;}
    }
    // Every physical gas-only wall uses the selected family. Mixed CHMT walls
    // have a different application adapter and retain their explicit slot map.
    for(int f=0;f<s->nFaces;++f)if(kinds[f]==2 && faceSlot[f]<0)
    {setLastErrorText("gasUGKP boundaryLayer descriptors must cover all physical walls");return 1;}
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        double sum=0;
        for(int m=mOffsets[slot];m<mOffsets[slot+1];++m)
        {
            if(mCells[m]<0 || mCells[m]>=s->nCells || ownerSlot[mCells[m]]>=0
                || !std::isfinite(mWeight[m]) || mWeight[m]<=0)
            {setLastErrorText("invalid or circular boundaryLayer matching donor");return 1;}
            sum+=mWeight[m];
        }
        if(std::abs(sum-1)>1e-12){setLastErrorText("boundaryLayer matching weights must sum to one");return 1;}
    }
    ugkwp::BoundaryLayerWorkspaceSizing sizing;
    if(a.model!=0)
    {
        int device=0,multiprocessors=0;std::size_t freeBytes=0,totalBytes=0;
        if(cudaGetDevice(&device)!=cudaSuccess
            || cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess
            || cudaMemGetInfo(&freeBytes,&totalBytes)!=cudaSuccess)
        {setLastErrorText("cannot query boundaryLayer scratch memory/device");return 1;}
        sizing=ugkwp::resolveBoundaryLayerWorkspace(a.wallCount,a.workspaceSlots,multiprocessors,
            freeBytes,ugkwp::gasBoundaryLayerWorkspaceBytes<double,ugkwp::compiledGasSpecies>(a.nodes));
        if(sizing.slots<=0){setLastErrorText("insufficient bounded boundaryLayer scratch memory");return 1;}
    }
    BoundaryLayerAllocation candidate;auto& w=candidate.gasBoundaryLayer;auto& model=candidate.gasBoundaryLayerModel;
    w.count=a.wallCount;int rc=0;
#define WALL_ALLOC(field,count) rc|=allocateSharedGasZero(candidate.field,count)
    WALL_ALLOC(gasBoundaryLayer.exchange,a.wallCount);WALL_ALLOC(gasBoundaryLayer.sst,a.wallCount);
    WALL_ALLOC(gasBoundaryLayer.status,a.wallCount);WALL_ALLOC(gasBoundaryLayer.speciesFlux,std::size_t(a.wallCount)*a.speciesCount);
    WALL_ALLOC(gasBoundaryLayerModel.input,a.wallCount);WALL_ALLOC(gasBoundaryLayerModel.output,a.wallCount);
    WALL_ALLOC(gasBoundaryLayerModel.status,a.wallCount);
    model.workspaceCount=sizing.slots;
    model.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(a.nodes);
    unsigned char* scratch=nullptr;
    rc|=allocate(scratch,sizing.bytes,"allocate bounded wall scratch");
    model.workspace=scratch;
    WALL_ALLOC(gasWallQuadratureDistance,a.quadratureCount);WALL_ALLOC(gasWallQuadratureWeight,a.quadratureCount);
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
    rc|=copyToDevice(candidate.gasWallQuadratureDistance,qDistance,a.quadratureCount,"upload wall quadrature distances");
    rc|=copyToDevice(candidate.gasWallQuadratureWeight,qWeight,a.quadratureCount,"upload wall quadrature weights");
    std::vector<ugkwp::gaswall::WallInput<double,ugkwp::compiledGasSpecies>> inputs(a.wallCount);
    std::vector<ugkwp::GasBoundaryLayerExchange<double>> exchanges(a.wallCount);
    for(unsigned slot=0;slot<a.wallCount;++slot)
    {
        auto& in=inputs[slot];const double*g=geometry+7*slot;
        for(int d=0;d<3;++d)in.normal[d]=g[d];
        in.ownerDistance=g[3];in.matchingDistance=g[4];
        in.quadrature={candidate.gasWallQuadratureDistance+qOffsets[slot],candidate.gasWallQuadratureWeight+qOffsets[slot],
            qOffsets[slot+1]-qOffsets[slot],g[5],g[6]};
        for(int species=0;species<ugkwp::compiledGasSpecies;++species)
        {in.massFlux[species]=speciesFlux[species*a.wallCount+slot];exchanges[slot].mass-=areas[faces[slot]]*in.massFlux[species];}
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
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1
(void* handle,std::uint32_t walls,std::uint32_t species,double* diagnostics)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"download boundaryLayer") || !diagnostics || !s->gasBoundaryLayer.enabled
        || walls!=unsigned(s->gasBoundaryLayer.count) || species!=ugkwp::compiledGasSpecies)
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
