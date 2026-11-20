namespace {
bool prepareWindowRecession(const HostState& base,const std::vector<ImplicitWindowRemoval>& removal,
    const PhysicsConfig& p,Real dt,HostState& next,CoupledMeshSweeps& swept,CpuMaterialReport& report,
    std::string& error){
    next.filmAux=dryGeometryAux(base);
    const auto& surface=base.surface;
    const auto count=surface.solidFace.size();
    if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession||count==0)return true;
    if(surface.area.size()!=count||surface.solidCell.size()!=count
        ||(!surface.ablating.empty()&&surface.ablating.size()!=count)
        ||removal.size()!=count||!validateMaterialSweepRemainder(base,error,p.diagnostics)){
        if(error.empty())error="invalid material recession face layout";
        return false;
    }
    std::vector<Real> targets(count);
    auto remainder=base.solidSweepRemainder;
    bool needsMotion=false;
    for(std::size_t f=0;f<count;++f){
        const bool mobile=surface.ablating.empty()||surface.ablating[f];
        const int c=surface.solidCell[f];
        if(c<0||std::size_t(c)>=base.solid.size()){error="material recession donor missing";return false;}
        PhysicalReal condensed[Nc]{};
        bool removed=false;
        for(int k=0;k<Nc;++k){
            condensed[k]=removal[f].mass[k];
            removed|=condensed[k]!=0;
        }
        if(!mobile){
            if(removed){error="accepted surface reaction on a fixed material face";return false;}
            if(!remainder.empty())remainder[f]=0;
        }else if(removed&&addMaterialRecessionTarget(condensed,base.solid[c],base.solidMesh.volumes[c],p,targets[f])
            !=MaterialRecessionError::None){
            error="cannot recover the material recession target";return false;
        }
        const Real previous=remainder.empty()?0:remainder[f];
        needsMotion|=targets[f]+previous!=0;
        next.filmAux[f]=materialRecessionFront(next.filmAux[f],targets[f],surface.area[f],dt);
    }
    if(!needsMotion){
        next.solidSweepRemainder=std::move(remainder);
        return true;
    }
    HostState old;
    old.gasMesh=base.gasMesh;old.solidMesh=base.solidMesh;old.surface=base.surface;
    old.filmAux=dryGeometryAux(base);old.solidSweepRemainder=std::move(remainder);
    SweepConstraintReport motion;
    if(!moveCoupledMeshesConstrained(old,next.filmAux,targets,base.solid.size(),dt,next.gasMesh,next.solidMesh,next.surface,
        motion,error,materialSweepControls(p.numerics.meshMotion),p.tolerances,nullptr,&swept))return false;
    if(!updateMaterialSweepRemainder(old,targets,swept.solid,next.solidMesh,next.solidSweepRemainder,error,p.diagnostics))return false;
    report.geometrySolves=1;
    report.geometryCorrection=!identicalPoints(base.gasMesh.points,next.gasMesh.points)
        ||!identicalPoints(base.solidMesh.points,next.solidMesh.points);
    const Real gcl=std::max(maximumGclResidual(base.solidMesh,next.solidMesh,swept.solid),
        maximumGclResidual(base.gasMesh,next.gasMesh,swept.gas));
    if(!finite(gcl)){error="nonfinite window geometric conservation residual";return false;}
    report.gclResidual=gcl;
    next.budget.gclResidual=std::max(next.budget.gclResidual,gcl);
    for(std::size_t f=0;f<count;++f){
        const auto displacement=swept.solid[surface.solidFace[f]];
        next.filmAux[f]=materialRecessionFront(old.filmAux[f],displacement,surface.area[f],dt);
        next.filmAux[f].area=next.surface.area[f];
        next.filmAux[f].normal=PhysicalVec3{PhysicalReal(next.surface.normal[f].x),
            PhysicalReal(next.surface.normal[f].y),PhysicalReal(next.surface.normal[f].z)};
        report.recessionVolume+=displacement;
    }
    return true;
}

bool prepareImplicitThermalGraph(MaterialBackend& backend,Foam::fvMesh& mesh,const HostState& endpoint,
    const PhysicsConfig& physics,const std::vector<Real>& swept,Real dt,const CpuMaterialControls& controls,
    MaterialConductionWorkspace& workspace,ugkp::gpu::SparseConjugateGradient& sparse,std::string& error){
    if(!prepareConductionWorkspace(mesh,endpoint,physics,workspace,error))return false;
    if(workspace.implicitConfigured&&implicitWindowMeshConfigured(backend,endpoint.solidMesh,swept))return true;
    const auto& native=*workspace.mesh;
    const std::vector<Real> volumes(native.V().begin(),native.V().end());
    if(!configureImplicitWindowMesh(backend,endpoint.solidMesh,swept,volumes,error))return false;
    auto& temperature=*workspace.temperature;auto& conductivity=*workspace.conductivity;
    if(!ugkp::gpu::residentMaterialStartupSupported(conductivity,temperature)){
        error="material thermal discretization cannot provide a resident diffusion operator";return false;
    }
    temperature.correctBoundaryConditions();
    auto faceK=ugkp::gpu::thermalHarmonicBoundaryValues(conductivity);
    const auto device=implicitWindowThermalFields(backend);
    auto& diffusion=workspace.diffusion;
    auto equation=ugkp::gpu::residentImplicitThermalEquation(diffusion,sparse,faceK(),temperature,
        *workspace.capacity,*workspace.source,dt,false,device.capacity,device.source,device.stream,
        device.conductivity,false,&conductivity,&faceK());
    Foam::dictionary solver;
    solver.add("maxIter",Foam::label(controls.linearMaxIterations));
    if(!equation.configureResidentLocalSolve(solver)){
        error="material thermal matrix cannot provide a resident local solve";return false;
    }
    equation.fluxValues(false,false);
    workspace.implicitConfigured=configureImplicitWindowConduction(backend,diffusion,sparse,error);
    return workspace.implicitConfigured;
}

template<class Operation>
bool implicitWindowHostDiagnostics(const HostState& base,const CouplingInterval& window,const PhysicsConfig& config,
    std::string& error,const Operation& operation){
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,window.identity.sequence,reinterpret_cast<std::uintptr_t>(&config),
        base.acceptedSteps,window.end);
    auto physics=config;physics.diagnostics=&context;physics.tolerances.diagnostics=&context;
    const bool healthy=operation(physics);
    auto snapshot=ugkp::diagnostics::takeSnapshot(context,ugkp::diagnostics::sourceCatalogue());
    if(!healthy&&!snapshot.hasFatal){
        snapshot.hasFatal=true;
        snapshot.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,10053,4);
        snapshot.fatal.step=context.step;snapshot.fatal.simulationTime=context.simulationTime;
        snapshot.fatal.instanceId=context.instanceId;snapshot.fatal.operationSequence=context.operationSequence;
        snapshot.sources.push_back({10053,"applications/CHMT/materials/ImplicitMaterialWindow.inl",
            __func__,__LINE__,error});
    }
    ugkp::diagnostics::dispatch(snapshot,ugkp::diagnostics::makeFoamSink());
    return healthy;
}
}

bool CpuMaterialDriver::prepareImplicitParent(const HostState& base,const CouplingInterval& window,
    const CpuMaterialControls& controls,bool deferRadiation,MaterialBackend*& bound,bool& appendSolve,std::string& error){
    bound=nullptr;appendSolve=false;
    if(base.solid.empty())return true;
    if(!data_->mesh){error="missing coupled material FV mesh";return false;}
    return implicitWindowHostDiagnostics(base,window,data_->model.physics,error,[&](const PhysicsConfig& physics){
        if(!data_->gpu&&!data_->begin(base,physics,error))return false;
        auto& device=*data_->gpu;
        if(!beginImplicitMaterialExchange(device,base,window,controls,error))return false;
        if(physics.meshMotion.policy==MeshMotionPolicy::Static&&!deferRadiation){
            if(!data_->conduction)data_->conduction=std::make_unique<ugkp::gpu::SparseConjugateGradient>();
            if(!prepareImplicitThermalGraph(device,*data_->mesh,base,physics,{},window.end-window.begin,
                controls,data_->conductionWorkspace,*data_->conduction,error))return false;
            appendSolve=true;
        }
        bound=&device;
        return true;
    });
}

bool CpuMaterialDriver::finishImplicitParent(const HostState& base,const HostState& gasEndpoint,const CouplingInterval& window,
    const CpuMaterialControls& controls,const IntervalRadiationDeviceView& radiation,bool inParent,
    HostState& output,CpuMaterialReport& report,std::string& error){
    report=CpuMaterialReport{};
    if(gasEndpoint.time!=window.end||!sameMeshConnectivity(base.solidMesh,gasEndpoint.solidMesh)
        ||!sameMeshConnectivity(base.gasMesh,gasEndpoint.gasMesh)){
        error="resident material window endpoint does not match the gas window";return false;
    }
    HostState next=gasEndpoint;next.ledger.clear();
    if(base.solid.empty()){output=std::move(next);return true;}
    auto& device=*data_->gpu;
    if(!implicitWindowHostDiagnostics(base,window,data_->model.physics,error,[&](const PhysicsConfig& physics){
        std::vector<ImplicitWindowRemoval> removal;
        if(physics.meshMotion.policy==MeshMotionPolicy::CoupledRecession
            &&!collectImplicitWindowRemoval(device,removal,error))return false;
        CoupledMeshSweeps swept;
        const Real dt=window.end-window.begin;
        report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;
        if(!prepareWindowRecession(base,removal,physics,dt,next,swept,report,error))return false;
        report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;
        if(!inParent){
            if(!data_->conduction)data_->conduction=std::make_unique<ugkp::gpu::SparseConjugateGradient>();
            if(!prepareImplicitThermalGraph(device,*data_->mesh,next,physics,swept.solid,dt,controls,
                data_->conductionWorkspace,*data_->conduction,error)||!runImplicitMaterialSolve(device,radiation,error))return false;
        }
        return true;
    }))return false;
    ImplicitWindowReport receipt;
    if(!collectImplicitMaterialSolve(device,next.solid,receipt,error))return false;
    next.budget.boundaryEnergy+=receipt.boundaryEnergy;
    report.materialSteps=1;report.implicitIterations=receipt.iterations;report.implicitConverged=receipt.converged;
    report.massResidual=receipt.massResidual;report.thermalResidual=receipt.energyResidual;
    report.exchangePackets=receipt.packets;report.exchangeFrames=receipt.frames;
    report.materialLinearSolves=receipt.linearSolves;report.materialLinearIterations=receipt.linearIterations;
    report.deviceCommit={window.end,window.identity.sequence,base.acceptedSteps+1,true};
    report.failureOrigin=MaterialFailureOrigin::None;
    output=std::move(next);error.clear();return true;
}

bool CpuMaterialDriver::commitImplicitInventory(const HostState& state,std::string& error){
    return state.solid.empty()||commitImplicitMaterialInventory(*data_->gpu,state.solid,error);
}
