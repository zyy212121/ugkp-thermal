namespace {
bool prepareWindowRecession(const HostState& base,const MaterialWindowExchange& exchange,
    const PhysicsConfig& p,Real dt,HostState& next,CoupledMeshSweeps& swept,CpuMaterialReport& report,
    std::string& error){
    next.filmAux=dryGeometryAux(base);
    const auto& surface=base.surface;
    const auto count=surface.solidFace.size();
    if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession||count==0)return true;
    if(surface.area.size()!=count||surface.solidCell.size()!=count
        ||(!surface.ablating.empty()&&surface.ablating.size()!=count)
        ||exchange.removed.size()!=count||!validateMaterialSweepRemainder(base,error,p.diagnostics)){
        if(error.empty())error="invalid material recession face layout";
        return false;
    }
    std::vector<Real> targets(count);
    HostState old=base;old.filmAux=next.filmAux;
    bool needsMotion=false;
    for(std::size_t f=0;f<count;++f){
        const bool mobile=surface.ablating.empty()||surface.ablating[f];
        const int c=surface.solidCell[f];
        if(c<0||std::size_t(c)>=base.solid.size()){error="material recession donor missing";return false;}
        Real condensed[Nc]{};
        bool removed=false;
        for(int k=0;k<Nc;++k){
            condensed[k]=exchange.removed[f][k].value();
            removed|=condensed[k]!=0;
        }
        if(!mobile){
            if(removed){error="accepted surface reaction on a fixed material face";return false;}
            if(!old.solidSweepRemainder.empty())old.solidSweepRemainder[f]=0;
        }else if(removed&&addMaterialRecessionTarget(condensed,base.solid[c],base.solidMesh.volumes[c],p,targets[f])
            !=MaterialRecessionError::None){
            error="cannot recover the material recession target";return false;
        }
        const Real previous=old.solidSweepRemainder.empty()?0:old.solidSweepRemainder[f];
        needsMotion|=targets[f]+previous!=0;
        next.filmAux[f]=materialRecessionFront(old.filmAux[f],targets[f],surface.area[f],dt);
    }
    if(!needsMotion){
        next.solidSweepRemainder=std::move(old.solidSweepRemainder);
        return true;
    }
    SweepConstraintReport motion;
    if(!moveCoupledMeshesConstrained(old,next.filmAux,targets,dt,next.gasMesh,next.solidMesh,next.surface,
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

bool implicitTemperatureUpdate(HostState& next,MaterialConductionWorkspace& workspace,
    ugkp::gpu::SparseConjugateGradient& sparse,const std::vector<MaterialCaloric>& caloric,
    const std::vector<MaterialPrimitive>& primitive,const std::vector<Real>& target,
    std::vector<Real>& temperatures,const PhysicsConfig& p,Real dt,const CpuMaterialControls& controls,
    Real& thermalResidual,Real& change,Real& boundaryInput,std::string& error){
    using namespace Foam;
    auto& mesh=*workspace.mesh;
    auto& T=*workspace.temperature;auto& k=*workspace.conductivity;
    auto& capacity=*workspace.capacity;auto& source=*workspace.source;
    const auto n=temperatures.size();
    for(std::size_t c=0;c<n;++c){
        const auto anchor=PhysicalReal(temperatures[c]);
        T[c]=anchor;k[c]=primitive[c].conductivity;
        const Real cp=caloric[c].capacity(anchor);
        if(!(cp>0)||!finite(cp)){error="invalid implicit material caloric capacity";return false;}
        capacity[c]=cp/mesh.V()[c];
        source[c]=(target[c]-caloric[c].energy(anchor)+cp*T[c])/(mesh.V()[c]*dt);
    }
    T.correctBoundaryConditions();k.correctBoundaryConditions();
    capacity.correctBoundaryConditions();source.correctBoundaryConditions();
    const auto interpolation=surfaceInterpolationScheme<scalar>::New(mesh,
        ugkp::solverControls::thermalDiscretization(mesh).selection("interpolationSchemes","interpolate("+k.name()+')'));
    auto faceK=interpolation().interpolate(k);
    if(dynamic_cast<const ugkp::gpu::FoamThermalLinear*>(&interpolation()))
        ugkp::gpu::interpolateThermalField(k,mesh.weights(),faceK.ref(),true);
    auto equation=ugkp::gpu::residentImplicitThermalEquation(workspace.diffusion,sparse,faceK(),T,capacity,source,dt);
    const auto initial=equation.residual();
    long double residualEnergy=0,scale=0;
    for(std::size_t c=0;c<n;++c){
        residualEnergy+=dt*std::abs(initial()[c]);
        scale+=std::abs(target[c]);
    }
    if(!std::isfinite(residualEnergy)||!std::isfinite(scale)){error="nonfinite initial thermal residual";return false;}
    const auto energyTolerance=controls.energyAbsoluteTolerance+controls.energyRelativeTolerance*std::max(1.L,residualEnergy)
        +8*std::numeric_limits<EnergyReal>::epsilon()*scale;
    const Real relativeGoal=residualEnergy>0
        ?std::min(Real(1),Real(energyTolerance*controls.linearEnergyBudgetFraction/residualEnergy)):1;
    dictionary linear;
    linear.add("solver",word("PCG"));linear.add("preconditioner",word("jacobi"));
    linear.add("maxIter",label(controls.linearMaxIterations));
    linear.add("tolerance",std::numeric_limits<Foam::scalar>::min());linear.add("relTol",relativeGoal);
    const auto solved=equation.solve(linear);
    if(!finite(solved.finalResidual())){error="nonfinite coupled thermal linear residual";return false;}
    const Real linearTarget=std::max(solved.initialResidual()*relativeGoal,std::numeric_limits<Foam::scalar>::min());
    if(solved.finalResidual()>linearTarget)
        materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,"coupled thermal linear residual",
            solved.finalResidual(),linearTarget);
    const auto flux=equation.fluxValues(false);
    const auto& geometry=next.solidMesh;
    if(flux.size()!=geometry.owner.size()){error="coupled thermal face flux layout mismatch";return false;}
    boundaryInput=0;
    for(std::size_t f=0;f<flux.size();++f)
        if(geometry.neighbour[f]<0&&geometry.boundaryKind[f]!=BoundaryKind::Periodic)boundaryInput-=dt*flux[f];
    for(std::size_t c=0;c<n;++c){
        if(!finite(T[c])||T[c]<=0){error="inadmissible solved material temperature";return false;}
        long double energy=target[c];
        for(int i=geometry.cellFaceOffsets[c];i<geometry.cellFaceOffsets[c+1];++i)
            energy-=static_cast<long double>(dt)*geometry.cellFaceSigns[i]*flux[geometry.cellFaces[i]];
        auto& q=next.solid[c];q.energy=EnergyReal(energy);
        MaterialPrimitive endpoint;
        if(!std::isfinite(energy)||!recoverMaterial(q,geometry.volumes[c],p,endpoint)){
            error="invalid conservative material energy in cell "+std::to_string(c);return false;
        }
        q.porosity=endpoint.porosity;
        const Real heatDefect=std::abs(energy-caloric[c].energy(PhysicalReal(T[c])));
        const Real scale=std::max(Real(1),std::abs(Real(energy)-target[c]));
        const Real rounding=8*std::numeric_limits<EnergyReal>::epsilon()
            *std::max(std::abs(Real(energy)),std::abs(target[c]));
        const Real threshold=controls.energyAbsoluteTolerance+controls.energyRelativeTolerance*scale+rounding;
        thermalResidual=std::max(thermalResidual,heatDefect/std::max(threshold,std::numeric_limits<Real>::min()));
        change=std::max(change,std::abs(Real(endpoint.temperature)-temperatures[c])
            /std::max(Real(1),std::abs(Real(endpoint.temperature))));
        temperatures[c]=endpoint.temperature;
    }
    return finite(boundaryInput);
}

bool solveImplicitMaterialWindow(const HostState& base,const MaterialWindowExchange& exchange,
    const CoupledMeshSweeps& swept,Foam::fvMesh& native,HostState& next,MaterialConductionWorkspace& workspace,
    ugkp::gpu::SparseConjugateGradient& sparse,const PhysicsConfig& p,Real dt,
    const CpuMaterialControls& controls,CpuMaterialReport& report,std::string& error){
    const auto n=base.solid.size();
    ImplicitMaterialMass mass;
    if(!mass.begin(base,next.solidMesh,exchange,p.tolerances,error)
        ||!prepareConductionWorkspace(native,next,p,workspace,error))return false;
    next.solid.resize(n);
    std::vector<Real> temperatures(n),fixedEnergy(n),target(n);
    std::vector<MaterialCaloric> caloric(n);
    std::vector<MaterialPrimitive> primitive(n);
    for(std::size_t c=0;c<n;++c){
        MaterialPrimitive initial;
        if(!recoverMaterial(base.solid[c],base.solidMesh.volumes[c],p,initial)){
            error="invalid window initial material EOS in cell "+std::to_string(c);return false;
        }
        temperatures[c]=initial.temperature;
        auto acceptedEnergy=exchange.cells[c][Nc+Ns];
        acceptedEnergy.add(base.solid[c].energy);
        fixedEnergy[c]=acceptedEnergy.value();
        if(!finite(fixedEnergy[c])){error="nonfinite fixed material energy";return false;}
    }
    const bool physicalTransport=p.permeability>0||std::any_of(std::begin(p.poreEffectiveDiffusivity),
        std::end(p.poreEffectiveDiffusivity),[](auto value){return value>0;});
    const bool ale=std::any_of(swept.solid.begin(),swept.solid.end(),[](Real value){return value!=0;});
    const auto changeGoal=std::max(Real(controls.nonlinearRelativeTolerance),Real(4)*std::numeric_limits<PhysicalReal>::epsilon());
    Real boundaryInput=0,temperatureResidual=0;
    for(int i=0;i<controls.nonlinearMaxIterations;++i){
        Real massResidual=0,energyResidual=0,change=0;
        mass.reconstruct(p);
        target=fixedEnergy;
        if(!mass.prepare(temperatures,next.solidMesh.volumes,p,next.solid,caloric,primitive,error))return false;
        if(physicalTransport||ale){
            MaterialTransportResult rates;
            if(!evaluateMaterialTransport(next.solidMesh,next.solid,p,swept.solid,dt,controls.transportCfl,rates,error,nullptr,&primitive)
                ||!mass.transport(next.solidMesh,swept.solid,p,primitive,rates,dt,massResidual,error)){
                report.failureOrigin=MaterialFailureOrigin::TransportNumerical;return false;
            }
            const auto& geometry=next.solidMesh;
            for(std::size_t c=0;c<n;++c){
                Real power=0;
                for(int i=geometry.cellFaceOffsets[c];i<geometry.cellFaceOffsets[c+1];++i)
                    power-=geometry.cellFaceSigns[i]*Real(rates.faceFlux[geometry.cellFaces[i]].energy);
                Real poreMass=0;
                for(int species=0;species<Ns;++species)poreMass+=next.solid[c].pore[species];
                power+=poreMass*dot(p.gravity,rates.poreVelocity[c]);
                target[c]+=dt*power;
            }
        }
        if(!mass.react(next.solidMesh,p,temperatures,dt,massResidual,error)){
            report.failureOrigin=MaterialFailureOrigin::ReactionNumerical;return false;
        }
        if((physicalTransport||ale||p.enableReactions)
            &&!mass.prepare(temperatures,next.solidMesh.volumes,p,next.solid,caloric,primitive,error))return false;
        if(!implicitTemperatureUpdate(next,workspace,sparse,caloric,primitive,target,temperatures,p,dt,
            controls,energyResidual,change,boundaryInput,error))return false;
        report.implicitIterations=i+1;
        report.massResidual=massResidual;report.thermalResidual=energyResidual;
        temperatureResidual=change/changeGoal;
        if(massResidual<=1&&energyResidual<=1&&change<=changeGoal){
            report.implicitConverged=true;break;
        }
    }
    if(!report.implicitConverged){
        const Real residual=std::max({report.massResidual,report.thermalResidual,temperatureResidual});
        materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,
            "coupled implicit material window iteration budget exhausted",residual,1);
    }
    next.budget.boundaryEnergy+=boundaryInput;
    return true;
}
}

bool CpuMaterialDriver::advanceImplicitWindow(const HostState& base,const HostState& gasEndpoint,
    const CouplingInterval& window,const MaterialWindowExchange& exchange,const CpuMaterialControls& controls,
    HostState& output,CpuMaterialReport& report,std::string& error){
    report=CpuMaterialReport{};
    ugkp::diagnostics::DeviceContext diagnostic{};
    ugkp::diagnostics::reset(diagnostic,window.identity.sequence,reinterpret_cast<std::uintptr_t>(this),
        base.acceptedSteps,window.end);
    auto physics=data_->model.physics;physics.diagnostics=&diagnostic;physics.tolerances.diagnostics=&diagnostic;
    const auto advance=[&](){
        if(controls.nonlinearMaxIterations<=0||controls.linearMaxIterations<=0
            ||!finite(controls.nonlinearRelativeTolerance)||controls.nonlinearRelativeTolerance<0
            ||!finite(controls.energyAbsoluteTolerance)||controls.energyAbsoluteTolerance<0
            ||!finite(controls.energyRelativeTolerance)||controls.energyRelativeTolerance<0
            ||!finite(controls.linearEnergyBudgetFraction)||controls.linearEnergyBudgetFraction<=0
            ||!validCouplingInterval(window)
            ||window.begin!=base.time||gasEndpoint.time!=window.end
            ||!sameMeshConnectivity(base.solidMesh,gasEndpoint.solidMesh)
            ||!sameMeshConnectivity(base.gasMesh,gasEndpoint.gasMesh)
            ||base.solid.size()!=base.solidMesh.volumes.size()||exchange.cells.size()!=base.solid.size()
            ||physics.nReactions<0||physics.nReactions>Nr){
            error="invalid coupled material window input";report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;
        }
        if(!base.solid.empty()&&!data_->mesh){
            error="missing coupled material FV mesh";report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;
        }
        HostState next=gasEndpoint;next.ledger.clear();
        const Real dt=window.end-window.begin;
        CoupledMeshSweeps swept;
        report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;
        if(!prepareWindowRecession(base,exchange,physics,dt,next,swept,report,error))return false;
        report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;
        if(!base.solid.empty()){
            if(!data_->conduction)data_->conduction.reset(new ugkp::gpu::SparseConjugateGradient);
            if(!solveImplicitMaterialWindow(base,exchange,swept,*data_->mesh,next,data_->conductionWorkspace,
                *data_->conduction,physics,dt,controls,report,error))return false;
            report.materialSteps=1;
        }
        report.failureOrigin=MaterialFailureOrigin::None;output=std::move(next);error.clear();return true;
    };
    const bool ok=advance();
    auto snapshot=ugkp::diagnostics::takeSnapshot(diagnostic,ugkp::diagnostics::sourceCatalogue());
    if(!ok&&!snapshot.hasFatal){
        snapshot.hasFatal=true;
        snapshot.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,10053,4);
        snapshot.fatal.step=diagnostic.step;snapshot.fatal.simulationTime=diagnostic.simulationTime;
        snapshot.fatal.instanceId=diagnostic.instanceId;snapshot.fatal.operationSequence=diagnostic.operationSequence;
        snapshot.sources.push_back({10053,"applications/CHMT/materials/CpuMaterialDriver.C",__func__,__LINE__,error});
    }
    ugkp::diagnostics::dispatch(snapshot,ugkp::diagnostics::makeFoamSink());
    return ok;
}
