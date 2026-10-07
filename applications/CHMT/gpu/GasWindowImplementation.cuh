// Included by Backend.cu inside namespace chmt. This path deliberately never
// invokes material/film/interface-constitutive RHS or downloads their state.
namespace {
bool copyGasWindowInventory(StateBuffers& to,const StateBuffers& from,Backend& b){
    const auto stream=b.stream.get();
    return to.gas.copy(from.gas,b.fault,stream)&&to.sst.copy(from.sst,b.fault,stream)
        &&to.particles.copy(from.particles,b.fault,stream)&&to.voidFraction.copy(from.voidFraction,b.fault,stream);
}
bool resetGasWindowScratch(StateBuffers& state,Backend& b){
    const auto s=b.stream.get();
    return state.gasRhs.zero(s)&&state.gradient.zero(s)&&state.packets.zero(s)
        &&state.omegaConstraint.zero(s)&&state.particleRadiation.zero(s)&&state.particleBodyWork.zero(s)
        &&state.particleSupportWork.zero(s)&&state.particleSupportImpulse.zero(s);
}
__global__ void gasOnlyStableStepKernel(GasView gas,GeometryView geometry,const PhysicsConfig* physics,Real interval,Real* result){
    if(blockIdx.x||threadIdx.x)return;
    *result=gasOnlyStableStepLimit(gas,geometry,*physics,interval);
}
__global__ void prescribedWallKernel(GasView gas,GeometryView geometry,SurfaceView surface,
    const WallFaceSample* program,const WallFaceSample* nextProgram,Real alpha,PacketView packets,const PhysicsConfig* physics,Real dt,
    std::uint64_t step,int stage,bool withGradient,bool traceOnly,GasPrimitive* drive,GasGradient* driveGradient,Vec3* traction,DeviceStatus* status){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=surface.nFaces||gasStopped(status))return;
    const auto& p=*physics;const int face=surface.gasFace[f],cell=geometry.owner[face];
    GasWallInput in;DeviceStatus local;
    if(!recoverGas(gas.q[cell],geometry.evaluationVolume[cell],p,in.bulk,&local,cell)){
        gasDeviceError(status,ErrorCode::PropertyRange,cell,local.value,ErrorLocation::Cell);return;}
    if(withGradient)in.gradient=gas.gradient[cell];
    auto w=program[f];
    if(nextProgram){const auto next=nextProgram[f];w.temperature+=alpha*(next.temperature-w.temperature);w.velocity+=(next.velocity-w.velocity)*alpha;
        for(int s=0;s<Ns;++s){w.speciesRate[s]+=alpha*(next.speciesRate[s]-w.speciesRate[s]);w.poreRate[s]+=alpha*(next.poreRate[s]-w.poreRate[s]);w.poreSweepRate[s]+=alpha*(next.poreSweepRate[s]-w.poreSweepRate[s]);}
        for(int c=0;c<Nc;++c)w.condensedRate[c]+=alpha*(next.condensedRate[c]-w.condensedRate[c]);}
    in.temperature=w.temperature;in.velocity=w.velocity;in.primaryKind=w.primaryKind;
    in.dt=dt;in.area=surface.area[f];in.gasArea=mag(geometry.areaVector[face]);
    in.normal=-geometry.areaVector[face]/in.gasArea;in.gasDistance=surface.gasDistance[f];
    in.sweptVolume=-geometry.sweptVolume[face];in.normalSpeed=in.sweptVolume/(dt*in.gasArea);
    for(int s=0;s<Ns;++s){in.speciesRate[s]=w.speciesRate[s];in.poreRate[s]=w.poreRate[s];in.poreSweepRate[s]=w.poreSweepRate[s];}
    for(int c=0;c<Nc;++c)in.condensedRate[c]=w.condensedRate[c];
    SurfacePacketIdentity id;id.step=step;id.stage=stage;id.geometry=geometry.geometryVersion;
    id.face=geometry.faceIds[face];id.gasCell=cell;id.solidCell=surface.solidCell[f];id.filmFace=f;
    GasWallResult result;
    if(!evaluateGasWall(in,id,p,result)){gasDeviceError(status,ErrorCode::PropertyRange,f,w.temperature,ErrorLocation::Face);return;}
    surface.interfaceGasState[f]=result.trace;
    if(!traceOnly){packets.packets[2*f]=result.primary;packets.packets[2*f+1]=result.pore;
        drive[f]=in.bulk;driveGradient[f]=in.gradient;traction[f]=result.traction;}
}
__global__ void gasWindowDonorKernel(GasView gas,PacketView packets,DeviceStatus* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=gas.nCells||gasStopped(status))return;
    Real withdrawal[Ns]{};
    for(int i=0;i<packets.count;++i){const auto& p=packets.packets[i];if(!(requiredConsumers(p.kind)&ConsumeGas)||p.gasCell!=c)continue;
        for(int s=0;s<Ns;++s)withdrawal[s]+=gasPacketSpeciesWithdrawal(p,s);}
    for(int s=0;s<Ns;++s)if(!finite(withdrawal[s])||withdrawal[s]>gas.stageBase[c].species[s]){
        gasDeviceError(status,ErrorCode::Inventory,c,withdrawal[s],ErrorLocation::Cell);return;}
}
CHMT_HD inline bool emptyParticleWindowPacket(const ExchangePacket& q){
    if(q.kind!=ExchangeKind::ParticleGas&&q.kind!=ExchangeKind::ParticleWall)return false;
    if(q.mass!=0||q.energy!=0||mag(q.momentum)!=0||q.advective!=0||q.conductive!=0||q.pressureWork!=0||q.viscousWork!=0||q.radiation!=0||q.liquidKineticAdvection!=0)return false;
    for(int s=0;s<Ns;++s)if(q.species[s]!=0||q.pore[s]!=0||q.poreSweep[s]!=0)return false;
    for(int c=0;c<Nc;++c)if(q.condensed[c]!=0)return false;
    return true;
}
__global__ void gasWindowAuditKernel(GasView gas,const GasQ* baseGas,const GasQ* evaluationGas,
    const ParticleQ* particles,const ParticleQ* baseParticles,int np,const Real* radiation,
    const Real* particleBody,const Real* particleSupport,const Vec3* particleImpulse,
    GeometryView geometry,PacketView packets,const PhysicsConfig* physics,Real dt,
    Budget original,Budget* output,StageDiagnostics* diagnostics,DeviceStatus* status){
    if(blockIdx.x||threadIdx.x||gasStopped(status))return;
    const auto& p=*physics;Budget budget=original;StageDiagnostics result;
    result.minimumMass=std::numeric_limits<Real>::max();result.minimumTemperature=std::numeric_limits<Real>::max();
    Real external=0,interfaceEnergy=0;
    for(int c=0;c<gas.nCells;++c){
        result.energyBefore+=baseGas[c].energy;result.energyAfter+=gas.q[c].energy;
        result.minimumMass=minValue(result.minimumMass,gas.q[c].mass);result.minimumTemperature=minValue(result.minimumTemperature,gas.primitive[c].temperature);
        const Real body=dt*dot(p.gravity,evaluationGas[c].momentum);budget.bodyWork+=body;external+=body;
        if(p.enableSst){budget.turbulenceProduction+=dt*gas.sst.production[c];budget.turbulenceDissipation+=dt*gas.sst.dissipation[c];budget.turbulenceOmegaConstraint+=gas.sst.omegaConstraint[c];}
    }
    if(p.enableSst){budget.turbulenceInventory=0;for(int c=0;c<gas.nCells;++c)budget.turbulenceInventory+=gas.sst.q[c].rhoK;}
    for(int f=0;f<geometry.nFaces;++f){const auto kind=geometry.boundaryKind[f];
        if(kind==BoundaryKind::Internal||kind==BoundaryKind::Periodic||kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;
        const auto flux=gas.faceFlux[f];budget.boundaryMass+=dt*flux.mass;budget.boundaryEnergy+=dt*flux.energy;budget.boundaryMomentum+=flux.momentum*dt;external-=dt*flux.energy;
        for(int s=0;s<Ns;++s){budget.boundarySpecies[s]+=dt*flux.species[s];for(int e=0;e<p.nElements;++e)budget.boundaryElements[e]+=dt*flux.species[s]*p.species[s].element[e];}
        if(p.enableSst)budget.turbulenceBoundaryFlux+=dt*gas.sst.faceFlux[f].rhoK;
    }
    for(int i=0;i<np;++i){result.energyBefore+=particleTotalEnergy(baseParticles[i]);result.energyAfter+=particleTotalEnergy(particles[i]);
        budget.radiation+=radiation[i];budget.bodyWork+=particleBody[i];budget.supportWork+=particleSupport[i];budget.supportImpulse+=particleImpulse[i];external+=radiation[i]+particleBody[i]+particleSupport[i];}
    for(int i=0;i<packets.count;++i){auto& q=packets.packets[i];
        if(emptyParticleWindowPacket(q)){q.consumerMask=q.kind==ExchangeKind::ParticleGas?ConsumeGas|ConsumeParticle:ConsumeParticle;continue;}
        if(!validatePacketMath(q,p.tolerances,status)||q.consumerMask){gasDeviceError(status,ErrorCode::Packet,i,q.mass,ErrorLocation::Face);return;}
        const unsigned required=requiredConsumers(q.kind);const auto delta=packetDelta(q);
        if(required&ConsumeGas){budget.exchangeMass[GasParticipant]+=delta.gas.mass;budget.exchangeEnergy[GasParticipant]+=delta.gas.energy;budget.exchangeMomentum[GasParticipant]+=delta.gas.momentum;
            for(int s=0;s<Ns;++s)budget.exchangeSpecies[GasParticipant][s]+=delta.gas.species[s];q.consumerMask|=ConsumeGas;}
        if(required&ConsumeParticle){budget.exchangeMass[ParticleParticipant]+=delta.particleMass;budget.exchangeEnergy[ParticleParticipant]+=delta.particleEnergy;budget.exchangeMomentum[ParticleParticipant]+=delta.particleMomentum;
            for(int s=0;s<Ns;++s)budget.exchangeSpecies[ParticleParticipant][s]-=q.species[s];q.consumerMask|=ConsumeParticle;}
        if(q.kind==ExchangeKind::ParticleGas)++budget.consumedPackets;
        else interfaceEnergy+=q.kind==ExchangeKind::ParticleWall?-q.energy:q.energy;
    }
    const Real residual=result.energyAfter-result.energyBefore-external-interfaceEnergy;
    budget.numericalEnergyResidual+=residual;
    if(!validBudget(budget)||!finite(residual)||absValue(residual)>p.tolerances.absoluteEnergy+p.tolerances.relativeEnergy*maxValue(absValue(result.energyBefore),absValue(result.energyAfter))){
        gasDeviceError(status,ErrorCode::Inventory,0,residual,ErrorLocation::Configuration);return;}
    *output=budget;*diagnostics=result;
}
void setWindowStage(const HostMesh& base,const HostMesh& eval,const HostMesh& end,
    Real dt,const std::vector<Real>& sweeps,const std::vector<Vec3>& areas,HostStageGeometry& stage){
    stage.interval=dt;stage.geometryVersion=end.geometryVersion;stage.topologyHash=end.topologyHash;
    stage.oldVolume=base.volumes;stage.newVolume=end.volumes;stage.evaluationVolume=eval.volumes;stage.sweptVolume=sweeps;
    stage.areaVector=areas;stage.cellCentre=eval.cellCentres;stage.faceCentre=eval.faceCentres;stage.oldPoints=base.points;stage.newPoints=end.points;
}
bool windowGeometry(Backend& b,PendingGasWindow& window,const WallKnot& sample,const WallKnot& endpoint,
    const HostMesh& evalGas,const HostMesh& evalSolid,Real dt,HostMesh& gasEnd,HostMesh& solidEnd,
    SurfaceMesh& surface,HostStageGeometry& gasStage,HostStageGeometry& solidStage,std::string& error){
    if(window.staticGeometry){surface=window.surface;return true;}
    std::vector<Real> gs,ss;std::vector<Vec3> ga,sa;
    if(!makeStageGeometry(window.gasMesh,endpoint.gasPoints,dt,gasEnd,gs,error,b.model.physics.tolerances)
        ||!makeStageAreaVectors(window.gasMesh,endpoint.gasPoints,ga,error)
        ||!makeStageGeometry(window.solidMesh,endpoint.solidPoints,dt,solidEnd,ss,error,b.model.physics.tolerances)
        ||!makeStageAreaVectors(window.solidMesh,endpoint.solidPoints,sa,error))return false;
    const auto& p=b.model.physics;surface=window.program.surface;
    for(std::size_t f=0;f<surface.area.size();++f){const int gf=surface.gasFace[f],sf=surface.solidFace[f];
        const Real gasArea=mag(ga[gf]),solidArea=mag(sa[sf]);const Vec3 gn=-ga[gf]/gasArea,sn=sa[sf]/solidArea;
        const bool moving=gs[gf]!=0||ss[sf]!=0;
        if(moving&&(!closeValue(gasArea,solidArea,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)||mag(gn-sn)>1e-10)){
            error="moving offset gas/material surfaces remain unsupported";return false;}
        if(sample.faces[f].primaryKind==ExchangeKind::GasSolid&&(!closeValue(gasArea,solidArea,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)||mag(gn-sn)>1e-10)){
            error="dry gas/material interface geometry mismatch";return false;}
    }
    SurfaceMesh evaluationSurface=window.surface;
    const Real evaluationDt=sample.time-window.time;
    if(evaluationDt>0&&!rebuildTrajectorySurface(window.gasMesh,window.solidMesh,window.surface,evalGas,evalSolid,evaluationDt,evaluationSurface,error,b.model.physics.tolerances))return false;
    if(!rebuildTrajectorySurface(window.gasMesh,window.solidMesh,window.surface,gasEnd,solidEnd,dt,surface,error,b.model.physics.tolerances))return false;
    setWindowStage(window.gasMesh,evalGas,gasEnd,dt,gs,ga,gasStage);setWindowStage(window.solidMesh,evalSolid,solidEnd,dt,ss,sa,solidStage);
    if(!b.gasGeometry.upload(window.gasMesh,evalGas,gasEnd,gs,ga,b.fault)
        ||!b.solidGeometry.upload(window.solidMesh,evalSolid,solidEnd,ss,sa,b.fault)
        ||!b.surface.upload(evaluationSurface,b.fault)||!b.endpointSurface.upload(surface,b.fault)){error=b.fault.message;return false;}
    return true;
}
bool gasWindowStage(Backend& b,PendingGasWindow& window,const StateBuffers& evaluation,
    const HostMesh& evalGas,const HostMesh& evalSolid,StateBuffers& state,
    const WallKnot& sample,const WallKnot& endpoint,Real dt,int stage,
    HostMesh& gasEnd,HostMesh& solidEnd,SurfaceMesh& surface,HostStageGeometry& gasStage,HostStageGeometry& solidStage,std::string& error){
    auto stream=b.stream.get();const auto& p=b.model.physics;
    if(!freshStatus(b)||!copyGasWindowInventory(state,evaluation,b)||!resetGasWindowScratch(state,b)){error=b.fault.message;return false;}
    if(!windowGeometry(b,window,sample,endpoint,evalGas,evalSolid,dt,gasEnd,solidEnd,surface,gasStage,solidStage,error))return false;
    auto gas=state.gasView(*window.current,b.status.data(),b.gasGeometry.wallDistance.data());auto geometry=b.gasGeometry.view();
    const int count=surface.area.size();
    std::size_t knot=0;while(knot+1<window.program.knots.size()&&window.program.knots[knot+1].time<=sample.time)++knot;
    const std::size_t next=std::min(knot+1,window.program.knots.size()-1);
    const Real alpha=next==knot?0:(sample.time-window.program.knots[knot].time)/(window.program.knots[next].time-window.program.knots[knot].time);
    auto wall=[&](bool gradients){if(!count)return true;
        prescribedWallKernel<<<(count+127)/128,128,0,stream>>>(gas,geometry,b.surface.view(b.solidGeometry),window.wall.data()+knot*count,window.wall.data()+next*count,alpha,state.packetView(),state.physics.data(),dt,window.microSequence+1,stage,gradients,false,window.drive.data(),window.driveGradient.data(),window.traction.data(),b.status.data());
        return launch(b,static_cast<int>(cudaPeekAtLastError()),"evaluate immutable gas wall program",error)&&checked(b,error)&&scatter(b,b.surface,error);};
    if(!wall(false)||!launch(b,launchGasPrepare(gas,geometry,p,dt,stream),"prepare gas-only evaluation",error)||!checked(b,error)||!wall(true))return false;
    gasOnlyStableStepKernel<<<1,1,0,stream>>>(gas,geometry,state.physics.data(),dt,b.stableDt.data());
    if(!launch(b,static_cast<int>(cudaPeekAtLastError()),"gas-only CFL/diffusion bound",error)||!checked(b,error)||!b.stableDt.downloadOne(b.nextStableDt,stream)){if(error.empty())error=b.fault.message;return false;}
    if(!finite(b.nextStableDt)||b.nextStableDt<=0||dt>b.nextStableDt*(1+1e-12)){error="gas-only CFL/diffusion timestep exceeded";return false;}
    if(p.enableParticles&&!launch(b,launchParticleExchange(*window.current,evaluation,state,b.gasGeometry,b.surface,p,dt,window.microSequence+1,stage,2*count,b.status.data(),stream),"gas-window particle exchange",error))return false;
    gasWindowDonorKernel<<<(gas.nCells+127)/128,128,0,stream>>>(gas,state.packetView(),b.status.data());
    if(!launch(b,static_cast<int>(cudaPeekAtLastError()),"aggregate gas boundary donor check",error)||!checked(b,error)
        ||!launch(b,launchGasRhs(gas,geometry,state.packetView(),p,dt,stream),"gas-only RHS",error)
        ||!launch(b,launchGasUpdate(window.current->gas.data(),gas,geometry,dt,b.status.data(),stream),"gas-only conservative update",error))return false;
    if(p.enableSst&&!launch(b,launchSstUpdate(window.current->sst.data(),gas,geometry,p,dt,b.status.data(),stream),"gas-only SST update",error))return false;
    if(!launch(b,launchGasValidate(gas,b.gasGeometry.view(true),p,stream),"gas-window endpoint admissibility",error)||!checked(b,error))return false;
    // Endpoint trace recovery is read-only with respect to accepted packets,
    // driving samples and GPU inventories. It supplies the endpoint SST wall
    // constraint/restart boundary state, never a regenerated energy transfer.
    if(count){
        std::size_t endpointKnot=0;while(endpointKnot+1<window.program.knots.size()&&window.program.knots[endpointKnot+1].time<=endpoint.time)++endpointKnot;
        const std::size_t endpointNext=std::min(endpointKnot+1,window.program.knots.size()-1);
        const Real endpointAlpha=endpointNext==endpointKnot?0:(endpoint.time-window.program.knots[endpointKnot].time)/(window.program.knots[endpointNext].time-window.program.knots[endpointKnot].time);
        prescribedWallKernel<<<(count+127)/128,128,0,stream>>>(gas,b.gasGeometry.view(true),b.endpointSurface.view(b.solidGeometry),window.wall.data()+endpointKnot*count,window.wall.data()+endpointNext*count,endpointAlpha,state.packetView(),state.physics.data(),dt,window.microSequence+1,stage,false,true,nullptr,nullptr,nullptr,b.status.data());
        if(!launch(b,static_cast<int>(cudaPeekAtLastError()),"recover immutable endpoint wall trace",error)||!checked(b,error)||!scatter(b,b.endpointSurface,error))return false;
    }
    if(p.enableSst){if(!launch(b,launchWallTrace(gas,b.gasGeometry.view(true),b.gasGeometry.wallTrace.data(),p,b.status.data(),stream),"gas-window endpoint wall trace",error)
        ||!launch(b,launchSstConstrainWalls(gas,b.gasGeometry.view(true,true),p,b.status.data(),stream),"gas-window SST walls",error)
        ||!launch(b,launchSstValidate(gas,p,b.status.data(),stream),"gas-window SST admissibility",error))return false;}
    if(!checked(b,error))return false;
    if(stage==1){gasWindowAuditKernel<<<1,1,0,stream>>>(gas,window.current->gas.data(),evaluation.gas.data(),state.particles.data(),window.current->particles.data(),state.particles.size(),state.particleRadiation.data(),state.particleBodyWork.data(),state.particleSupportWork.data(),state.particleSupportImpulse.data(),geometry,state.packetView(),state.physics.data(),dt,window.budget,state.trialBudget.data(),b.diagnostics.data(),b.status.data());
        if(!launch(b,static_cast<int>(cudaPeekAtLastError()),"audit gas-only accepted stage",error)||!checked(b,error))return false;}
    return true;
}
}
bool beginGasWindow(Backend& b,const WallProgram& program,std::string& error,const GasWindowLimits& limits){
    try{
        if(!b.ready||!b.fault.message.empty()){error=b.fault.message.empty()?"backend not ready":b.fault.message;return false;}
        if(b.pending){error="a gas window is already pending";return false;}
        if(!validateGasWallProgram(program,b.accepted,error))return false;
        const auto& p=b.model.physics;
        if(b.accepted.gas.empty()||b.accepted.solid.empty()||!b.accepted.normalEnthalpy.empty()){
            error="gas multirate window requires gas and 3D material inventories, not a resolved-normal standalone problem";return false;}
        if(p.enableParticles&&p.particle.contactDuration>0){error="finite-duration particle contact needs a CPU wall-temperature program and is not supported by gas windows";return false;}
        for(const auto& particle:b.accepted.particles)if(particle.state==ParticleContact){error="active finite-duration particle contact cannot enter a gas window";return false;}
        if(p.enableFilm&&p.material.enableSurfaceReactions)for(const auto& film:b.accepted.film)if(film.mass>0){error="wet surface reactions remain unsupported";return false;}
        if(p.enableFilm&&p.material.enablePoreOutflow&&p.permeability>0)for(const auto& film:b.accepted.film)if(film.mass>0){error="wet pore discharge remains unsupported";return false;}
        if(p.material.enableMelting)for(const auto& q:b.accepted.solid)if(q.porosity>p.tolerances.relativeGeometry){error="porous melting remains unsupported";return false;}
        for(const auto& v:program.surface.baseVelocity)if(mag(v)!=0){error="nonstationary material skeleton remains unsupported";return false;}
        for(std::size_t f=0;f<program.surface.area.size();++f){
            if(program.surface.gasFace[f]<0||program.surface.solidFace[f]<0||program.surface.solidCell[f]<0){error="gas windows require material-owned interface faces";return false;}}
        std::unique_ptr<PendingGasWindow> window(new PendingGasWindow);
        if(limits.maximumRecords==0||limits.maximumHistoryBytes<sizeof(GasIntervalRecord)){error="gas history capacity must be positive";return false;}
        window->limits=limits;window->staticGeometry=true;
        for(const auto& knot:program.knots){for(std::size_t i=0;i<knot.gasPoints.size();++i)window->staticGeometry=window->staticGeometry&&mag(knot.gasPoints[i]-b.accepted.gasMesh.points[i])==0;
            for(std::size_t i=0;i<knot.solidPoints.size();++i)window->staticGeometry=window->staticGeometry&&mag(knot.solidPoints[i]-b.accepted.solidMesh.points[i])==0;}
        if(window->staticGeometry)for(const auto& knot:program.knots)for(const auto& face:knot.faces)if(face.normalVelocity!=0||face.solidNormalVelocity!=0){error="stationary trajectory has nonzero wall recession program";return false;}
        window->program=program;window->gasMesh=b.accepted.gasMesh;window->solidMesh=b.accepted.solidMesh;
        window->surface=program.surface;window->time=program.interval.begin;window->budget=b.accepted.budget;
        const int count=2*program.surface.area.size()+2*b.accepted.particles.size();
        if(!window->history.begin(program.interval,error)||!window->reserve.reset(b.accepted.solid,b.accepted.film,b.accepted.time,error))return false;
        if(!program.donorPlan.knots.empty()&&!window->reserve.setPlan(program.donorPlan,error))return false;
        if(!window->current->initialize(b.accepted,p,b.fault,count)
            ||!b.midpoint->packets.resize(count,b.fault)||!b.trial->packets.resize(count,b.fault)
            ||!window->drive.resize(program.surface.area.size(),b.fault)||!window->driveGradient.resize(program.surface.area.size(),b.fault)||!window->traction.resize(program.surface.area.size(),b.fault)
            ||!b.stream.sync()){error=b.fault.message;return false;}
        std::vector<WallFaceSample> wallSamples;wallSamples.reserve(program.knots.size()*program.surface.area.size());
        for(const auto& knot:program.knots)wallSamples.insert(wallSamples.end(),knot.faces.begin(),knot.faces.end());
        if(!window->wall.upload(wallSamples,b.fault)){error=b.fault.message;return false;}
        if(window->staticGeometry){std::vector<Real> gs(b.accepted.gasMesh.owner.size(),0),ss(b.accepted.solidMesh.owner.size(),0);
            if(!b.gasGeometry.upload(window->gasMesh,window->gasMesh,window->gasMesh,gs,window->gasMesh.areaVectors,b.fault)
                ||!b.solidGeometry.upload(window->solidMesh,window->solidMesh,window->solidMesh,ss,window->solidMesh.areaVectors,b.fault)
                ||!b.surface.upload(window->surface,b.fault)||!b.endpointSurface.upload(window->surface,b.fault)){error=b.fault.message;return false;}}
        b.pending=std::move(window);error.clear();return true;
    }catch(const std::exception& e){error=std::string("begin gas window preserved macro base: ")+e.what();return false;}
}
static bool advanceGasMicrostepImpl(Backend& b,Real gasMaxDt,GasMicroReport& report,std::string& error){
    try{
        if(!b.pending||!b.ready||!b.fault.message.empty()){error=b.fault.message.empty()?"no gas window is pending":b.fault.message;return false;}
        auto& window=*b.pending;const auto& p=b.model.physics;
        if(!finite(gasMaxDt)||gasMaxDt<=0||window.time>=window.program.interval.end){error="invalid gas microstep cap or completed window";return false;}
        const std::size_t nf=window.program.surface.area.size(),np=window.current->particles.size();
        const std::size_t recordBytes=sizeof(GasIntervalRecord)+(2*nf+2*np)*sizeof(ExchangePacket)
            +nf*(sizeof(GasPrimitive)+sizeof(GasGradient)+sizeof(Vec3)+sizeof(Real)+sizeof(unsigned));
        // Reserve factor two bounds vector growth and container allocation overhead.
        if(window.history.records().size()>=window.limits.maximumRecords||recordBytes>window.limits.maximumHistoryBytes/2
            ||window.historyBytes>window.limits.maximumHistoryBytes-2*recordBytes){error="gas history capacity reached; shorten the coupling interval and replay from its base";return false;}
        const Real knot=nextGasWallKnot(window.program,window.time);
        Real dt=minValue(minValue(gasMaxDt,p.maxDt),knot-window.time);
        if(window.nextGasDt>0)dt=minValue(dt,window.nextGasDt);
        std::string failure;
        for(int retry=0;retry<=p.tolerances.maxRetries;++retry){
            // Exact final landing may be below minDt, but a failed sub-minimum
            // physical trial is never silently accepted.
            if(!(dt>0)||window.time+dt==window.time||(dt<p.minDt&&window.time+dt!=knot)){failure="minimum gas timestep reached: "+failure;break;}
            const Real end=dt==knot-window.time?knot:window.time+dt;
            dt=end-window.time;const Real middle=window.time+.5*dt;
            WallKnot start,mid,finish;
            if(!sampleGasWallProgram(window.program,window.time,start,failure,!window.staticGeometry)
                ||!sampleGasWallProgram(window.program,middle,mid,failure,!window.staticGeometry)
                ||!sampleGasWallProgram(window.program,end,finish,failure,!window.staticGeometry))break;
            HostMesh halfGas,halfSolid,fullGas,fullSolid;SurfaceMesh halfSurface,fullSurface;
            HostStageGeometry halfGasStage,halfSolidStage,fullGasStage,fullSolidStage;
            bool ok=gasWindowStage(b,window,*window.current,window.gasMesh,window.solidMesh,*b.midpoint,
                start,mid,.5*dt,0,halfGas,halfSolid,halfSurface,halfGasStage,halfSolidStage,failure);
            if(ok)ok=gasWindowStage(b,window,*b.midpoint,window.staticGeometry?window.gasMesh:halfGas,window.staticGeometry?window.solidMesh:halfSolid,*b.trial,mid,finish,dt,1,
                fullGas,fullSolid,fullSurface,fullGasStage,fullSolidStage,failure);
            if(ok){
                GasIntervalRecord record;record.microSequence=window.microSequence+1;record.begin=window.time;record.end=end;
                record.gasGeometry=window.staticGeometry?window.gasMesh.geometryVersion:fullGas.geometryVersion;record.solidGeometry=window.staticGeometry?window.solidMesh.geometryVersion:fullSolid.geometryVersion;
                Budget budget;
                if(!b.trial->packets.download(record.packets,b.stream.get())||!window.drive.download(record.gasTrace,b.stream.get())
                    ||!window.driveGradient.download(record.gasGradient,b.stream.get())||!window.traction.download(record.gasTraction,b.stream.get())||!b.trial->trialBudget.downloadOne(budget,b.stream.get())){error=b.fault.message;return false;}
                record.packets.erase(std::remove_if(record.packets.begin(),record.packets.end(),emptyParticleWindowPacket),record.packets.end());
                record.radiationEnergy.resize(mid.faces.size());record.radiationReceiver.resize(mid.faces.size());
                for(std::size_t f=0;f<mid.faces.size();++f){record.radiationEnergy[f]=mid.faces[f].radiationFlux*(window.staticGeometry?window.surface.area[f]:mag(halfSolid.areaVectors[window.surface.solidFace[f]]))*dt;
                    record.radiationReceiver[f]=mid.faces[f].primaryKind==ExchangeKind::GasFilm?ConsumeFilm:ConsumeSolid;}
                // Neither a rejected microstep nor a failed allocation can spend
                // the material reserve or mutate the accepted gas timeline.
                MaterialDonorReserve::Transaction reserveTransaction;
                ok=window.reserve.prepare(record,reserveTransaction,failure);
                const Real gcl=window.staticGeometry?0:maxValue(maximumGclResidual(window.gasMesh,fullGas,fullGasStage.sweptVolume),maximumGclResidual(window.solidMesh,fullSolid,fullSolidStage.sweptVolume));
                if(!finite(gcl)){ok=false;failure="nonfinite gas-window geometric conservation residual";}
                if(ok&&!window.reserve.readyToCommit(reserveTransaction)){ok=false;failure="stale sparse material donor transaction";}
                if(ok)ok=window.history.appendAccepted(record,failure);
                if(ok){
                    // The unchanged reserve was checked before history append;
                    // publication is allocation-free and statically noexcept.
                    window.reserve.commitPrepared(std::move(reserveTransaction));
                    budget.gclResidual+=gcl;window.current.swap(b.trial);
                    if(!window.staticGeometry){window.gasMesh=std::move(fullGas);window.solidMesh=std::move(fullSolid);window.surface=std::move(fullSurface);}
                    window.gasStages[0]=std::move(halfGasStage);window.gasStages[1]=std::move(fullGasStage);
                    window.solidStages[0]=std::move(halfSolidStage);window.solidStages[1]=std::move(fullSolidStage);
                    window.time=end;++window.microSequence;window.rejectedTrials+=retry;window.budget=budget;window.historyBytes+=2*recordBytes;window.lastGasDt=dt;
                    window.nextGasDt=minValue(minValue(gasMaxDt,b.nextStableDt),dt*(retry?1:1.2));
                    report.acceptedDt=dt;report.nextGasDt=window.nextGasDt;report.time=end;report.microSequence=window.microSequence;report.rejectedTrials=retry;
                    error.clear();return true;
                }
            }
            if(!b.fault.message.empty()){error=b.fault.message;return false;}
            if(b.lastStatus.code==static_cast<int>(ErrorCode::Unsupported)){error=failure;return false;}
            if(b.lastStatus.location==ErrorLocation::Particle&&b.lastStatus.code==static_cast<int>(ErrorCode::Inventory)&&b.lastStatus.value<0)dt=particleEventRetryDt(dt,b.lastStatus.value);
            else if(b.nextStableDt>0&&b.nextStableDt<dt)dt=minValue(.5*dt,b.nextStableDt);
            else dt*=.5;
        }
        error="gas window retains its previous accepted microstate: "+failure;return false;
    }catch(const std::exception& e){error=std::string("gas microstep preserved synchronized macro state: ")+e.what();return false;}
}
bool advanceGasMicrostep(Backend& b,Real gasMaxDt,GasMicroReport& report,std::string& error){
    report=GasMicroReport{};
    const bool result=advanceGasMicrostepImpl(b,gasMaxDt,report,error);
    if(!result)report.recoverable=b.fault.message.empty()&&b.lastStatus.code!=static_cast<int>(ErrorCode::Unsupported)
        &&error.find("unsupported")==std::string::npos&&error.find("topology")==std::string::npos
        &&error.find("invalid gas microstep")==std::string::npos;
    return result;
}
bool gasWindowHistory(const Backend& b,IntervalHistory& history,std::string& error){
    if(!b.pending){error="no pending gas window";return false;}
    try{history=b.pending->history;error.clear();return true;}catch(const std::exception& e){error=e.what();return false;}
}
bool gasWindowStageGeometry(const Backend& b,std::array<HostStageGeometry,2>& gas,
    std::array<HostStageGeometry,2>& solid,std::string& error){
    if(!b.pending||b.pending->microSequence==0){error="no accepted gas microstep geometry";return false;}
    try{if(b.pending->staticGeometry){const auto& w=*b.pending;
            const std::vector<Real> gs(w.gasMesh.owner.size(),0),ss(w.solidMesh.owner.size(),0);
            setWindowStage(w.gasMesh,w.gasMesh,w.gasMesh,.5*w.lastGasDt,gs,w.gasMesh.areaVectors,gas[0]);
            setWindowStage(w.gasMesh,w.gasMesh,w.gasMesh,w.lastGasDt,gs,w.gasMesh.areaVectors,gas[1]);
            setWindowStage(w.solidMesh,w.solidMesh,w.solidMesh,.5*w.lastGasDt,ss,w.solidMesh.areaVectors,solid[0]);
            setWindowStage(w.solidMesh,w.solidMesh,w.solidMesh,w.lastGasDt,ss,w.solidMesh.areaVectors,solid[1]);
        }else{gas=b.pending->gasStages;solid=b.pending->solidStages;}error.clear();return true;}catch(const std::exception& e){error=e.what();return false;}
}
bool downloadGasWindow(const Backend& b,HostState& result,std::string& error){
    if(!b.pending||!b.pending->history.complete()){error="only a complete gas window can be downloaded for material correction";return false;}
    try{const auto& window=*b.pending;HostState state=b.accepted;
        state.gasMesh=window.gasMesh;state.solidMesh=window.solidMesh;state.surface=window.surface;
        if(!gasWindowStageGeometry(b,state.gasStages,state.solidStages,error))return false;
        state.time=window.time;state.nextDt=window.nextGasDt;state.lastAcceptedDt=window.time-b.accepted.time;
        state.budget=window.budget;state.ledger.clear();state.filmStorage.clear();
        state.rejectedSteps=b.accepted.rejectedSteps+window.rejectedTrials;
        if(!window.current->gas.download(state.gas,b.stream.get())||!window.current->sst.download(state.sst,b.stream.get())
            ||!window.current->particles.download(state.particles,b.stream.get())||!window.current->voidFraction.download(state.gasVoidFraction,b.stream.get())
            ||!b.gasGeometry.gasBoundary.download(state.gasMesh.boundaryPrimitive,b.stream.get())){error=b.fault.message;return false;}
        result=std::move(state);error.clear();return true;
    }catch(const std::exception& e){error=std::string("download provisional gas window: ")+e.what();return false;}
}
bool rollbackGasWindow(Backend& b,std::string& error){
    if(!b.pending){error.clear();return true;}
    // Device execution must finish before releasing the trial owner. Even if
    // CUDA fails, accepted host and device inventory pointers were never swapped.
    const bool synchronized=b.stream.sync();b.pending.reset();
    if(!synchronized){error=b.fault.message;return false;}
    error.clear();return true;
}
bool commitGasWindow(Backend& b,const HostState& candidate,std::string& error){
    try{
        if(!b.pending||!b.pending->history.complete()||candidate.time!=b.pending->program.interval.end){error="gas/material commit requires a completed matching window";return false;}
        if(candidate.gasMesh.points.size()!=b.pending->gasMesh.points.size()||candidate.solidMesh.points.size()!=b.pending->solidMesh.points.size()){
            error="gas/material commit topology differs from consumed trajectory";return false;}
        for(std::size_t i=0;i<candidate.gasMesh.points.size();++i)if(mag(candidate.gasMesh.points[i]-b.pending->gasMesh.points[i])!=0){error="gas commit changed consumed trajectory";return false;}
        for(std::size_t i=0;i<candidate.solidMesh.points.size();++i)if(mag(candidate.solidMesh.points[i]-b.pending->solidMesh.points[i])!=0){error="material commit requires gas trajectory replay";return false;}
        HostState actual;if(!downloadGasWindow(b,actual,error))return false;
        if(candidate.gas.size()!=actual.gas.size()||candidate.sst.size()!=actual.sst.size()||candidate.particles.size()!=actual.particles.size()){
            error="candidate modified GPU-owned inventory layout";return false;}
        for(std::size_t c=0;c<actual.gas.size();++c){const auto& x=candidate.gas[c];const auto& y=actual.gas[c];
            if(x.mass!=y.mass||x.energy!=y.energy||mag(x.momentum-y.momentum)!=0){error="material candidate modified accepted gas inventory";return false;}
            for(int s=0;s<Ns;++s)if(x.species[s]!=y.species[s]){error="material candidate modified accepted gas species";return false;}}
        for(std::size_t c=0;c<actual.sst.size();++c)if(candidate.sst[c].rhoK!=actual.sst[c].rhoK||candidate.sst[c].rhoOmega!=actual.sst[c].rhoOmega){error="material candidate modified SST inventory";return false;}
        // The committed candidate is constructed from the exact gas snapshot;
        // restore the authoritative particle bytes, including random counters.
        if(b.accepted.acceptedSteps==std::numeric_limits<std::uint64_t>::max()||b.accepted.commitSequence==std::numeric_limits<std::uint64_t>::max()){
            error="macro commit counter overflow";return false;}
        HostState synchronized=candidate;synchronized.particles=actual.particles;synchronized.gasVoidFraction=actual.gasVoidFraction;
        synchronized.acceptedSteps=b.accepted.acceptedSteps+1;synchronized.commitSequence=b.accepted.commitSequence+1;
        synchronized.lastAcceptedDt=candidate.time-b.accepted.time;
        std::unique_ptr<Backend> replacement(createBackend(b.model,synchronized,error));if(!replacement)return false;
        if(!b.stream.sync()){error=b.fault.message;return false;}
        if(!publishResourceReplacement(b.committed,b.midpoint,b.trial,replacement->committed,replacement->midpoint,replacement->trial,b.fault,replacement->fault,error))return false;
        b.accepted=std::move(replacement->accepted);b.interfaceCount=replacement->interfaceCount;b.totalPackets=replacement->totalPackets;
        b.pending.reset();error.clear();return true;
    }catch(const std::exception& e){error=std::string("gas/material commit preserved synchronized base: ")+e.what();return false;}
}
