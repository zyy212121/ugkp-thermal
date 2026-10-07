#include "coupling/Coordinator.H"
#include "gas/BoundarySample.H"
#include "gpu/BackendResources.H"
#include "gas/KernelSupport.cuh"
#include "ablation/InterfaceKernels.H"
#include "particles/ParticleMath.H"
namespace chmt { namespace {
__global__ void storageKernel(const FilmQ* base,const FilmAux* baseAux,FilmView film,GasView gas,
    GeometryView geometry,SurfaceView surface,PhysicsConfig physics,Real dt,std::uint64_t step,int stage,DeviceStatus* status){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=surface.nFaces||gasStopped(status))return;
    if(film.storage){auto& u=film.storage[f];u=FilmPressureVolumeUpdate{};u.step=step;u.stage=stage;u.geometry=geometry.geometryVersion;u.filmFace=f;
        u.face=surface.gasFace&&surface.gasFace[f]>=0?geometry.faceIds[surface.gasFace[f]]:surface.persistentId[f];
        u.oldPV=baseAux[f].pressure*base[f].mass/physics.liquid.rho;}
    if(surface.gasFace&&surface.gasFace[f]>=0){const int c=geometry.owner[surface.gasFace[f]];GasPrimitive w;
        if(!recoverGas(gas.q[c],geometry.evaluationVolume[c],physics,w)){gasDeviceError(status,ErrorCode::PropertyRange,c,0,ErrorLocation::Cell);return;}
        if(film.aux)film.aux[f].pressure=w.pressure;
    }
    (void)dt;
}
__global__ void scatterKernel(SurfaceView surface,GasPrimitive* destination,DeviceStatus* status){int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=surface.nFaces||gasStopped(status))return;
    if(surface.gasFace[f]>=0)destination[surface.gasFace[f]]=surface.interfaceGasState[f];}
__global__ void solidUpdateKernel(const SolidQ* base,SolidView solid,PacketView packets,SurfaceView surface,Real dt,DeviceStatus* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=solid.nCells||gasStopped(status))return;
    SolidQ next=assembleSolidCandidate(base[c],solid.rhs[c],dt,packets.packets,packets.count,c);
    for(int f=0;f<surface.nFaces;++f)if(surface.gasFace&&surface.gasFace[f]>=0){const auto& p=packets.packets[3*f];if(p.kind==ExchangeKind::GasSolid&&p.solidCell==c)next.energy+=dt*surface.radiationPower[f];}
    if(!validSolidInventory(next)){gasDeviceError(status,ErrorCode::Inventory,c,next.energy,ErrorLocation::Cell);return;}solid.q[c]=next;
}
__global__ void filmUpdateKernel(const FilmQ* base,FilmView film,GasView gas,GeometryView endpoint,
    SurfaceView surface,PhysicsConfig physics,PacketView packets,Real dt,DeviceStatus* status){
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=film.nFaces||gasStopped(status))return;
    Real pressure=film.aux[f].pressure,radiation=0;
    if(surface.gasFace&&surface.gasFace[f]>=0){const int c=endpoint.owner[surface.gasFace[f]];GasPrimitive w;
        if(!recoverGas(gas.q[c],endpoint.newVolume[c],physics,w)){gasDeviceError(status,ErrorCode::PropertyRange,c,0,ErrorLocation::Cell);return;}
        pressure=w.pressure;if(filmReceivesSurfaceRadiation(true,&packets.packets[3*f]))radiation=surface.radiationPower[f];}
    auto storage=film.storage[f];storage.newPV=0;
    FilmQ next=assembleFilmCandidate(base[f],film.rhs[f],dt,packets.packets,packets.count,f,storage,radiation);
    storage.newPV=pressure*next.mass/physics.liquid.rho;next.enthalpy+=storage.newPV;
    if(!validFilmInventory(next)){gasDeviceError(status,ErrorCode::Inventory,f,next.enthalpy,ErrorLocation::Face);return;}
    film.q[f]=next;film.aux[f].pressure=pressure;film.storage[f]=storage;
}
__global__ void frontsKernel(const FilmAux* base,FilmAux* candidate,int count,Real dt){int f=blockIdx.x*blockDim.x+threadIdx.x;if(f<count){candidate[f].solidFront=base[f].solidFront+dt*candidate[f].solidNormalVelocity;candidate[f].gasFront=base[f].gasFront+dt*candidate[f].normalVelocity;}}
__global__ void normalUpdateKernel(const Real* baseH,const Real* baseHeat,NormalThermalView v,Real dt,DeviceStatus* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.nFaces||gasStopped(status))return;
    for(int i=v.offsets[c];i<v.offsets[c+1];++i){v.enthalpy[i]=baseH[i]+dt*v.rhs[i];if(!finite(v.enthalpy[i]))gasDeviceError(status,ErrorCode::Nonfinite,i,v.enthalpy[i],ErrorLocation::Layer);}
    v.acceptedWallHeat[c]=baseHeat[c]+dt*(v.bottomHeatRate[c]+v.topHeatRate[c]);
}
__global__ void wallTraceKernel(GasView gas,GeometryView endpoint,GasPrimitive* wall,PhysicsConfig p,DeviceStatus* status){
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=endpoint.nFaces||gasStopped(status))return;
    const auto kind=endpoint.boundaryKind[f];if(kind!=BoundaryKind::NoSlip&&kind!=BoundaryKind::Interface)return;
    if(kind==BoundaryKind::Interface){wall[f]=endpoint.boundaryPrimitive[f];return;}
    const int c=endpoint.owner[f];GasPrimitive w;if(!recoverGas(gas.q[c],endpoint.newVolume[c],p,w)){gasDeviceError(status,ErrorCode::PropertyRange,c,0,ErrorLocation::Cell);return;}
    GasPrimitive trace;
    if(!wallThermodynamicTrace(w,endpoint.boundaryPrimitive[f],fixedTemperature(endpoint.thermalBoundary,f),p,trace)){
        gasDeviceError(status,ErrorCode::PropertyRange,f,w.temperature,ErrorLocation::Face);return;
    }
    wall[f]=trace;
}
__device__ bool externalFace(BoundaryKind k){return k!=BoundaryKind::Internal&&k!=BoundaryKind::Periodic&&k!=BoundaryKind::Interface&&k!=BoundaryKind::Empty;}
__global__ void auditKernel(GasView gas,const GasQ* baseGas,const GasQ* evaluationGas,
    SolidView solid,const SolidQ* baseSolid,const SolidQ* evaluationSolid,FilmView film,const FilmQ* baseFilm,const FilmAux* baseAux,
    NormalThermalView normal,const Real* baseH,const Real* oldNormalPressure,
    const ParticleQ* particles,const ParticleQ* baseParticles,int np,const Real* particleRadiation,const Real* particleBody,const Real* particleSupport,const Vec3* particleImpulse,
    GeometryView gg,GeometryView sg,SurfaceView surface,PacketView packets,const PhysicsConfig* physics,Real dt,Budget original,Budget* output,StageDiagnostics* diagnostics,DeviceStatus* status){
    if(blockIdx.x||threadIdx.x||gasStopped(status))return;
    const PhysicsConfig& p=*physics;Budget budget=original;StageDiagnostics result;result.minimumMass=std::numeric_limits<Real>::max();result.minimumTemperature=std::numeric_limits<Real>::max();result.dtLimit=std::numeric_limits<Real>::max();
    Real external=0,deltaK=0,kineticAdv=0,filmDeltaU=0,filmOut=0,filmInput=0;
    for(int c=0;c<gas.nCells;++c){result.energyBefore+=baseGas[c].energy;result.energyAfter+=gas.q[c].energy;result.minimumMass=minValue(result.minimumMass,gas.q[c].mass);result.minimumTemperature=minValue(result.minimumTemperature,gas.primitive[c].temperature);
        const Real body=dt*dot(p.gravity,evaluationGas[c].momentum);budget.bodyWork+=body;external+=body;
        if(p.enableSst){budget.turbulenceProduction+=dt*gas.sst.production[c];budget.turbulenceDissipation+=dt*gas.sst.dissipation[c];budget.turbulenceOmegaConstraint+=gas.sst.omegaConstraint[c];}
    }
    if(p.enableSst){budget.turbulenceInventory=0;for(int c=0;c<gas.nCells;++c)budget.turbulenceInventory+=gas.sst.q[c].rhoK;}
    for(int f=0;f<gg.nFaces;++f)if(externalFace(gg.boundaryKind[f])){const auto flux=gas.faceFlux[f];budget.boundaryMass+=dt*flux.mass;budget.boundaryEnergy+=dt*flux.energy;budget.boundaryMomentum+=flux.momentum*dt;external-=dt*flux.energy;
        for(int s=0;s<Ns;++s){budget.boundarySpecies[s]+=dt*flux.species[s];for(int e=0;e<p.nElements;++e)budget.boundaryElements[e]+=dt*flux.species[s]*p.species[s].element[e];}
        if(p.enableSst)budget.turbulenceBoundaryFlux+=dt*gas.sst.faceFlux[f].rhoK;}
    for(int c=0;c<solid.nCells;++c){Real totalMass=0;for(int j=0;j<Nc;++j)totalMass+=solid.q[c].condensed[j];for(int s=0;s<Ns;++s)totalMass+=solid.q[c].pore[s];result.minimumMass=minValue(result.minimumMass,totalMass);result.energyBefore+=baseSolid[c].energy;result.energyAfter+=solid.q[c].energy;result.minimumTemperature=minValue(result.minimumTemperature,solid.temperature[c]);Real poreMass=0;for(int s=0;s<Ns;++s)poreMass+=evaluationSolid[c].pore[s];const Real body=dt*poreMass*dot(p.gravity,solid.poreVelocity[c]);budget.bodyWork+=body;external+=body;}
    for(int f=0;f<sg.nFaces;++f)if(externalFace(sg.boundaryKind[f])){const auto flux=solid.faceFlux[f];budget.boundaryEnergy+=dt*flux.energy;external-=dt*flux.energy;
        for(int c=0;c<Nc;++c){budget.boundaryMass+=dt*flux.condensed[c];for(int e=0;e<p.nElements;++e)budget.boundaryElements[e]+=dt*flux.condensed[c]*p.condensed[c].element[e];}
        for(int s=0;s<Ns;++s){budget.boundaryMass+=dt*flux.pore[s];budget.boundarySpecies[s]+=dt*flux.pore[s];for(int e=0;e<p.nElements;++e)budget.boundaryElements[e]+=dt*flux.pore[s]*p.species[s].element[e];}}
    for(int f=0;f<surface.nFaces;++f)if(surface.gasFace&&surface.gasFace[f]>=0){const Real radiation=dt*surface.radiationPower[f];budget.radiation+=radiation;external+=radiation;}
    for(int f=0;f<film.nFaces;++f){result.minimumMass=minValue(result.minimumMass,film.q[f].mass);const Real u0=baseFilm[f].enthalpy-film.storage[f].oldPV,u1=film.q[f].enthalpy-film.storage[f].newPV;
        const auto rate=film.rateBudget[f];result.energyBefore+=u0+baseAux[f].kineticEnergy;result.energyAfter+=u1+film.aux[f].kineticEnergy;
        deltaK+=film.aux[f].kineticEnergy-baseAux[f].kineticEnergy;filmDeltaU+=u1-u0;kineticAdv+=dt*(rate.edgeKineticOutflow+rate.interfaceKineticOutflow);
        filmOut+=dt*(rate.edgeEnergyOutflow+rate.edgeKineticOutflow);const bool coupledRadiation=surface.gasFace&&surface.gasFace[f]>=0;
        const bool filmRadiation=filmReceivesSurfaceRadiation(coupledRadiation,coupledRadiation?&packets.packets[3*f]:nullptr);
        filmInput+=dt*(rate.bodyPower+rate.supportPower+rate.prescribedTopPower+(filmRadiation?rate.radiationPower:0));
        const Real mechanical=dt*(rate.bodyPower+rate.supportPower+rate.prescribedTopPower);
        budget.bodyWork+=dt*rate.bodyPower;budget.supportWork+=dt*(rate.supportPower+rate.prescribedTopPower);external+=mechanical-dt*(rate.edgeEnergyOutflow+rate.edgeKineticOutflow);budget.boundaryEnergy+=dt*(rate.edgeEnergyOutflow+rate.edgeKineticOutflow);
        if(!surface.gasFace||surface.gasFace[f]<0){budget.radiation+=dt*rate.radiationPower;external+=dt*rate.radiationPower;}
        budget.filmPressureVolume+=filmPressureVolumeDelta(film.storage[f]);film.storage[f].consumed=true;
        if(film.q[f].mass>0)result.minimumTemperature=minValue(result.minimumTemperature,film.aux[f].temperature);
    }
    for(int i=0;i<packets.count;++i){auto& packet=packets.packets[i];if(requiredConsumers(packet.kind)&ConsumeFilm){filmOut+=(packet.kind==ExchangeKind::SolidFilm?-1:1)*packet.energy;}
        if(!accountAssembledPacket(packet,budget,p.tolerances,status))return;}
    result.reducedResidual=filmDeltaU+filmOut-filmInput-kineticAdv;
    result.kineticDefect=deltaK+kineticAdv;budget.filmReducedResidual+=result.reducedResidual;budget.filmKineticStorage+=deltaK;budget.filmKineticDefect+=result.kineticDefect;
    for(int col=0;col<normal.nFaces;++col){for(int i=normal.offsets[col];i<normal.offsets[col+1];++i){result.minimumMass=minValue(result.minimumMass,normal.mass[i]);result.energyBefore+=baseH[i]-normal.newVolume[i]*oldNormalPressure[col];result.energyAfter+=normal.enthalpy[i]-normal.newVolume[i]*normal.pressure[col];result.minimumTemperature=minValue(result.minimumTemperature,normal.temperature[i]);}const Real heat=dt*(normal.bottomHeatRate[col]+normal.topHeatRate[col]);budget.boundaryEnergy-=heat;external+=heat;}
    for(int i=0;i<np;++i){result.energyBefore+=particleTotalEnergy(baseParticles[i]);result.energyAfter+=particleTotalEnergy(particles[i]);budget.radiation+=particleRadiation[i];budget.bodyWork+=particleBody[i];budget.supportWork+=particleSupport[i];budget.supportImpulse+=particleImpulse[i];external+=particleRadiation[i]+particleBody[i]+particleSupport[i];}
    const Real remainder=result.energyAfter-result.energyBefore-external-result.kineticDefect;
    budget.numericalEnergyResidual+=remainder;
    if(!validBudget(budget)||!finite(remainder)||!finite(result.reducedResidual)||!finite(result.kineticDefect)){gasDeviceError(status,ErrorCode::Nonfinite,0,remainder,ErrorLocation::Configuration);return;}
    *output=budget;*diagnostics=result;
}
} // namespace
int launchInitializeStorage(const StateBuffers& base,StateBuffers& state,GeometryBuffers& gasGeometry,SurfaceBuffers& surface,const PhysicsConfig& p,Real dt,std::uint64_t step,int stage,DeviceStatus* status,void* stream){
    if(!surface.nFaces||!state.filmAux.size())return 0;auto gas=state.gasView(base,status,gasGeometry.wallDistance.data());GeometryBuffers absent;auto sv=surface.view(absent);
    storageKernel<<<(surface.nFaces+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(base.film.data(),base.filmAux.data(),state.filmView(base,status),gas,gasGeometry.view(),sv,p,dt,step,stage,status);return launchError();}
int launchScatterInterface(SurfaceBuffers& surface,GeometryBuffers& geometry,DeviceStatus* status,void* stream){if(!surface.nFaces)return 0;GeometryBuffers absent;scatterKernel<<<(surface.nFaces+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(surface.view(absent),geometry.gasBoundary.data(),status);return launchError();}
int launchParticipantUpdate(const StateBuffers& base,StateBuffers& state,GeometryBuffers& gasGeometry,SurfaceBuffers& surface,SurfaceBuffers& endpoint,const PhysicsConfig& p,Real dt,DeviceStatus* status,void* stream){
    auto s=static_cast<cudaStream_t>(stream);GeometryBuffers absent;auto sv=surface.view(absent);auto packets=state.packetView();
    if(state.solid.size()){solidUpdateKernel<<<(state.solid.size()+127)/128,128,0,s>>>(base.solid.data(),state.solidView(base,status),packets,sv,dt,status);if(int e=launchError())return e;}
    if(state.film.size()){filmUpdateKernel<<<(state.film.size()+127)/128,128,0,s>>>(base.film.data(),state.filmView(base,status),state.gasView(base,status,nullptr),gasGeometry.view(true),sv,p,packets,dt,status);if(int e=launchError())return e;}
    if(state.filmAux.size()){frontsKernel<<<(state.filmAux.size()+127)/128,128,0,s>>>(base.filmAux.data(),state.filmAux.data(),state.filmAux.size(),dt);if(int e=launchError())return e;}
    if(state.normalH.size()){auto v=endpoint.normalView(state);normalUpdateKernel<<<(v.nFaces+63)/64,64,0,s>>>(base.normalH.data(),base.normalWallHeat.data(),v,dt,status);if(int e=launchError())return e;}return 0;
}
int launchWallTrace(GasView gas,GeometryView geometry,GasPrimitive* trace,const PhysicsConfig& p,DeviceStatus* status,void* stream){if(!geometry.nFaces)return 0;wallTraceKernel<<<(geometry.nFaces+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(gas,geometry,trace,p,status);return launchError();}
int launchStageAudit(const StateBuffers& base,const StateBuffers& evaluation,StateBuffers& state,GeometryBuffers& gasGeometry,GeometryBuffers& solidGeometry,SurfaceBuffers& surface,SurfaceBuffers& endpoint,const Real* oldPressure,const PhysicsConfig& p,Real dt,const Budget& prior,StageDiagnostics* diagnostics,DeviceStatus* status,void* stream){
    auto normal=endpoint.normalView(state);
    auditKernel<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(state.gasView(base,status,gasGeometry.wallDistance.data()),base.gas.data(),evaluation.gas.data(),state.solidView(base,status),base.solid.data(),evaluation.solid.data(),state.filmView(base,status),base.film.data(),base.filmAux.data(),normal,base.normalH.data(),oldPressure,state.particles.data(),base.particles.data(),state.particles.size(),state.particleRadiation.data(),state.particleBodyWork.data(),state.particleSupportWork.data(),state.particleSupportImpulse.data(),gasGeometry.view(),solidGeometry.view(),surface.view(solidGeometry),state.packetView(),state.physics.data(),dt,prior,state.trialBudget.data(),diagnostics,status);return launchError();}
} // namespace chmt
