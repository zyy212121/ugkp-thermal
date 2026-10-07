#include "gpu/Backend.H"
#include "gpu/BackendResources.H"
#include "gpu/ResourceTransaction.H"
#include "gpu/StateValidation.H"
#include "coupling/Coordinator.H"
#include "gas/GasKernels.H"
#include "gas/KernelSupport.cuh"
#include "gas/SstKernels.H"
#include "materials/MaterialKernels.H"
#include "materials/ReactionStepControl.H"
#include "materials/DarcyGradient.H"
#include "film/FilmKernels.H"
#include "film/ResolvedNormalThermal.H"
#include "ablation/InterfaceKernels.H"
#include "particles/ParticleKernels.H"
#include "mesh/Geometry.H"
#include "mesh/Motion.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "mesh/Remap1D.H"
#include "CHMTBuildIdentity.H"
#include "gpu/GasWallMath.H"
#include "gpu/GasStability.H"
#include "gpu/GasWindowProgram.H"
#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "gpu/GasWindowState.H"
namespace chmt {
struct Backend {
    CudaFault fault;
    DeviceStream stream;
    ModelConfig model;
    HostState accepted;
    std::unique_ptr<StateBuffers> committed{new StateBuffers},midpoint{new StateBuffers},trial{new StateBuffers};
    GeometryBuffers gasGeometry,solidGeometry;
    SurfaceBuffers surface,endpointSurface,baseSurface;
    Buffer<DeviceStatus> status;
    Buffer<StageDiagnostics> diagnostics;
    Buffer<Real> stableDt;
    Real nextStableDt=0,finalCouplingResidual=0;
    int stageIterations=0;
    int interfaceCount=0,totalPackets=0;
    DeviceStatus lastStatus{};
    bool ready=false;
    std::unique_ptr<PendingGasWindow> pending;
};
namespace {
__global__ void stableStepKernel(GasView gas,GeometryView geometry,SolidView solid,GeometryView solidGeometry,FilmView film,SurfaceView surface,NormalThermalView normal,const PhysicsConfig* physics,Real interval,Real* result){
    if(blockIdx.x||threadIdx.x)return;const PhysicsConfig& p=*physics;Real limit=p.maxDt;
    for(int c=0;c<gas.nCells;++c){Real sum=0,diffusion=0;const auto w=gas.primitive[c];
        for(int k=geometry.cellFaceOffsets[c];k<geometry.cellFaceOffsets[c+1];++k){const int f=geometry.cellFaces[k];if(geometry.boundaryKind[f]==BoundaryKind::Empty)continue;const Real area=mag(geometry.areaVector[f]);const Vec3 n=geometry.areaVector[f]/area;
            const Real wn=geometry.sweptVolume[f]/(interval*area);sum+=area*(absValue(dot(w.velocity,n)-wn)+w.soundSpeed);
            const Real distance=mag(geometry.faceCentre[f]-geometry.cellCentre[c]);Real D=0;for(int s=0;s<Ns;++s)D=maxValue(D,p.gasDiffusivity[s]);Real cv=0;for(int s=0;s<Ns;++s)cv+=w.Y[s]*(p.species[s].cp0-p.species[s].R+p.species[s].cp1*w.temperature);
            const Real nut=p.enableSst?gas.sst.primitive[c].eddyViscosity:0;
            Real R=0;for(int species=0;species<Ns;++species)R+=w.Y[species]*p.species[species].R;
            if(p.enableSst)D+=nut/p.sst.turbulentSchmidt;
            const Real thermal=p.gasConductivity/(w.rho*cv)+(p.enableSst?nut*(cv+R)/(cv*p.sst.turbulentPrandtl):0);
            if(distance>0)diffusion+=area*maxValue(maxValue(p.gasViscosity/w.rho+nut,D),thermal)/distance;}
        if(sum+2*diffusion>0)limit=minValue(limit,p.cfl*geometry.evaluationVolume[c]/(sum+2*diffusion));}
    for(int c=0;c<solid.nCells;++c){const auto q=solid.q[c];MaterialPrimitive material;
        if(!recoverMaterial(q,solidGeometry.evaluationVolume[c],p,material)){*result=0;return;}
        Real capacity=0,poreMass=0;for(int j=0;j<Nc;++j)capacity+=q.condensed[j]*(p.condensed[j].cp0+p.condensed[j].cp1*material.temperature);
        for(int species=0;species<Ns;++species){poreMass+=q.pore[species];capacity+=q.pore[species]*(p.species[species].cp0-p.species[species].R+p.species[species].cp1*material.temperature);}
        const Real volume=solidGeometry.evaluationVolume[c];const Real storage=porePressureStorage(q,material.porosity*volume,material.temperature,p);
        Real pressureRate=0,thermalRate=0,advectionRate=0;
        for(int entry=solidGeometry.cellFaceOffsets[c];entry<solidGeometry.cellFaceOffsets[c+1];++entry){const int face=solidGeometry.cellFaces[entry];const auto kind=solidGeometry.boundaryKind[face];if(kind==BoundaryKind::Empty)continue;
            const Real area=mag(solidGeometry.areaVector[face]);const int other=neighbourCell(c,face,solidGeometry);
            if(other>=0){const Vec3 displacement=neighbourDisplacement(c,face,other,solidGeometry);
                const Real distance=mag(displacement);MaterialPrimitive neighbour;
                if(!recoverMaterial(solid.q[other],solidGeometry.evaluationVolume[other],p,neighbour)){*result=0;return;}
                const Real faceDensity=maxValue(material.poreDensity,neighbour.poreDensity);
                const Vec3 normal=solidGeometry.areaVector[face]*(solidGeometry.cellFaceSigns[entry]/area);
                Real inverseLength=0;
                if(!darcyPressureStencilInverseLength(displacement,normal,solid.pressureGradientSensitivity[c],
                    solid.pressureGradientSensitivity[other],inverseLength)){*result=0;return;}
                if(storage>0)pressureRate+=darcyPressureRelaxation(storage,p.permeability,p.poreViscosity,faceDensity,area,1/inverseLength);
                const Real conductance=area*seriesConductance(material.conductivity,neighbour.conductivity,distance/2,distance/2);
                if(capacity>0)thermalRate+=conductance/capacity;
                advectionRate+=area*mag(solid.poreVelocity[c])/volume;
            }else if(kind==BoundaryKind::Interface){
                if(p.material.enablePoreOutflow&&p.permeability>0&&storage>0){for(int f=0;f<surface.nFaces;++f)if(surface.solidCell&&surface.solidCell[f]==c&&surface.solidFace[f]==face&&surface.gasFace[f]>=0){
                    const int gc=geometry.owner[surface.gasFace[f]];const auto w=gas.primitive[gc];const Real rho=maxValue(material.poreDensity,w.rho);
                    const Real conductance=p.permeability*rho*area/(p.poreViscosity*surface.solidDistance[f]);pressureRate+=conductance/storage;
                    Real R=0,cv=0;for(int species=0;species<Ns;++species){R+=w.Y[species]*p.species[species].R;cv+=w.Y[species]*(p.species[species].cp0-p.species[species].R+p.species[species].cp1*w.temperature);}
                    const Real gasStorage=geometry.evaluationVolume[gc]/(R*w.temperature);
                    pressureRate+=conductance*(cv+R)/(cv*gasStorage);
                }}
            }else if(solidGeometry.boundaryPrimitive&&solidGeometry.boundaryPrimitive[face].temperature>0){const Real distance=mag(solidGeometry.faceCentre[face]-solidGeometry.cellCentre[c]);if(capacity>0&&distance>0)thermalRate+=area*material.conductivity/(distance*capacity);}
            advectionRate+=absValue(solidGeometry.sweptVolume[face])/interval/volume;
        }
        if(pressureRate+thermalRate+advectionRate>0)limit=minValue(limit,p.cfl/(pressureRate+thermalRate+advectionRate));
        if(p.enableReactions){SolidQ reactionTrial;Real reactionDt=0;bool reactionActive=false;
            // Share the CPU-local empty-reactant Jacobian and nonlinear defect
            // bound. This selects a timestep only; legacy participant assembly
            // and its explicit temporal scheme remain unchanged.
            if(!reactionStepCandidate(q,volume,p,minValue(interval,limit),p.cfl,reactionTrial,reactionDt,reactionActive)){*result=0;return;}
            if(reactionActive)limit=minValue(limit,reactionDt);}
        (void)poreMass;
    }
    for(int f=0;f<surface.nFaces;++f)if(surface.gasFace&&surface.gasFace[f]>=0){
        const int gc=geometry.owner[surface.gasFace[f]],sc=surface.solidCell?surface.solidCell[f]:-1;const Real area=surface.area[f];const auto w=gas.primitive[gc];Real Cg=0,Cs=0,Cf=0,ks=0;
        for(int species=0;species<Ns;++species)Cg+=gas.q[gc].species[species]*(p.species[species].cp0-p.species[species].R+p.species[species].cp1*w.temperature);
        if(sc>=0){MaterialPrimitive material;if(!recoverMaterial(solid.q[sc],solidGeometry.evaluationVolume[sc],p,material)){*result=0;return;}ks=material.conductivity;
            for(int j=0;j<Nc;++j)Cs+=solid.q[sc].condensed[j]*(p.condensed[j].cp0+p.condensed[j].cp1*material.temperature);
            for(int species=0;species<Ns;++species)Cs+=solid.q[sc].pore[species]*(p.species[species].cp0-p.species[species].R+p.species[species].cp1*material.temperature);}
        const Real Gg=p.gasConductivity>0?area/(surface.gasDistance[f]/p.gasConductivity+p.material.gasContactResistance):0;
        const Real Gs=sc>=0&&ks>0?area/(surface.solidDistance[f]/ks+p.material.solidContactResistance):0;
        Real rate=0;
        if(film.nFaces&&film.q[f].mass>0){Cf=film.q[f].mass*(p.liquid.cp0+p.liquid.cp1*film.aux[f].temperature);const Real Gf=p.liquid.conductivity>0?2*area*p.liquid.conductivity/film.aux[f].thickness:0;
            const Real top=Gg>0&&Gf>0?1/(1/Gg+1/Gf):0,bottom=Gs>0&&Gf>0?1/(1/Gs+1/Gf):0;
            if(Cg>0&&Cf>0)rate+=top*(1/Cg+1/Cf);if(Cs>0&&Cf>0)rate+=bottom*(1/Cs+1/Cf);
        }else if(Cg>0&&Cs>0&&Gg>0&&Gs>0)rate+=(1/Cg+1/Cs)/(1/Gg+1/Gs);
        if(p.enableRadiation){const Real T=surface.interfaceGasState[f].temperature;const Real receiver=Cf>0?Cf:Cs;if(receiver>0)rate+=4*p.emissivity*5.670374419e-8*area*T*T*T/receiver;}
        if(rate>0)limit=minValue(limit,p.cfl/rate);
    }
    for(int f=0;f<film.nFaces;++f){Real sum=0,conductance=0;for(int e=0;e<surface.nEdges;++e){if(surface.edgeOwner[e]!=f&&surface.edgeNeighbour[e]!=f)continue;const int o=surface.edgeOwner[e],n=surface.edgeNeighbour[e];const Vec3 velocity=n>=0?(film.aux[o].meanVelocity+film.aux[n].meanVelocity)*.5:film.aux[o].meanVelocity;
            sum+=absValue(surface.edgeLength[e]*dot(velocity,surface.edgeConormal[e])-surface.sweptEdgeArea[e]/interval);
            if(n>=0){const Real distance=mag(surface.edgeOwnerOffset[e]-surface.edgeNeighbourOffset[e]);if(distance>0)conductance+=p.liquid.conductivity*surface.edgeLength[e]*.5*(film.aux[o].thickness+film.aux[n].thickness)/distance;}}
        if(sum>0)limit=minValue(limit,p.cfl*surface.area[f]/sum);
        const Real capacity=film.q[f].mass*(p.liquid.cp0+p.liquid.cp1*film.aux[f].temperature);
        if(p.enableRadiation&&(!surface.gasFace||surface.gasFace[f]<0)){const Real T=maxValue(film.aux[f].temperature,p.ambientTemperature);conductance+=4*p.emissivity*5.670374419e-8*surface.area[f]*T*T*T;}
        if(conductance>0&&capacity>0)limit=minValue(limit,.45*capacity/conductance);}
    for(int col=0;col<normal.nFaces;++col)for(int i=normal.offsets[col];i<normal.offsets[col+1];++i){const Real k=maxValue(p.liquid.conductivity,p.condensed[p.material.phaseCondensed].conductivity);const auto solidThermo=p.condensed[p.material.phaseCondensed];const Real cp=minValue(minValue(p.liquid.cp0+p.liquid.cp1*p.liquid.Tmin,p.liquid.cp0+p.liquid.cp1*p.liquid.Tmax),minValue(solidThermo.cp0+solidThermo.cp1*solidThermo.Tmin,solidThermo.cp0+solidThermo.cp1*solidThermo.Tmax));const Real conductance=4*k*normal.area[col]/normal.thickness[i];if(conductance>0)limit=minValue(limit,.45*normal.mass[i]*cp/conductance);}
    *result=limit;
}
bool launch(Backend& b,int code,const char* operation,std::string& error){if(!b.fault.check(static_cast<cudaError_t>(code),operation)){error=b.fault.message;return false;}return true;}
bool checked(Backend& b,std::string& error){if(!b.stream.sync()||!b.status.downloadOne(b.lastStatus,b.stream.get())){error=b.fault.message;return false;}
    if(b.lastStatus.code){error="device physics error "+std::to_string(b.lastStatus.code)+" at index "+std::to_string(b.lastStatus.index)+" value "+std::to_string(b.lastStatus.value);return false;}return true;}
bool freshStatus(Backend& b){b.lastStatus=DeviceStatus{};return b.status.uploadOne(b.lastStatus,b.fault);}
std::string escapeJson(const std::string& text){std::ostringstream out;for(unsigned char c:text){switch(c){case '"':out<<"\\\"";break;case '\\':out<<"\\\\";break;case '\n':out<<"\\n";break;case '\r':out<<"\\r";break;case '\t':out<<"\\t";break;default:if(c<32){out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;}else out<<char(c);}}return out.str();}
Vec3 prescribedPoint(Vec3 X,const MeshMotionConfig& motion,Real time){const Real t=::sin(motion.angularFrequency*(time-motion.timeOrigin));return {X.x+motion.amplitude.x*::sin(motion.spatialWaveNumber.x*(X.x-motion.spatialOrigin.x))*t,X.y+motion.amplitude.y*::sin(motion.spatialWaveNumber.y*(X.y-motion.spatialOrigin.y))*t,X.z+motion.amplitude.z*::sin(motion.spatialWaveNumber.z*(X.z-motion.spatialOrigin.z))*t};}
bool geometryFor(Backend& b,const HostState& base,const HostState& evaluation,const std::vector<FilmAux>& estimate,
    Real dt,HostState& endpoint,HostStageGeometry& gasStage,HostStageGeometry& solidStage,std::string& error){
    const auto& p=b.model.physics;endpoint=base;std::vector<Vec3> gasPoints=base.gasMesh.points,solidPoints=base.solidMesh.points;
    if(p.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){for(std::size_t i=0;i<gasPoints.size();++i)gasPoints[i]=prescribedPoint(base.gasMesh.referencePoints[i],p.meshMotion,base.time+dt);}
    if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){if(!moveCoupledMeshes(base,estimate,dt,endpoint.gasMesh,endpoint.solidMesh,endpoint.surface,error))return false;gasPoints=endpoint.gasMesh.points;solidPoints=endpoint.solidMesh.points;}
    std::vector<Real> gasSweep,solidSweep;std::vector<Vec3> gasAreas,solidAreas;
    if(!makeStageGeometry(base.gasMesh,gasPoints,dt,endpoint.gasMesh,gasSweep,error)||!makeStageAreaVectors(base.gasMesh,gasPoints,gasAreas,error)
        ||!makeStageGeometry(base.solidMesh,solidPoints,dt,endpoint.solidMesh,solidSweep,error)||!makeStageAreaVectors(base.solidMesh,solidPoints,solidAreas,error))return false;
    endpoint.gasMesh.referencePoints=base.gasMesh.referencePoints;endpoint.solidMesh.referencePoints=base.solidMesh.referencePoints;
    if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession){endpoint.surface.oldArea=base.surface.area;endpoint.surface.sweptEdgeArea.assign(base.surface.edgeOwner.size(),0);endpoint.surface.meshVelocity.assign(base.surface.area.size(),Vec3{});}
    if(!base.normalEnthalpy.empty())for(std::size_t c=0;c<endpoint.surface.normalPressure.size();++c)endpoint.surface.normalPressure[c]=base.surface.normalPressure[c]+dt*base.surface.normalPressureRate[c];
    auto trace=[&](const HostMesh& old,const HostMesh& eval,const HostMesh& end,const std::vector<Real>& sweeps,const std::vector<Vec3>& areas,HostStageGeometry& out){out.interval=dt;out.geometryVersion=end.geometryVersion;out.topologyHash=end.topologyHash;out.oldVolume=old.volumes;out.newVolume=end.volumes;out.evaluationVolume=eval.volumes;out.sweptVolume=sweeps;out.areaVector=areas;out.cellCentre=eval.cellCentres;out.faceCentre=eval.faceCentres;out.oldPoints=old.points;out.newPoints=end.points;};
    trace(base.gasMesh,evaluation.gasMesh,endpoint.gasMesh,gasSweep,gasAreas,gasStage);trace(base.solidMesh,evaluation.solidMesh,endpoint.solidMesh,solidSweep,solidAreas,solidStage);
    if(!b.gasGeometry.upload(base.gasMesh,evaluation.gasMesh,endpoint.gasMesh,gasSweep,gasAreas,b.fault)||!b.solidGeometry.upload(base.solidMesh,evaluation.solidMesh,endpoint.solidMesh,solidSweep,solidAreas,b.fault)){error=b.fault.message;return false;}
    SurfaceMesh evalSurface=evaluation.surface;evalSurface.oldArea=base.surface.area;evalSurface.sweptEdgeArea=endpoint.surface.sweptEdgeArea;
    if(!b.surface.upload(evalSurface,b.fault)||!b.endpointSurface.upload(endpoint.surface,b.fault)){error=b.fault.message;return false;}
    return true;
}
bool scatter(Backend& b,SurfaceBuffers& surface,std::string& error){return launch(b,launchScatterInterface(surface,b.gasGeometry,b.status.data(),b.stream.get()),"scatter interface trace",error)&&checked(b,error);}
bool recoverEvaluation(Backend& b,StateBuffers& state,Real dt,int stage,std::string& error){
    const auto& p=b.model.physics;auto stream=b.stream.get();auto status=b.status.data();auto gas=state.gasView(*b.committed,status,b.gasGeometry.wallDistance.data());auto solid=state.solidView(*b.committed,status);auto film=state.filmView(*b.committed,status);auto gg=b.gasGeometry.view(),sg=b.solidGeometry.view();auto surface=b.surface.view(b.solidGeometry);
    if(!launch(b,launchInitializeStorage(*b.committed,state,b.gasGeometry,b.surface,p,dt,b.accepted.acceptedSteps+1,stage,status,stream),"initialize accepted film storage",error))return false;
    if(solid.nCells&&!launch(b,launchMaterialRhs(solid,sg,PacketView{},p,dt,status,stream),"evaluate material RHS",error))return false;
    if(film.nFaces&&!launch(b,launchFilmRhs(film,surface,PacketView{},p,dt,status,stream),"seed film transport",error))return false;
    if(b.interfaceCount){auto provisionalGas=gas;provisionalGas.gradient=nullptr;
        if(!launch(b,launchInterfaceState(provisionalGas,solid,film,gg,surface,p,InterfaceTraceMode::ProvisionalModel,PacketView{},0,status,stream),"bootstrap current-model interface trace",error)||!checked(b,error)||!scatter(b,b.surface,error))return false;}
    if(gas.nCells&&!launch(b,launchGasPrepare(gas,gg,p,dt,stream),"prepare gas evaluation",error))return false;
    return checked(b,error);
}
bool closeValue(Real a,Real c,Real absolute,Real relative){return finite(a)&&finite(c)&&absValue(a-c)<=absolute+relative*maxValue(absValue(a),absValue(c));}
bool converge(const std::vector<ExchangePacket>& oldPackets,const std::vector<ExchangePacket>& packets,
    const std::vector<FilmAux>& oldAux,const std::vector<FilmAux>& aux,const PhysicsConfig& p,Real dt,Real& residual){
    if(oldPackets.size()!=packets.size()||oldAux.size()!=aux.size())return false;residual=0;bool ok=true;
    for(std::size_t i=0;i<packets.size();++i){const auto& a=oldPackets[i];const auto& q=packets[i];if(a.kind!=q.kind)ok=false;
        ok=closeValue(a.mass,q.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass)&&ok;
        ok=closeValue(a.energy,q.energy,p.tolerances.absoluteEnergy,p.tolerances.relativeEnergy)&&ok;
        for(int s=0;s<Ns;++s)ok=closeValue(a.species[s],q.species[s],p.tolerances.absoluteMass,p.tolerances.relativeMass)&&ok;
        residual=maxValue(residual,absValue(a.energy-q.energy)/(p.tolerances.absoluteEnergy+p.tolerances.relativeEnergy*maxValue(absValue(a.energy),absValue(q.energy))));}
    for(std::size_t f=0;f<aux.size();++f){if(aux[f].thickness>0||oldAux[f].thickness>0)ok=closeValue(aux[f].temperature,oldAux[f].temperature,p.tolerances.absoluteTemperature,p.tolerances.relativeTemperature)&&ok;
        const Real area=maxValue(aux[f].area,oldAux[f].area);
        ok=closeValue(aux[f].solidNormalVelocity*area*dt,oldAux[f].solidNormalVelocity*area*dt,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)&&ok;
        ok=closeValue(aux[f].normalVelocity*area*dt,oldAux[f].normalVelocity*area*dt,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)&&ok;}
    return ok;
}
bool interfaceIteration(Backend& b,StateBuffers& state,Real dt,int stage,std::string& error){
    if(!b.interfaceCount)return true;const auto& p=b.model.physics;auto stream=b.stream.get();auto status=b.status.data();
    auto gas=state.gasView(*b.committed,status,b.gasGeometry.wallDistance.data());auto solid=state.solidView(*b.committed,status);auto film=state.filmView(*b.committed,status);auto surface=b.surface.view(b.solidGeometry);auto gg=b.gasGeometry.view();
    std::vector<ExchangePacket> previous;std::vector<FilmAux> prior;Real residual=0;
    for(int iteration=0;iteration<p.tolerances.maxCouplingIterations;++iteration){
        ++b.stageIterations;
        if(!launch(b,launchInterfacePackets(gas,solid,film,gg,surface,state.packetView(),p,dt,b.accepted.acceptedSteps+1,stage,status,stream),"construct interface packets",error)||!checked(b,error)||!scatter(b,b.surface,error))return false;
        if(gas.nCells&&!launch(b,launchGasPrepare(gas,gg,p,dt,stream),"refresh interface gas gradients",error))return false;
        if(film.nFaces&&!launch(b,launchFilmRhs(film,surface,state.packetView(),p,dt,status,stream),"refresh packet-owned film work",error))return false;
        if(!checked(b,error))return false;
        std::vector<ExchangePacket> packets;std::vector<FilmAux> aux;if(!state.packets.download(packets,stream)||!state.filmAux.download(aux,stream)){error=b.fault.message;return false;}
        if(iteration&&converge(previous,packets,prior,aux,p,dt,residual)){b.finalCouplingResidual=residual;return true;}
        previous=std::move(packets);prior=std::move(aux);
    }
    error="interface packet/temperature/speed coupling did not converge";return false;
}
bool endpointChecks(Backend& b,StateBuffers& state,Real dt,std::string& error){
    const auto& p=b.model.physics;auto status=b.status.data();auto stream=b.stream.get();auto gas=state.gasView(*b.committed,status,b.gasGeometry.endpointWallDistance.data());auto solid=state.solidView(*b.committed,status);auto film=state.filmView(*b.committed,status);auto gg=b.gasGeometry.view(true),sg=b.solidGeometry.view(true);auto surface=b.endpointSurface.view(b.solidGeometry);surface.solidAreaVector=b.solidGeometry.endpointAreaVector.data();
    if(gas.nCells&&!launch(b,launchGasValidate(gas,gg,p,stream),"validate endpoint gas",error))return false;
    if(solid.nCells&&!launch(b,launchMaterialValidate(solid,sg,p,status,stream),"validate endpoint material",error))return false;
    if(film.nFaces&&!launch(b,launchFilmValidate(film,surface,p,status,stream),"validate endpoint film",error))return false;
    if(state.normalH.size()&&!launch(b,launchNormalThermalValidate(b.endpointSurface.normalView(state),p,status,stream),"validate endpoint normal enthalpy",error))return false;
    if(!checked(b,error))return false;
    if(b.interfaceCount){
        std::vector<GasPrimitive> previous;bool converged=false;
        for(int iteration=0;iteration<p.tolerances.maxCouplingIterations;++iteration){
            ++b.stageIterations;
            if(!launch(b,launchGasMeanPrepare(gas,gg,p,dt,stream),"recover endpoint mean-flow gradients without pending SST algebra",error)||!checked(b,error)
                ||!launch(b,launchInterfaceState(gas,solid,film,gg,surface,p,InterfaceTraceMode::DiscreteAcceptedFlux,state.packetView(),dt,status,stream),"recover discrete-flux endpoint trace",error)||!checked(b,error)||!scatter(b,b.endpointSurface,error))return false;
            std::vector<GasPrimitive> current;if(!b.endpointSurface.interfaceGasState.download(current,stream)){error=b.fault.message;return false;}
            converged=iteration>0;for(std::size_t f=0;converged&&f<current.size();++f){const int face=b.accepted.surface.gasFace[f],cell=b.accepted.gasMesh.owner[face];const Real mass=b.accepted.gas[cell].mass;
                converged=closeValue(previous[f].temperature,current[f].temperature,p.tolerances.absoluteTemperature,p.tolerances.relativeTemperature);
                for(int s=0;s<Ns;++s)converged=converged&&closeValue(previous[f].Y[s]*mass,current[f].Y[s]*mass,p.tolerances.absoluteMass,p.tolerances.relativeMass);}
            if(converged)break;previous=std::move(current);
        }
        if(!converged){error="endpoint discrete-flux wall trace did not converge";return false;}
    }
    if(p.enableSst){if(!launch(b,launchWallTrace(gas,b.gasGeometry.view(true),b.gasGeometry.wallTrace.data(),p,status,stream),"recover endpoint wall thermodynamics",error)
        ||!launch(b,launchSstConstrainWalls(gas,b.gasGeometry.view(true,true),p,status,stream),"constrain endpoint SST walls",error)
        ||!launch(b,launchSstValidate(gas,p,status,stream),"validate projected endpoint SST",error))return false;}
    return checked(b,error);
}
bool stepStage(Backend& b,const HostState& evaluationHost,const StateBuffers& evaluation,StateBuffers& state,
    Real dt,int stage,HostState& endpointHost,HostStageGeometry& gasStage,HostStageGeometry& solidStage,StepReport& report,std::string& error){
    const auto& p=b.model.physics;auto stream=b.stream.get();b.stageIterations=0;b.finalCouplingResidual=0;std::vector<FilmAux> estimate=evaluationHost.filmAux;
    // Geometry starts from a true base-derived front estimate, not the predictor's
    // cumulative displacement reused as a full-step increment.
    for(std::size_t f=0;f<estimate.size();++f){estimate[f].solidFront=b.accepted.filmAux[f].solidFront+dt*estimate[f].solidNormalVelocity;estimate[f].gasFront=b.accepted.filmAux[f].gasFront+dt*estimate[f].normalVelocity;}
    const int iterations=b.interfaceCount||p.meshMotion.policy==MeshMotionPolicy::CoupledRecession?p.tolerances.maxCouplingIterations:1;
    for(int iteration=0;iteration<iterations;++iteration){
        if(!freshStatus(b)||!state.copyInventory(evaluation,b.fault,stream)||!state.resetScratch(stream)||!b.stream.sync()){error=b.fault.message;return false;}
        if(!geometryFor(b,b.accepted,evaluationHost,estimate,dt,endpointHost,gasStage,solidStage,error))return false;
        if(!recoverEvaluation(b,state,dt,stage,error)||!interfaceIteration(b,state,dt,stage,error))return false;
        auto gas=state.gasView(*b.committed,b.status.data(),b.gasGeometry.wallDistance.data());auto gg=b.gasGeometry.view();
        stableStepKernel<<<1,1,0,stream>>>(gas,gg,state.solidView(*b.committed,b.status.data()),b.solidGeometry.view(),state.filmView(*b.committed,b.status.data()),b.surface.view(b.solidGeometry),b.surface.normalView(state),state.physics.data(),dt,b.stableDt.data());
        if(!launch(b,static_cast<int>(cudaPeekAtLastError()),"evaluate explicit timestep bound",error)||!checked(b,error)||!b.stableDt.downloadOne(b.nextStableDt,stream)){if(error.empty())error=b.fault.message;return false;}
        if(!finite(b.nextStableDt)||b.nextStableDt<=0||dt>b.nextStableDt*(1+1e-12)){error="explicit CFL/thermal timestep bound exceeded";return false;}
        if(p.enableParticles&&!launch(b,launchParticleExchange(*b.committed,evaluation,state,b.gasGeometry,b.surface,p,dt,b.accepted.acceptedSteps+1,stage,b.interfaceCount,b.status.data(),stream),"particle midpoint exchange/tracking",error))return false;
        if(gas.nCells){if(!launch(b,launchGasRhs(gas,gg,state.packetView(),p,dt,stream),"evaluate gas/SST RHS",error)
            ||!launch(b,launchGasUpdate(b.committed->gas.data(),gas,gg,dt,b.status.data(),stream),"update gas from accepted base",error))return false;
            if(p.enableSst&&!launch(b,launchSstUpdate(b.committed->sst.data(),gas,gg,p,dt,b.status.data(),stream),"update SST from accepted base",error))return false;}
        if(state.normalH.size()&&!launch(b,launchResolvedNormalThermalRhs(b.surface.normalView(state),p,dt,b.status.data(),stream),"evaluate normal thermal RHS",error))return false;
        if(!launch(b,launchParticipantUpdate(*b.committed,state,b.gasGeometry,b.surface,b.endpointSurface,p,dt,b.status.data(),stream),"assemble material/film/normal participants",error)||!endpointChecks(b,state,dt,error))return false;
        std::vector<FilmAux> actual;if(!state.filmAux.download(actual,stream)){error=b.fault.message;return false;}
        bool settled=true;
        if(iterations>1){settled=iteration>0;for(std::size_t f=0;f<actual.size();++f){const Real area=maxValue(actual[f].area,estimate[f].area);
            settled=settled&&closeValue(actual[f].solidFront*area,estimate[f].solidFront*area,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)
                &&closeValue(actual[f].thickness*area,estimate[f].thickness*area,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry);}}
        if(!settled){estimate=std::move(actual);continue;}
        if(b.interfaceCount){std::vector<Real> solidResidual,filmResidual;if(!b.surface.solidRecessionResidual.download(solidResidual,stream)||!b.surface.filmVolumeResidual.download(filmResidual,stream)){error=b.fault.message;return false;}
            for(std::size_t f=0;f<solidResidual.size();++f){const Real scale=actual[f].area*(absValue(actual[f].solidNormalVelocity)+absValue(actual[f].normalVelocity));const Real tolerance=p.tolerances.absoluteGeometry/dt+p.tolerances.relativeGeometry*scale;
                if(!finite(solidResidual[f])||!finite(filmResidual[f])||absValue(solidResidual[f])>tolerance||absValue(filmResidual[f])>tolerance){error="interface independent swept-volume compatibility failed";return false;}}}
        if(!launch(b,launchStageAudit(*b.committed,evaluation,state,b.gasGeometry,b.solidGeometry,b.surface,b.endpointSurface,b.baseSurface.normalPressure.data(),p,dt,b.accepted.budget,b.diagnostics.data(),b.status.data(),stream),"audit assembled transaction",error)||!checked(b,error))return false;
        StageDiagnostics diagnostics;Budget budget;if(!b.diagnostics.downloadOne(diagnostics,stream)||!state.trialBudget.downloadOne(budget,stream)){error=b.fault.message;return false;}
        const Real energyScale=maxValue(absValue(diagnostics.energyBefore),absValue(diagnostics.energyAfter));
        if(absValue(diagnostics.reducedResidual)>p.tolerances.absoluteEnergy+p.tolerances.relativeEnergy*energyScale){error="film reduced-energy balance failed";return false;}
        const Real numericalChange=budget.numericalEnergyResidual-b.accepted.budget.numericalEnergyResidual;
        if(absValue(numericalChange)>p.tolerances.absoluteEnergy+p.tolerances.relativeEnergy*energyScale){error="assembled total-energy numerical residual exceeds tolerance";return false;}
        if(!state.download(endpointHost,stream)||!b.gasGeometry.gasBoundary.download(endpointHost.gasMesh.boundaryPrimitive,stream)){error=b.fault.message;return false;}
        if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){
            // Legacy physical targets retain their declared old-area/front
            // convention. Predictor stages read the same accepted carry; only
            // this candidate gets the remainder measured on its actual path.
            std::vector<Real> physical(b.accepted.surface.solidFace.size());
            for(std::size_t f=0;f<physical.size();++f){const int face=b.accepted.surface.solidFace[f];
                physical[f]=mag(b.accepted.solidMesh.areaVectors[face])*(endpointHost.filmAux[f].solidFront-b.accepted.filmAux[f].solidFront);}
            if(!updateMaterialSweepRemainder(b.accepted,physical,solidStage.sweptVolume,endpointHost.solidMesh,endpointHost.solidSweepRemainder,error))return false;
        }
        endpointHost.budget=budget;report.iterations+=b.stageIterations+iteration+1;report.couplingResidual=b.finalCouplingResidual;report.minimumMass=diagnostics.minimumMass;report.minimumTemperature=diagnostics.minimumTemperature;report.filmReducedResidual=diagnostics.reducedResidual;report.filmKineticDefect=diagnostics.kineticDefect;
        return true;
    }
    error="coupled endpoint geometry did not converge";return false;
}
} // namespace
Backend* createBackend(const ModelConfig& model,const HostState& input,std::string& error){
    try{
        if(model.sourceFingerprint!=CHMT_SOURCE_FINGERPRINT||model.baseCommit!=CHMT_UPSTREAM_BASE){error="model source/upstream identity differs from this compiled CHMT binary";return nullptr;}
        HostState initial=input;if(!validateRuntimeState(model,initial,error))return nullptr;
        int devices=0;auto code=cudaGetDeviceCount(&devices);if(code!=cudaSuccess||devices<1){error=std::string("CUDA device unavailable: ")+cudaGetErrorString(code);return nullptr;}
        std::unique_ptr<Backend> b(new Backend);b->model=model;b->accepted=initial;
        const bool coupled=!initial.surface.gasFace.empty()&&initial.surface.gasFace[0]>=0;
        b->interfaceCount=coupled?InterfacePacketsPerFace*initial.surface.area.size():0;
        if(initial.particles.size()>(std::size_t(INT_MAX)-b->interfaceCount)/2){error="particle packet count exceeds int ABI";return nullptr;}
        b->totalPackets=b->interfaceCount+2*initial.particles.size();
        if(!b->stream.create(b->fault)||!b->status.resize(1,b->fault)||!b->diagnostics.resize(1,b->fault)||!b->stableDt.resize(1,b->fault)
            ||!b->committed->initialize(initial,model.physics,b->fault,b->totalPackets)
            ||!b->midpoint->initialize(initial,model.physics,b->fault,b->totalPackets)
            ||!b->trial->initialize(initial,model.physics,b->fault,b->totalPackets)
            ||!b->baseSurface.upload(initial.surface,b->fault)){error=b->fault.message;return nullptr;}
        std::vector<Real> gasSweeps(initial.gasMesh.owner.size(),0),solidSweeps(initial.solidMesh.owner.size(),0);
        if(!b->gasGeometry.upload(initial.gasMesh,initial.gasMesh,initial.gasMesh,gasSweeps,initial.gasMesh.areaVectors,b->fault)
            ||!b->solidGeometry.upload(initial.solidMesh,initial.solidMesh,initial.solidMesh,solidSweeps,initial.solidMesh.areaVectors,b->fault)
            ||!b->surface.upload(initial.surface,b->fault)||!b->endpointSurface.upload(initial.surface,b->fault)
            ||!freshStatus(*b)||!b->committed->resetScratch(b->stream.get())){error=b->fault.message;return nullptr;}
        // Normal initial H/fraction may encode exact cut-cell averages. Validate
        // those supplied inventories on host, but do not equilibrate/overwrite f
        // before the first real time stage and initial observation export.
        if(!initial.normalEnthalpy.empty()||initial.acceptedSteps>0){
            // A restored accepted snapshot retains its exact endpoint trace,
            // inventories, auxiliary history and phase fraction. Do not replace
            // its discrete accepted wall trace with fresh instantaneous kinetics.
            if(model.physics.enableSst&&!launch(*b,launchSstValidate(b->committed->gasView(*b->committed,b->status.data(),b->gasGeometry.wallDistance.data()),model.physics,b->status.data(),b->stream.get()),"validate restored SST inventory",error))return nullptr;
            if(!checked(*b,error))return nullptr;
        }else{
            if(!recoverEvaluation(*b,*b->committed,model.physics.minDt,0,error))return nullptr;
            auto gas=b->committed->gasView(*b->committed,b->status.data(),b->gasGeometry.wallDistance.data());
            if(model.physics.enableSst){if(!launch(*b,launchWallTrace(gas,b->gasGeometry.view(true),b->gasGeometry.wallTrace.data(),model.physics,b->status.data(),b->stream.get()),"initialize actual wall trace",error)
                ||!launch(*b,launchSstConstrainWalls(gas,b->gasGeometry.view(true,true),model.physics,b->status.data(),b->stream.get()),"initialize SST wall constraint",error)
                ||!launch(*b,launchSstValidate(gas,model.physics,b->status.data(),b->stream.get()),"validate initial SST wall state",error))return nullptr;}
            if(!checked(*b,error))return nullptr;
            if(!b->committed->filmAux.download(b->accepted.filmAux,b->stream.get())||!b->committed->sst.download(b->accepted.sst,b->stream.get())||!b->committed->solid.download(b->accepted.solid,b->stream.get())||!b->gasGeometry.gasBoundary.download(b->accepted.gasMesh.boundaryPrimitive,b->stream.get())){error=b->fault.message;return nullptr;}
        }
        b->ready=true;error.clear();return b.release();
    }catch(const std::exception& e){error=std::string("create CUDA backend: ")+e.what();return nullptr;}
}
static bool advanceImpl(Backend& b,double requestedDt,StepReport& report,std::string& error){
    if(b.pending){error="legacy advance is unavailable during a pending gas window";return false;}
    if(!b.ready||!b.fault.message.empty()){error=b.fault.message.empty()?"backend is not ready":b.fault.message;return false;}
    const auto& p=b.model.physics;if(!finite(requestedDt)||requestedDt<=0){error="requested timestep must be finite and positive";return false;}
    if(!b.midpoint->packets.resize(b.totalPackets,b.fault)||!b.trial->packets.resize(b.totalPackets,b.fault)){error=b.fault.message;return false;}
    Real dt=minValue(requestedDt,p.maxDt);std::string lastError;
    for(int retry=0;retry<=p.tolerances.maxRetries;++retry){
        if(dt<p.minDt){lastError="minimum timestep reached: "+lastError;break;}
        if(!b.baseSurface.upload(b.accepted.surface,b.fault)){error=b.fault.message;return false;}
        HostState midpointHost,endpointHost;HostStageGeometry gasHalf,solidHalf,gasFull,solidFull;StepReport trialReport;
        bool ok=stepStage(b,b.accepted,*b.committed,*b.midpoint,.5*dt,0,midpointHost,gasHalf,solidHalf,trialReport,lastError);
        if(ok)ok=stepStage(b,midpointHost,*b.midpoint,*b.trial,dt,1,endpointHost,gasFull,solidFull,trialReport,lastError);
        if(ok){endpointHost.gasStages[0]=std::move(gasHalf);endpointHost.gasStages[1]=std::move(gasFull);endpointHost.solidStages[0]=std::move(solidHalf);endpointHost.solidStages[1]=std::move(solidFull);
            endpointHost.rejectedSteps=b.accepted.rejectedSteps+retry;endpointHost.nextDt=minValue(minValue(p.maxDt,b.nextStableDt),dt*(retry?1:1.2));
            const Real fullGcl=maxValue(maximumGclResidual(b.accepted.gasMesh,endpointHost.gasMesh,endpointHost.gasStages[1].sweptVolume),maximumGclResidual(b.accepted.solidMesh,endpointHost.solidMesh,endpointHost.solidStages[1].sweptVolume));
            const Real halfGcl=maxValue(maximumGclResidual(b.accepted.gasMesh,midpointHost.gasMesh,endpointHost.gasStages[0].sweptVolume),maximumGclResidual(b.accepted.solidMesh,midpointHost.solidMesh,endpointHost.solidStages[0].sweptVolume));
            const Real gcl=maxValue(fullGcl,halfGcl);
            if(!finite(gcl)){lastError="nonfinite accepted GCL residual";ok=false;}else{endpointHost.budget.gclResidual+=gcl;trialReport.gclResidual=gcl;}
            if(ok&&!publishAcceptedCandidate(b.accepted,std::move(endpointHost),1,dt,b.stream.sync(),lastError))ok=false;
            if(ok){b.committed.swap(b.trial);trialReport.acceptedDt=dt;trialReport.nextDt=b.accepted.nextDt;trialReport.rejectedTrials=retry;trialReport.commitSequence=b.accepted.commitSequence;trialReport.budget=b.accepted.budget;
                trialReport.minimumVolume=std::numeric_limits<Real>::max();for(Real v:b.accepted.gasMesh.volumes)trialReport.minimumVolume=minValue(trialReport.minimumVolume,v);for(Real v:b.accepted.solidMesh.volumes)trialReport.minimumVolume=minValue(trialReport.minimumVolume,v);for(Real v:b.accepted.surface.normalLayerVolume)trialReport.minimumVolume=minValue(trialReport.minimumVolume,v);
                report=trialReport;error.clear();return true;}
        }
        if(!b.fault.message.empty()){error=b.fault.message;return false;}
        if(b.lastStatus.code==static_cast<int>(ErrorCode::Unsupported)){error=lastError+" (unsupported capability; timestep retry cannot fix it)";return false;}
        if(b.lastStatus.location==ErrorLocation::Particle&&b.lastStatus.code==static_cast<int>(ErrorCode::Inventory)&&b.lastStatus.value<0)
            dt=particleEventRetryDt(dt,b.lastStatus.value);
        else dt*=.5;
    }
    error="advance left last accepted state unchanged: "+lastError;return false;
}
bool advance(Backend& b,double requestedDt,StepReport& report,std::string& error){
    try{return advanceImpl(b,requestedDt,report,error);}
    catch(const std::exception& exception){error=std::string("advance preserved last accepted state: ")+exception.what();return false;}
}
bool downloadState(const Backend& b,HostState& output,std::string& error){
    if(!b.ready){error="backend not ready";return false;}
    // Each accepted HostState was downloaded only after the complete device
    // transaction synchronized. A later CUDA context failure must not prevent
    // writing this last accepted snapshot; no failed trial data are exposed.
    try{HostState snapshot=b.accepted;output=std::move(snapshot);error.clear();return true;}
    catch(const std::exception& exception){error=std::string("download accepted snapshot: ")+exception.what();return false;}
}
static bool restoreStateImpl(Backend& b,const HostState& input,std::string& error){
    if(b.pending){error="rollback the pending gas window before restoring state";return false;}
    // Replacement allocation and validation happen independently. Failed restore
    // cannot overwrite the accepted owner, its device buffers, or its reports.
    if(!b.stream.sync()){error=b.fault.message;return false;}
    std::unique_ptr<Backend> replacement(createBackend(b.model,input,error));if(!replacement)return false;
    // Reuse the already allocated and synchronized replacement inventories.
    // Their preparation errors belong only to replacement->fault. Rebinding is
    // a no-fail pointer operation at commit, so no second allocation can poison
    // the old backend or multiply peak memory beyond old + replacement owners.
    if(!publishResourceReplacement(b.committed,b.midpoint,b.trial,replacement->committed,
        replacement->midpoint,replacement->trial,b.fault,replacement->fault,error))return false;
    b.accepted=std::move(replacement->accepted);b.interfaceCount=replacement->interfaceCount;b.totalPackets=replacement->totalPackets;
    error.clear();return true;
}
bool restoreState(Backend& b,const HostState& input,std::string& error){
    try{return restoreStateImpl(b,input,error);}
    catch(const std::exception& exception){error=std::string("restore rejected: ")+exception.what();return false;}
}
void destroyBackend(Backend* backend)noexcept{delete backend;}
std::string backendBuildInfo(){
    std::ostringstream out;out<<"{\"schema_version\":1,\"artifact_kind\":\"CHMT_BACKEND_BUILD\",\"solver_commit\":\""<<escapeJson(CHMT_SOLVER_COMMIT)<<"\",\"build_id\":\""<<escapeJson(CHMT_BUILD_ID)<<"\",\"source_fingerprint\":\""<<escapeJson(CHMT_SOURCE_FINGERPRINT)<<"\",\"upstream_base\":\""<<escapeJson(CHMT_UPSTREAM_BASE)<<"\",\"precision\":\"FP64\",\"compiler\":\""<<escapeJson(std::string(CHMT_CUDA_COMPILER)+"; "+CHMT_HOST_COMPILER)<<"\",\"cuda_arch\":\""<<escapeJson(CHMT_CUDA_ARCH)<<"\",\"openfoam_version\":\""<<escapeJson(CHMT_OPENFOAM_VERSION)<<"\",\"species\":[";
    for(int s=0;s<Ns;++s){if(s)out<<',';out<<'"'<<escapeJson(compiledSpeciesNames[s])<<'"';}out<<"],\"hardware\":";
    int device=0;cudaDeviceProp properties{};auto code=cudaGetDevice(&device);if(code==cudaSuccess)code=cudaGetDeviceProperties(&properties,device);
    if(code!=cudaSuccess)out<<"{\"available\":false,\"error\":\""<<escapeJson(cudaGetErrorString(code))<<"\"}";
    else out<<"{\"available\":true,\"name\":\""<<escapeJson(properties.name)<<"\",\"compute_capability\":\""<<properties.major<<'.'<<properties.minor<<"\",\"global_memory_bytes\":"<<properties.totalGlobalMem<<'}';
    out<<",\"manifest\":"<<CHMT_BUILD_MANIFEST_JSON<<'}';return out.str();
}
bool remapGas1D(const std::vector<Real>& oldEdges,const std::vector<GasQ>& oldQ,const std::vector<Real>& newEdges,std::vector<GasQ>& result,std::string& error){
    if(oldQ.empty()||oldQ.size()>INT_MAX||newEdges.size()<2||newEdges.size()>std::size_t(INT_MAX)||oldEdges.size()!=oldQ.size()+1){error="invalid remap array counts";return false;}
    for(std::size_t i=0;i<oldEdges.size();++i)if(!finite(oldEdges[i])||(i&&oldEdges[i]<=oldEdges[i-1])){error="old remap edges are not finite/increasing";return false;}
    for(std::size_t i=0;i<newEdges.size();++i)if(!finite(newEdges[i])||(i&&newEdges[i]<=newEdges[i-1])){error="new remap edges are not finite/increasing";return false;}
    const Real tolerance=64*std::numeric_limits<Real>::epsilon()*(oldEdges.back()-oldEdges.front());if(absValue(oldEdges.front()-newEdges.front())>tolerance||absValue(oldEdges.back()-newEdges.back())>tolerance){error="remap domains differ";return false;}
    CudaFault fault;DeviceStream stream;Buffer<Real> oldGeometry,newGeometry;Buffer<GasQ> oldState,newState;Buffer<DeviceStatus> status;
    if(!stream.create(fault)||!oldGeometry.upload(oldEdges,fault)||!newGeometry.upload(newEdges,fault)||!oldState.upload(oldQ,fault)||!newState.resize(newEdges.size()-1,fault)||!status.uploadOne(DeviceStatus{},fault)){error=fault.message;return false;}
    if(!fault.check(static_cast<cudaError_t>(launchConservativeRemap1D(oldGeometry.data(),oldQ.size(),oldState.data(),newGeometry.data(),newEdges.size()-1,newState.data(),status.data(),stream.get())),"launch conservative CUDA overlap")||!stream.sync()){error=fault.message;return false;}
    DeviceStatus checkedStatus;if(!status.downloadOne(checkedStatus,stream.get())){error=fault.message;return false;}if(checkedStatus.code){error="CUDA remap rejected donor/geometry at cell "+std::to_string(checkedStatus.index);return false;}
    std::vector<GasQ> candidate;if(!newState.download(candidate,stream.get())){error=fault.message;return false;}
    if(!oldGeometry.release()||!newGeometry.release()||!oldState.release()||!newState.release()||!status.release()||!stream.close()){error=fault.message;return false;}
    result=std::move(candidate);error.clear();return true;
}
#include "gpu/GasWindowImplementation.cuh"
} // namespace chmt
