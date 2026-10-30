#pragma once
#include "gpu/SharedGasDeviceStorage.cuh"
#include "../../../common/thermal/ParticleGasBinding.H"
#include "../../../common/thermal/ParticleGeometryApi.H"
#include "../../../common/thermal/ParticleStateApi.H"
#include "../../../common/thermal/ParticleSurfaceState.H"
#include "../../../common/thermal/ParticleInitialization.H"
#include "../../../common/thermal/GpuBackendApi.H"
namespace chmt {
// Application state adapter only. Evolution and contact/thermal mathematics
// live in the common thermal owner, also used by CHT.
class ParticleCouplingStorage {
    void* owner_=nullptr;
    CudaFault fault_;
    Buffer<int> boundaryKind_,faceOffsets_,facePoints_;
    Buffer<ugkp::thermal::ParticleGeometryPoint> currentPoints_,nextPoints_;
    Buffer<double> restitution_,tangential_;
    Buffer<double> surfaceTemperature_,surfaceUx_,surfaceUy_,surfaceUz_,surfaceEffusivity_;
public:
    ParticleCouplingStorage()=default;
    ParticleCouplingStorage(const ParticleCouplingStorage&)=delete;
    ParticleCouplingStorage& operator=(const ParticleCouplingStorage&)=delete;
    ~ParticleCouplingStorage(){if(owner_)ugkwpGpuResidentStrictRelease(owner_);}
    void* owner()const{return owner_;}
    bool active()const{return owner_!=nullptr;}
    const double* gasFraction()const{return ugkp::thermal::particleGasVolumeFraction(owner_);}
    ugkp::thermal::ParticleWallExchangeView wallExchange()const{return ugkp::thermal::particleWallExchange(owner_);}
    ugkp::thermal::ParticleSurfaceState surface()const{return {surfaceTemperature_.data(),surfaceUx_.data(),surfaceUy_.data(),surfaceUz_.data(),surfaceEffusivity_.data()};}
    bool result(int code,std::string& error)const {
        if(code){error=ugkwpGpuResidentStrictLastError();return false;}return true;
    }
    bool create(const ModelConfig& model,const HostState& state,SharedGasDeviceStorage& gas,
        cudaStream_t stream,std::string& error) {
        if(!model.physics.enableParticles)return true;
        auto config=model.particlePhase;
        config.capacity=std::max(config.capacity,std::max(1,int(state.particles.size())));
        const int faces=state.gasMesh.owner.size();
        std::vector<unsigned char> contact(faces,0);
        for(int f:state.surface.gasFace)contact.at(f)=1;
        const auto& g=model.physics.gravity;const double gravity[3]={g.x,g.y,g.z};
        if(!gas.observeDiagnostics()){error=gas.error();return false;}
        if(!result(ugkp::thermal::createBoundParticlePhase(config,model.particleContact,
            ugkp::thermal::particleGasBinding(gas.hostView()),state.gasMesh.cellFaces.size(),
            contact.data(),gravity,&owner_),error))return false;
        if(!boundaryKind_.resize(faces,fault_)||!restitution_.resize(faces,fault_)||!tangential_.resize(faces,fault_)
            ||!surfaceTemperature_.resize(faces,fault_)||!surfaceUx_.resize(faces,fault_)
            ||!surfaceUy_.resize(faces,fault_)||!surfaceUz_.resize(faces,fault_)||!surfaceEffusivity_.resize(faces,fault_)){
            error=fault_.message;return false;}
        // Existing gas geometry policies provide wall/slip/periodic face kinds.
        std::vector<int> kinds(faces,0);
        for(int f=0;f<faces;++f){const auto kind=state.gasMesh.boundaryKind[f];
            kinds[f]=contact[f]?5:kind==BoundaryKind::Periodic?6:
                (kind==BoundaryKind::Slip||kind==BoundaryKind::Empty)?1:kind==BoundaryKind::NoSlip?2:0;
        }
        const std::vector<double> unit(faces,1.);
        if(!gas.write(boundaryKind_.data(),kinds)||!gas.write(restitution_.data(),unit)||!gas.write(tangential_.data(),unit)){
            error=gas.error();return false;}
        if(!result(ugkp::thermal::bindParticleSurface(owner_,faces,surface()),error))return false;
        return load(model,state,gas,stream,error);
    }
    bool refreshGeometry(SharedGasDeviceStorage& gas,cudaStream_t stream,std::string& error){
        if(!owner_)return true;
        if(!result(ugkp::thermal::bindParticleGas(owner_,ugkp::thermal::particleGasBinding(gas.hostView())),error))return false;
        gas.hostView().gasVolumeFraction=gasFraction();gas.hostView().particleWall=wallExchange();
        if(!gas.refreshView()){error=gas.error();return false;}
        return result(ugkp::thermal::refreshParticleTrackingGeometry(owner_,
            {gas.hostView().nFaces,boundaryKind_.data(),restitution_.data(),tangential_.data()},stream),error);
    }
    bool load(const ModelConfig& model,const HostState& state,SharedGasDeviceStorage& gas,
        cudaStream_t stream,std::string& error){
        if(!owner_)return true;
        if(!gas.observeDiagnostics()){error=gas.error();return false;}
        if(!refreshGeometry(gas,stream,error)||!gas.observeDiagnostics()){
            if(error.empty())error=gas.error();return false;}
        // Full state transfer only at a committed window/restart boundary.
        std::vector<ugkp::thermal::ParticleState<>> particles;particles.reserve(state.particles.size());
        for(const auto& value:state.particles){auto p=ugkp::thermal::adaptParticleState<ugkp::thermal::ParticleVector>(value);
            if(!ugkp::thermal::initializeMobileParticleHistory(p,model.particlePhase)){
                error="particle import requires a valid free-particle energy or complete attached thermal history";return false;}
            particles.push_back(p);
        }
        // The upload rebuilds moments and initializes epsGPrev. Restore the
        // accepted lagged phase fraction afterwards; do not recompute a restart.
        if(!result(ugkp::thermal::uploadParticleState(owner_,particles.size(),particles.data()),error))return false;
        if(!state.gasVoidFraction.empty()&&!result(ugkwpGpuResidentStrictUploadEpsGPrev(owner_,state.gasVoidFraction.data()),error))return false;
        return true;
    }
    bool mapGeometry(const HostMesh& previous,const HostMesh& next,SharedGasDeviceStorage& gas,
        cudaStream_t stream,std::string& error){
        if(!owner_)return true;
        if(previous.topologyHash!=next.topologyHash||previous.points.size()!=next.points.size()){
            error="attached particle mapping requires topology-preserving mesh motion";return false;}
        if(!faceOffsets_.size()){
            if(!faceOffsets_.resize(previous.faceOffsets.size(),fault_)||!facePoints_.resize(previous.facePoints.size(),fault_)
                ||!gas.write(faceOffsets_.data(),previous.faceOffsets)||!gas.write(facePoints_.data(),previous.facePoints)){
                error=fault_.message.empty()?gas.error():fault_.message;return false;}
        }
        auto uploadPoints=[&](const std::vector<Vec3>& points,auto& storage){
            std::vector<ugkp::thermal::ParticleGeometryPoint> values;values.reserve(points.size());
            for(const auto& p:points)values.push_back({p.x,p.y,p.z});
            return storage.resize(values.size(),fault_)&&gas.write(storage.data(),values);
        };
        if(!uploadPoints(previous.points,currentPoints_)||!uploadPoints(next.points,nextPoints_)){
            error=fault_.message.empty()?gas.error():fault_.message;return false;}
        ugkp::thermal::ParticleSurfaceGeometry old{int(previous.owner.size()),int(previous.points.size()),
            faceOffsets_.data(),facePoints_.data(),gas.hostView().faceOwner,currentPoints_.data()};
        auto endpoint=old;endpoint.points=nextPoints_.data();
        return result(ugkp::thermal::mapAttachedParticles(owner_,old,endpoint,nullptr,stream),error);
    }
    bool download(HostState& state,std::string& error){
        if(!owner_)return true;
        std::vector<ugkp::thermal::ParticleState<>> particles;
        if(!result(ugkp::thermal::downloadParticleState(owner_,particles),error))return false;
        state.particles.clear();state.particles.reserve(particles.size());
        for(const auto& p:particles)state.particles.push_back(ugkp::thermal::adaptParticleState<Vec3>(p));
        return true;
    }
};
}
