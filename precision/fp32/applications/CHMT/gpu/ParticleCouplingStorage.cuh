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
    const GpuReal* gasFraction()const{return ugkp::thermal::particleGasVolumeFraction(owner_);}
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
        const auto& inlet=model.particleInlet;
        if(!inlet.faces.empty()){
            std::vector<int> cells;std::vector<double> px,py,pz,u(inlet.faces.size(),0),
                temperature(inlet.faces.size(),inlet.temperature),theta(inlet.faces.size(),inlet.theta),
                diameter(inlet.faces.size(),config.diameter);
            for(int f:inlet.faces){
                const int cell=state.gasMesh.owner.at(f);cells.push_back(cell);
                const auto x=state.gasMesh.faceCentres.at(f)+(state.gasMesh.cellCentres.at(cell)-state.gasMesh.faceCentres.at(f))*1e-6;
                px.push_back(x.x);py.push_back(x.y);pz.push_back(x.z);
            }
            if(!result(ugkwpGpuResidentStrictUploadBoundarySources(owner_,inlet.faces.size(),cells.data(),inlet.faces.data(),
                px.data(),py.data(),pz.data(),u.data(),u.data(),u.data(),temperature.data(),theta.data(),diameter.data(),u.data()),error)
                ||!result(ugkwpGpuResidentStrictConfigureParticleInjectionVelocity(owner_,inlet.velocitySpecified,
                    inlet.velocity[0],inlet.velocity[1],inlet.velocity[2]),error)
                ||!result(ugkwpGpuResidentStrictConfigureScheduledInlet(owner_,inlet.faces.size(),inlet.faces.data(),inlet.temperature,
                    0,nullptr,nullptr,inlet.times.size(),inlet.times.data(),inlet.volumeFraction.data()),error))return false;
            // Deliberately retain this initial inlet geometry across mesh motion.
            std::vector<int> offsets{0},vertices;
            for(int f:inlet.faces){
                for(int j=state.gasMesh.faceOffsets[f];j<state.gasMesh.faceOffsets[f+1];++j)
                    vertices.push_back(state.gasMesh.facePoints[j]);
                offsets.push_back(int(vertices.size()));
            }
            std::vector<ugkp::thermal::ParticleGeometryPoint> points;
            for(const auto& p:state.gasMesh.points)points.push_back({p.x,p.y,p.z});
            if(!result(ugkwpGpuResidentStrictUploadBoundarySourceGeometry(owner_,
                int(vertices.size()),offsets.data(),vertices.data(),
                int(points.size()),points.data()),error))return false;
        }
        return load(model,state,gas,stream,error);
    }
    bool refreshGeometry(SharedGasDeviceStorage& gas,cudaStream_t stream,std::string& error){
        if(!owner_)return true;
        if(!result(ugkp::thermal::bindParticleGas(owner_,ugkp::thermal::particleGasBinding(gas.hostView())),error))return false;
        gas.hostView().gasVolumeFraction=gasFraction();gas.hostView().particleWall=wallExchange();
        if(!gas.refreshView()){error=gas.error();return false;}
        return refreshTrackingGeometry(gas,stream,error);
    }
    // Coordinate-only motion updates existing gas arrays in place. Their
    // borrowed addresses and physical configuration were bound at window load;
    // rebuilding tracking planes does not republish either owner's descriptor.
    bool refreshTrackingGeometry(const SharedGasDeviceStorage& gas,cudaStream_t stream,std::string& error){
        if(!owner_)return true;
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
        if(!model.particleInlet.faces.empty()){
            if(state.particleSourceFaces!=model.particleInlet.faces||state.particleSourceResidualMass.size()!=state.particleSourceFaces.size()){
                error="particle source residual layout mismatch";return false;
            }
            if(!result(ugkwpGpuResidentStrictUploadSourceResidualMass(owner_,state.particleSourceFaces.size(),
                state.particleSourceFaces.data(),state.particleSourceResidualMass.data()),error))return false;
        }
        return true;
    }
    bool mapGeometry(const ugkp::thermal::ParticleSurfaceGeometry& previous,
        const ugkp::thermal::ParticleSurfaceGeometry& next,cudaStream_t stream,std::string& error){
        return !owner_||result(ugkp::thermal::mapAttachedParticles(owner_,previous,next,nullptr,stream),error);
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
        int sources=0;
        if(!result(ugkwpGpuResidentStrictDownloadSourceResidualMass(owner_,&sources,0,nullptr,nullptr),error))return false;
        state.particleSourceFaces.resize(sources);state.particleSourceResidualMass.resize(sources);
        if(sources&&!result(ugkwpGpuResidentStrictDownloadSourceResidualMass(owner_,&sources,sources,
            state.particleSourceFaces.data(),state.particleSourceResidualMass.data()),error))return false;
        state.particles.clear();state.particles.reserve(particles.size());
        for(const auto& p:particles)state.particles.push_back(ugkp::thermal::adaptParticleState<Vec3>(p));
        return true;
    }
};
}
