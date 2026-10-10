#ifndef CHMT_ABLATION_WALLCLOSUREHOST_H
#define CHMT_ABLATION_WALLCLOSUREHOST_H
#include "ablation/InterfaceMath.H"
#include "coupling/IntervalContracts.H"
#include "../../../common/gasWall/WallGeometryBuilder.H"
#include <memory>
namespace chmt {
// Geometry is cached per immutable stage version. The caller invalidates this
// cache on rollback/alternate geometry, whose version can be reused by a trial.
class WallClosureHost {
    ugkwp::gaswall::WallGeometryCache<Real> cache_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<Real,Ns,24>> smallWorkspace_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<Real,Ns,48>> mediumWorkspace_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<Real,Ns,128>> largeWorkspace_;
    ugkwp::gaswall::WallStatus failureStatus_;
    ugkwp::gaswall::WallGeometryOptions<Real> geometryOptions_;
    static ugkwp::gaswall::WallGeometryOptions<Real> geometryOptions(const ugkwp::gaswall::WallModelConfig<Real>& config){
        ugkwp::gaswall::WallGeometryOptions<Real> options;
        if(config.family==ugkwp::gaswall::WallFamily::BoundaryLayer
            &&config.model==ugkwp::gaswall::BoundaryLayerModel::ReactingSst&&config.enableSst){
            options.profileNodes=config.nodes;options.profileStretch=config.stretch;
        }
        return options;
    }
    std::vector<int> faces_;
    std::uint64_t version_=0;bool ready_=false;
public:
    const ugkwp::gaswall::WallStatus& failureStatus()const{return failureStatus_;}
    void clearGeometry(){ready_=false;cache_.clear();}
    const std::vector<ugkwp::gaswall::WallGeometryDescriptor<Real>>& descriptors()const{return cache_.descriptors;}
    bool matchesGeometry(std::uint64_t version,const std::vector<int>& faces,const ugkwp::gaswall::WallModelConfig<Real>& config)const{
        const auto options=geometryOptions(config);
        return ready_&&version_==version&&faces_==faces
            &&geometryOptions_.relativeTolerance==options.relativeTolerance&&geometryOptions_.maxRayCells==options.maxRayCells
            &&geometryOptions_.quadratureRefinement==options.quadratureRefinement&&geometryOptions_.profileNodes==options.profileNodes
            &&geometryOptions_.profileStretch==options.profileStretch&&geometryOptions_.firstSegmentGaussOrder==options.firstSegmentGaussOrder;
    }
    bool prepareGeometry(const HostMesh& mesh,const std::vector<int>& faces,std::string& error,
        const ugkwp::gaswall::WallModelConfig<Real>& config=defaultWallModel()){
        if(matchesGeometry(mesh.geometryVersion,faces,config)){error.clear();return true;}
        const auto options=geometryOptions(config);
        ugkwp::gaswall::WallGeometryMesh<Real> input;input.geometryVersion=mesh.geometryVersion;
        input.volumes=mesh.volumes;input.owner=mesh.owner;input.neighbour=mesh.neighbour;
        input.faceOffsets=mesh.faceOffsets;input.facePoints=mesh.facePoints;
        input.points.reserve(mesh.points.size());input.cellCentres.reserve(mesh.cellCentres.size());
        for(const auto& v:mesh.points)input.points.push_back({{v.x,v.y,v.z}});
        for(const auto& v:mesh.cellCentres)input.cellCentres.push_back({{v.x,v.y,v.z}});
        if(mesh.boundaryKind.size()!=mesh.owner.size()){error="wall geometry boundary classification missing";return false;}
        input.physicalWall.resize(mesh.owner.size());
        for(std::size_t f=0;f<mesh.owner.size();++f)input.physicalWall[f]=mesh.boundaryKind[f]==BoundaryKind::NoSlip||mesh.boundaryKind[f]==BoundaryKind::Interface;
        ugkwp::gaswall::WallStatus status;
        if(!cache_.rebuild(input,faces,status,options)){error="boundaryLayer unsupported geometry at face "+std::to_string(status.node)+" code "+std::to_string(int(status.code));return false;}
        faces_=faces;version_=mesh.geometryVersion;geometryOptions_=options;ready_=true;error.clear();return true;
    }
    template<class Read> bool sampleWith(Read read,bool sst,std::vector<GasWallMatchingSample>& output,std::string& error)const{
        if(!ready_){error="wall geometry is not prepared";return false;}
        std::vector<GasWallMatchingSample> result(cache_.descriptors.size());
        for(std::size_t f=0;f<result.size();++f){auto& sample=result[f];const auto& d=cache_.descriptors[f];
            GasPrimitive owner;SstPrimitive ownerTurbulence;
            if(!read(d.ownerCell,owner,ownerTurbulence)||!finite(owner.pressure)||owner.pressure<=0){
                error="wall mechanical owner pressure is unavailable at actual stage";return false;}
            sample.mechanicalPressure=owner.pressure;
            for(std::size_t j=0;j<d.matchingCells.size();++j){GasPrimitive gas;SstPrimitive turbulence;
                if(!read(d.matchingCells[j],gas,turbulence)){error="wall matching state is unavailable at actual stage";return false;}
                const Real weight=d.matchingWeights[j];sample.pressure+=weight*gas.pressure;
                sample.state.temperature+=weight*gas.temperature;sample.state.velocity[0]+=weight*gas.velocity.x;
                sample.state.velocity[1]+=weight*gas.velocity.y;sample.state.velocity[2]+=weight*gas.velocity.z;
                for(int s=0;s<Ns;++s)sample.state.massFraction[s]+=weight*gas.Y[s];
                if(sst){sample.state.k+=weight*turbulence.k;sample.state.omega+=weight*turbulence.omega;}
            }
        }
        output=std::move(result);error.clear();return true;
    }
    bool sample(const std::vector<GasPrimitive>& gas,const std::vector<SstPrimitive>& turbulence,bool sst,
        std::vector<GasWallMatchingSample>& output,std::string& error)const{
        return sampleWith([&](int c,GasPrimitive& w,SstPrimitive& t){
            if(c<0||std::size_t(c)>=gas.size()||(sst&&std::size_t(c)>=turbulence.size()))return false;
            w=gas[c];if(sst)t=turbulence[c];return true;},sst,output,error);
    }
    bool sample(const HostState& state,const PhysicsConfig& p,std::vector<GasWallMatchingSample>& output,std::string& error)const{
        return sampleWith([&](int c,GasPrimitive& w,SstPrimitive& t){
            if(c<0||std::size_t(c)>=state.gas.size()||std::size_t(c)>=state.gasMesh.volumes.size()
                ||!recoverGas(state.gas[c],state.gasMesh.volumes[c],p,w))return false;
            if(p.enableSst){if(std::size_t(c)>=state.sst.size()||state.gas[c].mass<=0)return false;
                t.k=state.sst[c].rhoK/state.gas[c].mass;t.omega=state.sst[c].rhoOmega/state.gas[c].mass;}
            return true;},p.enableSst,output,error);
    }
    bool contexts(const std::vector<GasWallMatchingSample>& samples,const ugkwp::GasMechanismView<Real,Ns>& mechanism,
        const ugkwp::gaswall::WallModelConfig<Real>& config,std::vector<GasWallClosureContext>& output,std::string& error){
        if(!ready_||samples.size()!=cache_.descriptors.size()){error="wall matching snapshot/geometry count mismatch";return false;}
        failureStatus_={};
        void* workspace=nullptr;int capacity=0;
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport){
            if(config.nodes<4||config.nodes>128){error="wall profile nodes outside supported range";return false;}
            if(config.nodes<=24){if(!smallWorkspace_)smallWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<Real,Ns,24>);workspace=smallWorkspace_.get();capacity=24;}
            else if(config.nodes<=48){if(!mediumWorkspace_)mediumWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<Real,Ns,48>);workspace=mediumWorkspace_.get();capacity=48;}
            else{if(!largeWorkspace_)largeWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<Real,Ns,128>);workspace=largeWorkspace_.get();capacity=128;}
        }
        std::vector<GasWallClosureContext> result(samples.size());
        for(std::size_t f=0;f<samples.size();++f){auto& c=result[f];const auto& d=cache_.descriptors[f];
            if(!finite(samples[f].mechanicalPressure)||samples[f].mechanicalPressure<=0){error="wall mechanical pressure snapshot is invalid";return false;}
            c.workspace=workspace;c.workspaceCapacity=capacity;c.failureStatus=&failureStatus_;c.mechanism=mechanism;c.matching=samples[f].state;c.matchingPressure=samples[f].pressure;c.mechanicalPressure=samples[f].mechanicalPressure;
            c.matchingDistance=d.matchingDistance;c.ownerDistance=d.ownerDistance;c.quadrature=d.quadrature();}
        output=std::move(result);error.clear();return true;
    }
};
}
#endif
