#ifndef CHMT_ABLATION_WALLCLOSUREHOST_H
#define CHMT_ABLATION_WALLCLOSUREHOST_H
#include "ablation/InterfaceMath.H"
#include "coupling/IntervalContracts.H"
#include "../../../common/gasWall/WallGeometryBuilder.H"
#include <memory>
#include "../../../common/gasTransport/GasModelCast.H"
namespace chmt {
// Geometry is cached per immutable stage version. The caller invalidates this
// cache on rollback/alternate geometry, whose version can be reused by a trial.
class WallClosureHost {
    ugkwp::gaswall::WallGeometryCache<Real> cache_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,24>> smallWorkspace_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,48>> mediumWorkspace_;
    std::unique_ptr<ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,128>> largeWorkspace_;
    ugkwp::gaswall::WallStatus failureStatus_;
    ugkwp::gaswall::WallGeometryOptions<Real> geometryOptions_;
    static ugkwp::gaswall::WallGeometryOptions<Real> geometryOptions(const ugkwp::gaswall::WallModelConfig<PhysicalReal>&){
        ugkwp::gaswall::WallGeometryOptions<Real> options;options.constructQuadrature=false;
        return options;
    }
    std::vector<std::vector<EnergyReal>> physicalDistance_,physicalWeight_;
    // Canonical host mechanism input is immutable for this closure owner.
    // Repeated history contexts reuse these allocations, so every borrowed view
    // remains valid until the owner is reconfigured or destroyed.
    std::vector<ugkwp::GasReactionData<EnergyReal>> physicalReactions_;
    std::vector<ugkwp::GasStoichTerm<EnergyReal>> physicalReactants_,physicalProducts_;
    std::vector<ugkwp::GasColliderEfficiency<EnergyReal>> physicalEfficiencies_;
    std::vector<EnergyReal> physicalBasis_;
    template<class D,class S> static void copyPhysical(std::vector<D>& output,const S* input,int count) {
        output.resize(input&&count>0?static_cast<std::size_t>(count):0);
        for(std::size_t i=0;i<output.size();++i)output[i]=ugkwp::castGasModel<D>(input[i]);
    }
    template<class SourceReal> ugkwp::GasMechanismView<EnergyReal,Ns> physicalMechanism(const ugkwp::GasMechanismView<SourceReal,Ns>& source) {
        if constexpr(std::is_same<SourceReal,EnergyReal>::value)return source;
        else {
            auto view=ugkwp::castGasMechanismViewMetadata<EnergyReal>(source);
            copyPhysical(physicalReactions_,source.reactions,source.reactionCount);
            copyPhysical(physicalReactants_,source.reactants,source.reactantTermCount);
            copyPhysical(physicalProducts_,source.products,source.productTermCount);
            copyPhysical(physicalEfficiencies_,source.efficiencies,source.efficiencyCount);
            copyPhysical(physicalBasis_,source.stoichiometricBasis,Ns*source.independentRank);
            view.reactions=source.reactions?physicalReactions_.data():nullptr;
            view.reactants=source.reactants?physicalReactants_.data():nullptr;
            view.products=source.products?physicalProducts_.data():nullptr;
            view.efficiencies=source.efficiencies?physicalEfficiencies_.data():nullptr;
            view.stoichiometricBasis=source.stoichiometricBasis?physicalBasis_.data():nullptr;
            return view;
        }
    }
    std::vector<int> faces_;
    std::uint64_t version_=0;bool ready_=false;
public:
    const ugkwp::gaswall::WallStatus& failureStatus()const{return failureStatus_;}
    void clearGeometry(){ready_=false;cache_.clear();physicalDistance_.clear();physicalWeight_.clear();}
    const std::vector<ugkwp::gaswall::WallGeometryDescriptor<Real>>& descriptors()const{return cache_.descriptors;}
    bool matchesGeometry(std::uint64_t version,const std::vector<int>& faces,const ugkwp::gaswall::WallModelConfig<PhysicalReal>& config)const{
        const auto options=geometryOptions(config);
        return ready_&&version_==version&&faces_==faces
            &&geometryOptions_.relativeTolerance==options.relativeTolerance&&geometryOptions_.maxRayCells==options.maxRayCells
            &&geometryOptions_.constructQuadrature==options.constructQuadrature
            &&geometryOptions_.quadratureRefinement==options.quadratureRefinement&&geometryOptions_.profileNodes==options.profileNodes
            &&geometryOptions_.profileStretch==options.profileStretch&&geometryOptions_.firstSegmentGaussOrder==options.firstSegmentGaussOrder;
    }
    bool prepareGeometry(const HostMesh& mesh,const std::vector<int>& faces,std::string& error,
        const ugkwp::gaswall::WallModelConfig<PhysicalReal>& config=defaultWallModel(),
        ugkp::diagnostics::DeviceContext* diagnostics=nullptr){
        if(matchesGeometry(mesh.geometryVersion,faces,config)){error.clear();return true;}
        auto options=geometryOptions(config);options.diagnostics=diagnostics;
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
        physicalDistance_.resize(cache_.descriptors.size());physicalWeight_.resize(cache_.descriptors.size());
        for(std::size_t f=0;f<cache_.descriptors.size();++f){
            const auto& d=cache_.descriptors[f];
            physicalDistance_[f].assign(d.distance.begin(),d.distance.end());
            physicalWeight_[f].assign(d.volumeWeight.begin(),d.volumeWeight.end());
        }
        faces_=faces;version_=mesh.geometryVersion;geometryOptions_=options;ready_=true;error.clear();return true;
    }
    template<class Read> bool sampleWith(Read read,bool sst,std::vector<GasWallMatchingSample>& output,std::string& error)const{
        if(!ready_){error="wall geometry is not prepared";return false;}
        std::vector<GasWallMatchingSample> result(cache_.descriptors.size());
        for(std::size_t f=0;f<result.size();++f){auto& sample=result[f];const auto& d=cache_.descriptors[f];
            GasPrimitive owner;SstPrimitive ownerTurbulence;
            if(!read(d.ownerCell,owner,ownerTurbulence)){
                error="wall mechanical owner pressure is unavailable at actual stage";return false;}
            sample.mechanicalPressure=owner.pressure;
            for(std::size_t j=0;j<d.matchingCells.size();++j){GasPrimitive gas;SstPrimitive turbulence;
                if(!read(d.matchingCells[j],gas,turbulence)){error="wall matching state is unavailable at actual stage";return false;}
                const EnergyReal weight=EnergyReal(d.matchingWeights[j]);sample.pressure+=weight*EnergyReal(gas.pressure);
                sample.state.temperature+=weight*EnergyReal(gas.temperature);sample.state.velocity[0]+=weight*EnergyReal(gas.velocity.x);
                sample.state.velocity[1]+=weight*EnergyReal(gas.velocity.y);sample.state.velocity[2]+=weight*EnergyReal(gas.velocity.z);
                for(int s=0;s<Ns;++s)sample.state.massFraction[s]+=weight*EnergyReal(gas.Y[s]);
                if(sst){sample.state.k+=weight*EnergyReal(turbulence.k);sample.state.omega+=weight*EnergyReal(turbulence.omega);}
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
            if(state.gas.empty()||state.gasMesh.volumes.empty())return false;
            if(!recoverGas(state.gas[c],gasPhaseVolume(state,c),p,w,nullptr,c))return false;
            if(p.enableSst){if(state.sst.empty())return false;
                t.k=state.sst[c].rhoK/EnergyReal(state.gas[c].mass);t.omega=state.sst[c].rhoOmega/state.gas[c].mass;}
            return true;},p.enableSst,output,error);
    }
    template<class SourceReal> bool contexts(const std::vector<GasWallMatchingSample>& samples,const ugkwp::GasMechanismView<SourceReal,Ns>& mechanism,
        const ugkwp::gaswall::WallModelConfig<PhysicalReal>& config,std::vector<GasWallClosureContext>& output,std::string& error,bool* numericalFailure=nullptr){
        if(numericalFailure)*numericalFailure=true;
        if((!ready_)||(samples.size()!=cache_.descriptors.size())){
            if(numericalFailure)*numericalFailure=samples.size()<=cache_.descriptors.size();
            error="wall matching snapshot/geometry count mismatch";return false;
        }
        failureStatus_={};
        void* workspace=nullptr;int capacity=0;
        if(config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport){
            if((config.nodes<2||config.nodes>128)){error="wall profile nodes outside supported range";return false;}
            if(config.nodes<=24){if(!smallWorkspace_)smallWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,24>);workspace=smallWorkspace_.get();capacity=24;}
            else if(config.nodes<=48){if(!mediumWorkspace_)mediumWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,48>);workspace=mediumWorkspace_.get();capacity=48;}
            else{if(!largeWorkspace_)largeWorkspace_.reset(new ugkwp::gaswall::WallWorkspace<EnergyReal,Ns,128>);workspace=largeWorkspace_.get();capacity=128;}
        }
        const auto physical=physicalMechanism(mechanism);
        std::vector<GasWallClosureContext> result(samples.size());
        for(std::size_t f=0;f<samples.size();++f){auto& c=result[f];const auto& d=cache_.descriptors[f];
            if((!finite(samples[f].mechanicalPressure)||samples[f].mechanicalPressure<=0)){error="wall mechanical pressure snapshot is invalid";return false;}
            c.workspace=workspace;c.workspaceCapacity=capacity;c.failureStatus=&failureStatus_;c.mechanism=physical;c.matching=samples[f].state;c.matchingPressure=samples[f].pressure;c.mechanicalPressure=samples[f].mechanicalPressure;
            c.matchingDistance=d.matchingDistance;c.ownerDistance=d.ownerDistance;c.quadrature.distance=physicalDistance_[f].data();c.quadrature.volumeWeight=physicalWeight_[f].data();
            c.quadrature.count=static_cast<int>(physicalDistance_[f].size());c.quadrature.volume=EnergyReal(d.volume);c.quadrature.firstMoment=EnergyReal(d.firstMoment);}
        output=std::move(result);error.clear();return true;
    }
};
}
#endif
