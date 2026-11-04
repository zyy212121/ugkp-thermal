#ifndef CHMT_GPU_SHAREDGASDEVICESTORAGE_CUH
#define CHMT_GPU_SHAREDGASDEVICESTORAGE_CUH
#include "gpu/HostTransferQueue.H"
#include "../../../common/thermal/ParticleSurfaceState.H"
#include "../../../common/gasBoundary/Resident.cuh"
#if defined(__CUDACC__)
#include "../../../common/gasTransport/GasStatusCompletion.cuh"
#endif
#include "../../../common/GpuDiagnosticsOwner.cuh"
#include "ablation/WallClosureHost.H"
#include "../../../common/gasTransport/GasBoundaryLayerModelState.H"
#include "../../../common/gasTransport/GasBoundaryLayerWorkspace.H"
#include "core/HostState.H"
#include "materials/Thermodynamics.H"
#include "../../../common/gasTransport/GasStateView.H"
#include "../../../common/gasTransport/GasGeometryPolicy.H"
#include "../../../common/gasTransport/GasModelIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include <cstring>
#include <climits>
namespace chmt {
struct SharedGasDeviceView:ugkwp::GasStateView<Real,Real,ugkwp::SstCoefficients> {
    // Borrowed from the common particle owner; null is the particle-free phase.
    const Real* gasVolumeFraction=nullptr;
    ugkp::thermal::ParticleWallExchangeView particleWall;
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
    ugkwp::GasBoundaryLayerModelState<Real,Ns> gasBoundaryLayerModel;
};
template<class T>struct GasAllocation {bool allocate(T&,std::size_t,CudaFault&,cudaStream_t){return true;}
    void swap(GasAllocation&)noexcept{} void rebindFault(CudaFault&)noexcept{} bool release(){return true;}};
template<class T>struct GasAllocation<T*> {
    Buffer<T> values;
    bool allocate(T*& pointer,std::size_t n,CudaFault& fault,cudaStream_t stream){
        const bool fresh=values.size()!=n;
        if(!values.resize(n,fault)||(fresh&&!values.zero(stream)))return false;
        pointer=values.data();return true;
    }
    void swap(GasAllocation& other)noexcept{values.swap(other.values);}
    void rebindFault(CudaFault& fault)noexcept{values.rebindFault(fault);}
    bool release(){return values.release();}
};
// Allocation and marshaling only. No numerical gas, chemistry or RK formula.
class SharedGasDeviceStorage {
    using Time=Real;using Energy=Real;using SstCoefficients=ugkwp::SstCoefficients;
    CudaFault fault_;
    HostTransferQueue transfers_{fault_};
    Buffer<unsigned long long> statusFirst_;
    ugkp::diagnostics::GpuOwner* diagnostics_=nullptr;
    SharedGasDeviceView view_{};
    ugkp::boundary::Resident boundary_;
    Buffer<SharedGasDeviceView> device_;
#define UGKWP_GAS_STATE_FIELD(Type,Name) GasAllocation<Type> owner_##Name;
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    Buffer<Real> speciesRho_,speciesInitial_,speciesFlux_,speciesGX_,speciesGY_,speciesGZ_,speciesLimiter_,speciesBoundary_,speciesScale_,diffusivity_,sound_,cp_,r_,thermoCoeff_,elements_,basis_;
    Buffer<int> compositionFixed_,cellStatus_,faceStatus_;
    Buffer<ugkwp::SutherlandSpecies<Real>> viscosityData_;
    Buffer<ugkwp::PowerLawDiffusivity<Real>> powerLawData_;
    Buffer<Real> viscosityQuarter_,viscosityDenominator_;
    Buffer<ugkwp::SpeciesThermoData<Real>> thermoData_;
    Buffer<ugkwp::GasReactionData<Real>> reactions_;
    Buffer<ugkwp::GasStoichTerm<Real>> reactants_,products_;
    Buffer<ugkwp::GasColliderEfficiency<Real>> efficiencies_;
    Buffer<Real> geometryOld_,geometryNew_,geometrySweeps_;
    std::uint64_t identity_=0,thermoIdentity_=0,mechanismIdentity_=0,geometryVersion_=0;
    bool sstInitialized_=false,poisoned_=false;
    std::unique_ptr<SharedGasDeviceStorage> rejected_;
    Buffer<unsigned char> previousWorkspace_;
    std::vector<Real> volume_,wallFaceArea_,temperatureStaging_;
    bool readTemperature(std::size_t n){
        return enqueueRead(view_.Tgas,n,temperatureStaging_);
    }
    Tolerances geometryTolerances_;
    int activeSpecies_=-1;
    std::unique_ptr<WallClosureHost> wallGeometry_;
    std::vector<int> wallFaces_;
    std::vector<ugkwp::gaswall::WallInput<Real,Ns>> wallHostInput_;
    std::uint64_t wallGeometryVersion_=0;bool wallGeometryReady_=false,wallModelFailed_=false,wallInputDirty_=false;
    Buffer<int> wallFaceSlot_,wallOwnerSlot_,wallStatus_,wallFaceList_,wallMatchOffsets_,wallMatchCells_;
    Buffer<Real> wallSpecies_,wallDistance_,wallWeights_,wallMatchWeights_;
    Buffer<ugkwp::GasBoundaryLayerExchange<Real>> wallExchange_;
    Buffer<ugkwp::GasBoundaryLayerSstClosure<Real>> wallSst_;
    Buffer<ugkwp::gaswall::WallInput<Real,Ns>> wallInput_;
    Buffer<ugkwp::gaswall::WallResidentOutput<Real,Ns>> wallOutput_;
    Buffer<ugkwp::gaswall::WallOutput<Real,Ns>> wallEvaluationOutput_;
    Buffer<Real> wallFluxProfile_;
    Buffer<ugkwp::gaswall::WallStatus> wallModelStatus_;
    Buffer<Real> wallAcceptedHistory_,wallCandidateHistory_;
    Buffer<ugkwp::gaswall::WallHistoryFrame<Ns>> wallAcceptedFrame_,wallCandidateFrame_;
    Buffer<unsigned char> wallWorkspace_;
    static std::size_t count(const char* name,std::size_t cells,std::size_t faces,std::size_t addressing){
        if(!std::strcmp(name,"cellFaceId"))return addressing;
        if(!std::strncmp(name,"pressureSchedule",16))return 0;
        if(!std::strncmp(name,"face",4)||!std::strncmp(name,"gasBoundary",11)
            ||!std::strncmp(name,"riemannBoundary",15)||!std::strncmp(name,"gasPhi",6)
            ||!std::strncmp(name,"sstPhi",6)||!std::strncmp(name,"sstBoundary",11)
            ||!std::strcmp(name,"Sfx")||!std::strcmp(name,"Sfy")||!std::strcmp(name,"Sfz")
            ||!std::strcmp(name,"magSf")||!std::strcmp(name,"deltaCoeffs")||!std::strcmp(name,"scheduledInletFaceMask"))return faces;
        return cells;
    }
    template<class T>bool upload(T* destination,const std::vector<T>& source){return transfers_.upload(destination,source.data(),source.size());}
    template<class T>bool uploadBuffer(Buffer<T>& owner,const std::vector<T>& values){
        if(owner.size()!=values.size()&&!transfers_.drain())return false;
        return owner.resize(values.size(),fault_)&&upload(owner.data(),values);
    }
    template<class T>bool allocateSpecies(Buffer<T>& owner,T*& pointer,std::size_t n){const bool fresh=owner.size()!=n;if(!owner.resize(n,fault_)||(fresh&&!owner.zero(transfers_.stream())))return false;pointer=owner.data();return true;}
    static bool meshField(const char* name){
        // Boundary configuration and its evaluated mirrors keep ownership across
        // coordinate-only mesh updates; topology changes rebuild the storage.
        if(!std::strncmp(name,"gasBoundary",11)||!std::strncmp(name,"riemannBoundary",15))return false;
        const char* fields[]={"Cx","Cy","Cz","V","cellLength","sstWallDistance","cellPlaneStart","cellPlaneCount","cellFaceId","faceOwner","faceNeighbour","facePeriodicPair","Sfx","Sfy","Sfz","magSf","faceCx","faceCy","faceCz","facePeriodicDx","facePeriodicDy","facePeriodicDz","deltaCoeffs","faceWeight","gasBoundaryKind","riemannBoundaryKind","sstBoundaryK","sstBoundaryOmega","sstBoundaryKMode","sstBoundaryOmegaMode","gasBoundaryRho","gasBoundaryP","gasBoundaryT","gasBoundaryUx","gasBoundaryUy","gasBoundaryUz","gasBoundaryUFix","gasBoundaryTFix","gasBoundaryRhoFix","gasBoundaryPFix","riemannBoundaryRho","riemannBoundaryP","riemannBoundaryT","riemannBoundaryUx","riemannBoundaryUy","riemannBoundaryUz","riemannBoundaryUFix","riemannBoundaryTFix","riemannBoundaryRhoFix","riemannBoundaryPFix","gasBoundaryPWave","gasBoundaryPWaveGamma","gasBoundaryPWaveFieldInf","gasBoundaryPWaveLInf","riemannBoundaryPWave","riemannBoundaryPWaveGamma","riemannBoundaryPWaveFieldInf","riemannBoundaryPWaveLInf"};
        for(const char* field:fields)if(!std::strcmp(name,field))return true;
        return false;
    }
    void rebindOwners()noexcept{
#define UGKWP_GAS_STATE_FIELD(Type,Name) owner_##Name.rebindFault(fault_);
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        speciesRho_.rebindFault(fault_);
        speciesInitial_.rebindFault(fault_);
        speciesFlux_.rebindFault(fault_);
        speciesGX_.rebindFault(fault_);
        speciesGY_.rebindFault(fault_);
        speciesGZ_.rebindFault(fault_);
        speciesLimiter_.rebindFault(fault_);
        speciesScale_.rebindFault(fault_);
        sound_.rebindFault(fault_);
        cp_.rebindFault(fault_);
        r_.rebindFault(fault_);
        cellStatus_.rebindFault(fault_);
        faceStatus_.rebindFault(fault_);
        geometryOld_.rebindFault(fault_);
        geometryNew_.rebindFault(fault_);
        geometrySweeps_.rebindFault(fault_);
        diffusivity_.rebindFault(fault_);
        thermoCoeff_.rebindFault(fault_);
        elements_.rebindFault(fault_);
        basis_.rebindFault(fault_);
        thermoData_.rebindFault(fault_);viscosityData_.rebindFault(fault_);powerLawData_.rebindFault(fault_);viscosityQuarter_.rebindFault(fault_);viscosityDenominator_.rebindFault(fault_);
        reactions_.rebindFault(fault_);
        reactants_.rebindFault(fault_);
        products_.rebindFault(fault_);
        efficiencies_.rebindFault(fault_);
        wallFaceSlot_.rebindFault(fault_);
        wallOwnerSlot_.rebindFault(fault_);
        wallStatus_.rebindFault(fault_);
        wallFaceList_.rebindFault(fault_);
        wallMatchOffsets_.rebindFault(fault_);
        wallMatchCells_.rebindFault(fault_);
        wallSpecies_.rebindFault(fault_);
        wallDistance_.rebindFault(fault_);
        wallWeights_.rebindFault(fault_);
        wallMatchWeights_.rebindFault(fault_);
        wallExchange_.rebindFault(fault_);
        wallSst_.rebindFault(fault_);
        wallInput_.rebindFault(fault_);
        wallOutput_.rebindFault(fault_);
        wallEvaluationOutput_.rebindFault(fault_);wallFluxProfile_.rebindFault(fault_);
        wallModelStatus_.rebindFault(fault_);
        wallAcceptedHistory_.rebindFault(fault_);wallCandidateHistory_.rebindFault(fault_);
        wallAcceptedFrame_.rebindFault(fault_);wallCandidateFrame_.rebindFault(fault_);
        speciesBoundary_.rebindFault(fault_);
        compositionFixed_.rebindFault(fault_);
        device_.rebindFault(fault_);
        wallWorkspace_.rebindFault(fault_);
        previousWorkspace_.rebindFault(fault_);
    }

    void exchangeResident(SharedGasDeviceStorage& other,bool mesh,bool wall){
#define UGKWP_GAS_STATE_FIELD(Type,Name) if(mesh||!meshField(#Name))owner_##Name.swap(other.owner_##Name);
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        speciesRho_.swap(other.speciesRho_);
        speciesInitial_.swap(other.speciesInitial_);
        speciesFlux_.swap(other.speciesFlux_);
        speciesGX_.swap(other.speciesGX_);
        speciesGY_.swap(other.speciesGY_);
        speciesGZ_.swap(other.speciesGZ_);
        speciesLimiter_.swap(other.speciesLimiter_);
        speciesScale_.swap(other.speciesScale_);
        sound_.swap(other.sound_);
        cp_.swap(other.cp_);
        r_.swap(other.r_);
        cellStatus_.swap(other.cellStatus_);
        faceStatus_.swap(other.faceStatus_);
        geometryOld_.swap(other.geometryOld_);
        geometryNew_.swap(other.geometryNew_);
        geometrySweeps_.swap(other.geometrySweeps_);
        boundary_.swap(other.boundary_);
        speciesBoundary_.swap(other.speciesBoundary_);compositionFixed_.swap(other.compositionFixed_);
        if(mesh){volume_.swap(other.volume_);wallFaceArea_.swap(other.wallFaceArea_);}
        if(wall){
            wallFaceSlot_.swap(other.wallFaceSlot_);
            wallOwnerSlot_.swap(other.wallOwnerSlot_);
            wallStatus_.swap(other.wallStatus_);
            wallFaceList_.swap(other.wallFaceList_);
            wallMatchOffsets_.swap(other.wallMatchOffsets_);
            wallMatchCells_.swap(other.wallMatchCells_);
            wallSpecies_.swap(other.wallSpecies_);
            wallDistance_.swap(other.wallDistance_);
            wallWeights_.swap(other.wallWeights_);
            wallMatchWeights_.swap(other.wallMatchWeights_);
            wallExchange_.swap(other.wallExchange_);
            wallSst_.swap(other.wallSst_);
            wallInput_.swap(other.wallInput_);
            wallOutput_.swap(other.wallOutput_);
            wallEvaluationOutput_.swap(other.wallEvaluationOutput_);wallFluxProfile_.swap(other.wallFluxProfile_);
            wallModelStatus_.swap(other.wallModelStatus_);
            wallAcceptedHistory_.swap(other.wallAcceptedHistory_);wallCandidateHistory_.swap(other.wallCandidateHistory_);
            wallAcceptedFrame_.swap(other.wallAcceptedFrame_);wallCandidateFrame_.swap(other.wallCandidateFrame_);
            wallGeometry_.swap(other.wallGeometry_);wallFaces_.swap(other.wallFaces_);wallHostInput_.swap(other.wallHostInput_);
            std::swap(wallGeometryVersion_,other.wallGeometryVersion_);std::swap(wallGeometryReady_,other.wallGeometryReady_);std::swap(wallModelFailed_,other.wallModelFailed_);std::swap(wallInputDirty_,other.wallInputDirty_);
        }
        rebindOwners();other.rebindOwners();
    }
    void exchangeTables(SharedGasDeviceStorage& other){
        diffusivity_.swap(other.diffusivity_);
        thermoCoeff_.swap(other.thermoCoeff_);
        elements_.swap(other.elements_);
        basis_.swap(other.basis_);
        thermoData_.swap(other.thermoData_);viscosityData_.swap(other.viscosityData_);powerLawData_.swap(other.powerLawData_);viscosityQuarter_.swap(other.viscosityQuarter_);viscosityDenominator_.swap(other.viscosityDenominator_);
        reactions_.swap(other.reactions_);
        reactants_.swap(other.reactants_);
        products_.swap(other.products_);
        efficiencies_.swap(other.efficiencies_);
        rebindOwners();other.rebindOwners();
    }
    void discardUnpublishedOptional(){
        if(!view_.gasSpecies.rho)speciesRho_.release();
        if(!view_.gasSpecies.initial)speciesInitial_.release();
        if(!view_.gasSpecies.flux)speciesFlux_.release();
        if(!view_.gasSpecies.gradX)speciesGX_.release();
        if(!view_.gasSpecies.gradY)speciesGY_.release();
        if(!view_.gasSpecies.gradZ)speciesGZ_.release();
        if(!view_.gasSpecies.limiter)speciesLimiter_.release();
        if(!view_.gasSpecies.boundaryMassFraction)speciesBoundary_.release();
        if(!view_.gasSpecies.compositionBoundaryFixed)compositionFixed_.release();
        if(!view_.gasSpecies.positivityScale)speciesScale_.release();
        if(!view_.gasSpecies.soundSpeed)sound_.release();
        if(!view_.gasSpecies.heatCapacity)cp_.release();
        if(!view_.gasSpecies.gasConstant)r_.release();
        if(!view_.gasSpecies.cellStatus)cellStatus_.release();
        if(!view_.gasSpecies.faceStatus)faceStatus_.release();
    }
    bool releaseOwners(){bool ok=transfers_.drain();
#define UGKWP_GAS_STATE_FIELD(Type,Name) ok=owner_##Name.release()&&ok;
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        ok=speciesRho_.release()&&ok;
        ok=speciesInitial_.release()&&ok;
        ok=speciesFlux_.release()&&ok;
        ok=speciesGX_.release()&&ok;
        ok=speciesGY_.release()&&ok;
        ok=speciesGZ_.release()&&ok;
        ok=speciesLimiter_.release()&&ok;
        ok=speciesScale_.release()&&ok;
        ok=sound_.release()&&ok;
        ok=cp_.release()&&ok;
        ok=r_.release()&&ok;
        ok=cellStatus_.release()&&ok;
        ok=faceStatus_.release()&&ok;
        ok=geometryOld_.release()&&ok;
        ok=geometryNew_.release()&&ok;
        ok=geometrySweeps_.release()&&ok;
        ok=diffusivity_.release()&&ok;
        ok=thermoCoeff_.release()&&ok;
        ok=elements_.release()&&ok;
        ok=basis_.release()&&ok;
        ok=thermoData_.release()&&ok;ok=viscosityData_.release()&&ok;ok=powerLawData_.release()&&ok;ok=viscosityQuarter_.release()&&ok;ok=viscosityDenominator_.release()&&ok;
        ok=reactions_.release()&&ok;
        ok=reactants_.release()&&ok;
        ok=products_.release()&&ok;
        ok=efficiencies_.release()&&ok;
        ok=wallFaceSlot_.release()&&ok;
        ok=wallOwnerSlot_.release()&&ok;
        ok=wallStatus_.release()&&ok;
        ok=wallFaceList_.release()&&ok;
        ok=wallMatchOffsets_.release()&&ok;
        ok=wallMatchCells_.release()&&ok;
        ok=wallSpecies_.release()&&ok;
        ok=wallDistance_.release()&&ok;
        ok=wallWeights_.release()&&ok;
        ok=wallMatchWeights_.release()&&ok;
        ok=wallExchange_.release()&&ok;
        ok=wallSst_.release()&&ok;
        ok=wallInput_.release()&&ok;
        ok=wallOutput_.release()&&ok;
        ok=wallEvaluationOutput_.release()&&ok;ok=wallFluxProfile_.release()&&ok;
        ok=wallModelStatus_.release()&&ok;
        ok=wallAcceptedHistory_.release()&&ok;ok=wallCandidateHistory_.release()&&ok;
        ok=wallAcceptedFrame_.release()&&ok;ok=wallCandidateFrame_.release()&&ok;
        ok=speciesBoundary_.release()&&ok;
        ok=compositionFixed_.release()&&ok;
        ok=device_.release()&&ok;
        ok=wallWorkspace_.release()&&ok;
        ok=previousWorkspace_.release()&&ok;
        return ok;
    }
public:
    SharedGasDeviceStorage()=default;
    SharedGasDeviceStorage(const SharedGasDeviceStorage&)=delete;
    SharedGasDeviceStorage& operator=(const SharedGasDeviceStorage&)=delete;
    SharedGasDeviceView& hostView(){return view_;}
    const SharedGasDeviceView& hostView()const{return view_;}
    SharedGasDeviceView* deviceView()const{return device_.data();}
    const std::string& error()const{return fault_.message;}
    bool wallModelFailed()const{return wallModelFailed_;}
    void bindDiagnostics(ugkp::diagnostics::GpuOwner& owner){
        diagnostics_=&owner;view_.diagnostics=owner.device();
        view_.gasSpecies.thermo.diagnostics=owner.device();
    }
    ~SharedGasDeviceStorage(){transfers_.drain();}
    void bindStream(cudaStream_t stream){transfers_.bindStream(stream);}
    // Complete all sparse/full reads and the diagnostic snapshot together.
    // Callers publish candidate numerical data only after this succeeds.
    bool observeDiagnostics(){
        ugkp::diagnostics::DeviceContext context{};
        if(diagnostics_&&!transfers_.download(&context,diagnostics_->device(),1)){
            transfers_.drain();return false;
        }
        if(!transfers_.finish()){
            if(diagnostics_){
                diagnostics_->recordCudaFailure(fault_.lastCode);
                diagnostics_->dispatchPending();
            }
            return false;
        }
        if(diagnostics_){
            diagnostics_->consumeCompletedSnapshot(context);
            if(diagnostics_->dispatchPending()){fault_.message="fatal gas diagnostic at completed operation";return false;}
        }
        return true;
    }
    bool refreshView(){
        if(diagnostics_){view_.diagnostics=diagnostics_->device();view_.gasSpecies.thermo.diagnostics=diagnostics_->device();}
        if(device_.size()!=1&&!transfers_.drain())return false;
        return device_.resize(1,fault_)&&transfers_.upload(device_.data(),&view_,1);
    }
    bool bindStageGeometry(const HostStageGeometry& stage){
        if((view_.nCells&&(stage.oldVolume.empty()||stage.newVolume.empty()))||(view_.nFaces&&stage.sweptVolume.empty())){fault_.message="missing consumed stage geometry storage";return false;}
        if(!uploadBuffer(geometryOld_,stage.oldVolume)||!uploadBuffer(geometryNew_,stage.newVolume)||!uploadBuffer(geometrySweeps_,stage.sweptVolume))return false;
        view_.gasGeometry.absoluteGeometryTolerance=geometryTolerances_.absoluteGeometry;view_.gasGeometry.relativeGeometryTolerance=geometryTolerances_.relativeGeometry;
        view_.gasGeometry.enabled=true;view_.gasGeometry.oldVolume=geometryOld_.data();view_.gasGeometry.newVolume=geometryNew_.data();view_.gasGeometry.faceSweptVolume=geometrySweeps_.data();view_.gasGeometry.interval=stage.interval;
        return refreshView();
    }
    bool clearGeometry(){if(!view_.gasGeometry.enabled)return true;view_.gasGeometry={};return refreshView();}
    template<class T>bool enqueueRead(const T* pointer,std::size_t n,std::vector<T>& output){
        try{
            output.resize(n);
            if(transfers_.download(output.data(),pointer,n))return true;
        }catch(...){transfers_.drain();throw;}
        transfers_.drain();return false;
    }
    template<class T>bool read(const T* pointer,std::size_t n,std::vector<T>& output){
        std::vector<T> candidate;
        if(!enqueueRead(pointer,n,candidate)||!observeDiagnostics())return false;
        output=std::move(candidate);return true;
    }
    template<class T>bool write(T* pointer,const std::vector<T>& values){return upload(pointer,values);}
    void invalidateWallGeometry(){if(wallGeometry_)wallGeometry_->clearGeometry();wallGeometryReady_=false;}
    const WallClosureHost* wallGeometry()const{return wallGeometry_.get();}
    bool runtimeFailure()const{return fault_.runtimeFailure;}
    void beginOperation(){fault_.message.clear();fault_.runtimeFailure=false;}
    bool clearTrialStatus(){
        wallModelFailed_=false;
        view_.gasBoundaryLayer.preparedFirstStage=false;
        if(view_.gasBoundaryLayer.enabled&&(!wallStatus_.zero(transfers_.stream())||!wallExchange_.zero(transfers_.stream())||!wallModelStatus_.zero(transfers_.stream())))return false;
        return cellStatus_.zero(transfers_.stream())&&faceStatus_.zero(transfers_.stream());}
    bool checkStatus(std::string& error){
#if defined(__CUDACC__)
        if(!fault_.check(ugkwp::enqueueGasStatusCheck(cellStatus_.data(),cellStatus_.size(),
            faceStatus_.data(),faceStatus_.size(),diagnostics_?diagnostics_->device():nullptr,
            statusFirst_.data(),2630,26,transfers_.stream()),"queue shared gas status check")){
            error=fault_.message;return false;
        }
#else
        // Host-only CUDA shims preserve the same raw-status contract.
        std::vector<int> cells,faces;
        if(!enqueueRead(cellStatus_.data(),cellStatus_.size(),cells)
            ||!enqueueRead(faceStatus_.data(),faceStatus_.size(),faces)||!observeDiagnostics()){error=fault_.message;return false;}
        for(std::size_t c=0;c<cells.size();++c)if(cells[c]==int(ugkwp::GasTransportCode::InvalidStorage)){error="shared gas cell "+std::to_string(c)+" status "+std::to_string(cells[c]);return false;}
        for(std::size_t f=0;f<faces.size();++f)if(faces[f]==int(ugkwp::GasTransportCode::InvalidStorage)){error="shared gas face "+std::to_string(f)+" status "+std::to_string(faces[f]);return false;}
#endif
        error.clear();return true;
    }
    bool configure(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
        const ugkwp::GasMechanismConfiguration& mechanism,const HostState& state,std::string& error){
        if(!transfers_.drain()){error=fault_.message;return false;}
        beginOperation();
        if(!statusFirst_.resize(1,fault_)){error=fault_.message;return false;}
        if(poisoned_){error="shared gas device view rollback failed; storage must be destroyed";return false;}
        const bool resident=device_.data()!=nullptr;
        const bool retain=resident&&owner_rho.values.size()==state.gas.size()
            &&owner_gasPhiRho.values.size()==state.gasMesh.owner.size()
            &&owner_cellFaceId.values.size()==state.gasMesh.cellFaces.size();
        const bool importSst=retain&&model.physics.enableSst&&!sstInitialized_;
        const bool sameMesh=retain&&geometryVersion_==state.gasMesh.geometryVersion;
        const bool sameWall=sameMesh&&wallFaces_==state.surface.gasFace;
        std::unique_ptr<SharedGasDeviceStorage> next(new SharedGasDeviceStorage);
        next->bindStream(transfers_.stream());
        if(diagnostics_)next->bindDiagnostics(*diagnostics_);
        if(retain){next->view_=view_;next->sstInitialized_=sstInitialized_;next->geometryVersion_=geometryVersion_;
            next->exchangeResident(*this,sameMesh,sameWall);
            if(importSst){next->owner_rhoK.swap(owner_rhoK);next->owner_rhoOmega.swap(owner_rhoOmega);}}

        next->device_.swap(device_);next->wallWorkspace_.swap(wallWorkspace_);
        next->rebindOwners();rebindOwners();
        bool publishing=false,prepared=false;
        try{prepared=next->configureImpl(model,gas,mechanism,state,error,!retain,!sameMesh,publishing);}
        catch(const std::exception& failure){error=failure.what();}
        // Candidate uploads must finish before rollback swaps or frees owners.
        if(!next->transfers_.drain()){prepared=false;error=next->error();}
        if(!prepared){
            if(next->previousWorkspace_.data())next->wallWorkspace_.swap(next->previousWorkspace_);
            next->device_.swap(device_);next->wallWorkspace_.swap(wallWorkspace_);
            if(importSst){next->owner_rhoK.swap(owner_rhoK);next->owner_rhoOmega.swap(owner_rhoOmega);}
            if(retain)next->exchangeResident(*this,sameMesh,sameWall);
            rebindOwners();next->rebindOwners();
            if(publishing&&resident&&(!refreshView()||!transfers_.drain())){
                poisoned_=true;rejected_=std::move(next);error+="; shared gas device view rollback failed";
            }else if(retain)discardUnpublishedOptional();
            fault_.message=error;return false;
        }

        next->exchangeResident(*this,true,true);next->exchangeTables(*this);
        next->device_.swap(device_);next->wallWorkspace_.swap(wallWorkspace_);
        view_=next->view_;identity_=next->identity_;thermoIdentity_=next->thermoIdentity_;mechanismIdentity_=next->mechanismIdentity_;
        geometryVersion_=next->geometryVersion_;geometryTolerances_=next->geometryTolerances_;activeSpecies_=next->activeSpecies_;sstInitialized_=next->sstInitialized_;
        rebindOwners();next->rebindOwners();
        if(!next->releaseOwners()){error=next->error();fault_.message=error;return false;}
        error.clear();return true;
    }
private:
    bool configureImpl(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
        const ugkwp::GasMechanismConfiguration& mechanism,const HostState& state,std::string& error,
        bool initialize,bool geometry,bool& publishing){
        geometryTolerances_=model.physics.tolerances;
        if(model.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
            if(!wallGeometry_)wallGeometry_.reset(new WallClosureHost);
            if(wallFaces_.empty())wallFaces_=state.surface.gasFace;
            if(wallFaces_.empty()){error="boundaryLayer requires coupled physical wall faces";return false;}
            view_.gasBoundaryLayer.enabled=true;view_.gasBoundaryLayer.count=int(wallFaces_.size());
            auto& layer=view_.gasBoundaryLayerModel;layer.config=model.physics.wallModel;
            layer.config.enableSst=model.physics.enableSst;layer.config.turbulentPrandtl=model.physics.sst.turbulentPrandtl;
            layer.config.turbulentSchmidt=model.physics.sst.turbulentSchmidt;layer.useSuppliedWallState=true;
        }else view_.gasBoundaryLayer.enabled=false;
        const std::size_t nc=state.gas.size(),nf=state.gasMesh.owner.size();
        if(nc&&state.gasMesh.volumes.empty()){error="invalid shared gas dimensions";return false;}
        const auto& mesh=state.gasMesh;
        if((nc&&(mesh.cellCentres.empty()||mesh.cellFaceOffsets.empty()))||(nf&&(mesh.neighbour.empty()||mesh.periodicPartner.empty()
            ||mesh.areaVectors.empty()||mesh.faceCentres.empty()||mesh.boundaryKind.empty()||mesh.boundaryPrimitive.empty()))){error="incomplete common gas mesh addressing";return false;}
        const bool mixture=gas.mode!=ugkwp::GasMode::SingleLegacy;
        activeSpecies_=model.physics.singleGasSpecies;
        view_.nCells=int(nc);view_.nFaces=int(nf);
        if(initialize||geometry){view_.nInternalFaces=0;
            while(view_.nInternalFaces<int(nf)&&state.gasMesh.neighbour[view_.nInternalFaces]>=0)++view_.nInternalFaces;}
#define UGKWP_GAS_STATE_FIELD(Type,Name) if(!owner_##Name.allocate(view_.Name,count(#Name,nc,nf,state.gasMesh.cellFaces.size()),fault_,transfers_.stream())){error=fault_.message;return false;}
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        const bool importSpecies=mixture&&!view_.gasSpecies.rho;
        const bool importSst=model.physics.enableSst&&!sstInitialized_;
        auto& sp=view_.gasSpecies;sp.mode=gas.mode;sp.thermoControls=gas.chemistryControls.thermo;sp.chemistryControls=gas.chemistryControls;sp.turbulentSchmidt=gas.turbulentSchmidt;
#define ALLOC(Owner,Field,N) if(!allocateSpecies(Owner,sp.Field,(N))){error=fault_.message;return false;}
        if(mixture){
        ALLOC(speciesRho_,rho,Ns*nc);ALLOC(speciesInitial_,initial,Ns*nc);ALLOC(speciesFlux_,flux,Ns*nf);
        ALLOC(speciesGX_,gradX,Ns*nc);ALLOC(speciesGY_,gradY,Ns*nc);ALLOC(speciesGZ_,gradZ,Ns*nc);
        ALLOC(speciesLimiter_,limiter,nc);ALLOC(speciesBoundary_,boundaryMassFraction,Ns*nf);ALLOC(speciesScale_,positivityScale,Ns*nc);
        ALLOC(compositionFixed_,compositionBoundaryFixed,nf);ALLOC(sound_,soundSpeed,nc);ALLOC(cp_,heatCapacity,nc);ALLOC(r_,gasConstant,nc);
        }
        ALLOC(cellStatus_,cellStatus,nc);ALLOC(faceStatus_,faceStatus,nf);
        sp.chemistryStatus=nullptr;
#undef ALLOC
        if(mixture){
        sp.diffusivity=nullptr;sp.mechanism={};
        if(!uploadBuffer(thermoData_,gas.species)||!uploadBuffer(thermoCoeff_,gas.coefficients)||!uploadBuffer(elements_,gas.elementComposition)){error=fault_.message;return false;}
        sp.thermo=gas.thermoView<Ns>();sp.thermo.species=thermoData_.data();sp.thermo.coefficients=thermoCoeff_.data();sp.thermo.elementComposition=elements_.data();
        if(gas.diffusionModel==ugkwp::GasDiffusionModel::Constant||view_.gasBoundaryLayer.enabled){if(!uploadBuffer(diffusivity_,gas.diffusionCoefficients)){error=fault_.message;return false;}sp.diffusivity=diffusivity_.data();}
        if(gas.mode==ugkwp::GasMode::MixtureChemistry){
            if(!uploadBuffer(reactions_,mechanism.reactions)||!uploadBuffer(reactants_,mechanism.reactants)||!uploadBuffer(products_,mechanism.products)
                ||!uploadBuffer(efficiencies_,mechanism.efficiencies)||!uploadBuffer(basis_,mechanism.stoichiometricBasis)){error=fault_.message;return false;}
            sp.mechanism=mechanism.mechanismView<Ns>();sp.mechanism.reactions=reactions_.data();sp.mechanism.reactants=reactants_.data();sp.mechanism.products=products_.data();sp.mechanism.efficiencies=efficiencies_.data();sp.mechanism.stoichiometricBasis=basis_.data();
        }
        }else{
            sp.thermo={};sp.mechanism={};sp.diffusivity=nullptr;
            const auto& t=model.physics.species[activeSpecies_];
            view_.Rgas=t.R;view_.gasCp=t.cp0;view_.gammaGas=t.cp0/(t.cp0-t.R);
        }
        identity_=gas.speciesOrderHash;thermoIdentity_=gas.thermoHash;mechanismIdentity_=mechanism.mechanismHash;
        view_.rhoMin=1e-30;view_.TgasMin=1e-12;view_.gasMu=model.physics.gasViscosity;
        const auto& transport=model.physics;const int count=transport.gasViscositySpecies;
        const std::vector<ugkwp::SutherlandSpecies<Real>> data(transport.gasSutherland,transport.gasSutherland+count);
        const std::vector<Real> quarter(transport.gasMassRatioQuarter,transport.gasMassRatioQuarter+count*count);
        const std::vector<Real> denominator(transport.gasWilkeDenominator,transport.gasWilkeDenominator+count*count);
        if(!uploadBuffer(viscosityData_,data)||!uploadBuffer(viscosityQuarter_,quarter)||!uploadBuffer(viscosityDenominator_,denominator)){error=fault_.message;return false;}
        const std::vector<ugkwp::PowerLawDiffusivity<Real>> diffusion(transport.gasPowerLaw,transport.gasPowerLaw+(transport.gasPowerLawEnabled?Ns:0));
        if(!uploadBuffer(powerLawData_,diffusion)){error=fault_.message;return false;}
        view_.gasViscosity=molecularTransportView(transport);view_.gasViscosity.species=viscosityData_.data();
        view_.gasViscosity.massRatioQuarter=viscosityQuarter_.data();view_.gasViscosity.wilkeDenominator=viscosityDenominator_.data();
        view_.gasViscosity.diffusion=transport.gasPowerLawEnabled?powerLawData_.data():nullptr;
        view_.gasPrClamped=1;
        view_.gasThermalConductivity=model.physics.gasConductivity;
        // CHMT uses zero for disabled conduction, never a negative unset sentinel.
        view_.gasThermalConductivityConfigured=true;
        view_.turbulentPrandtl=model.physics.sst.turbulentPrandtl;
        view_.sstCoefficients=ugkwp::defaultSstCoefficients();
        const auto& st=model.physics.sst;auto& coeff=view_.sstCoefficients;
        coeff.a1=st.a1;coeff.betaStar=st.betaStar;coeff.beta1=st.beta1;coeff.beta2=st.beta2;
        coeff.gamma1=st.gamma1;coeff.gamma2=st.gamma2;coeff.alphaK1=st.sigmaK1;coeff.alphaK2=st.sigmaK2;
        coeff.alphaOmega1=st.sigmaOmega1;coeff.alphaOmega2=st.sigmaOmega2;coeff.c1=st.productionLimit;
        view_.sstKMin=maxValue(st.minimumK,1e-12);view_.sstOmegaMin=maxValue(st.minimumOmega,1e-12);
        view_.turbulenceModel=model.physics.enableSst?3:0;view_.sstConfigured=model.physics.enableSst;view_.sstWallTreatment=int(model.physics.wallModel.family);
        view_.sstWallKappa=.41;view_.sstWallE=9.8;view_.sstWallCmu=.09;view_.lesDeltaCoeff=1;view_.waleCw=.325;view_.smagorinskyCs=.17;
        if(geometry&&!uploadGeometry(state.gasMesh,false)){error=fault_.message;return false;}
        if(!geometry&&!prepareWallGeometry(state.gasMesh)){error=fault_.message;return false;}
        if(view_.gasBoundaryLayer.enabled&&wallFaceArea_.empty()){
            wallFaceArea_.reserve(state.gasMesh.areaVectors.size());
            for(const auto& area:state.gasMesh.areaVectors)wallFaceArea_.push_back(mag(area));
        }
        if(initialize){if(!uploadState(state,error))return false;}
        else{
            if(importSpecies){std::vector<Real> values(Ns*nc);for(std::size_t c=0;c<nc;++c)for(int s=0;s<Ns;++s)values[s*nc+c]=state.gas[c].species[s]/gasPhaseVolume(state,c);
                if(!upload(sp.rho,values)){error=fault_.message;return false;}}
            if(importSst){if(nc&&state.sst.empty()){error="missing supplied SST inventory";return false;}
                std::vector<Real> k(nc),omega(nc);for(std::size_t c=0;c<nc;++c){k[c]=state.sst[c].rhoK/gasPhaseVolume(state,c);omega[c]=state.sst[c].rhoOmega/gasPhaseVolume(state,c);}
                if(!upload(view_.rhoK,k)||!upload(view_.rhoOmega,omega)){error=fault_.message;return false;}}
        }
        sstInitialized_=sstInitialized_||model.physics.enableSst;
        if(!configureWallWorkspace(model.physics,error))return false;
        view_.gasBoundaryLayer.preparedFirstStage=false;
        publishing=true;
        if(!refreshView()){error=fault_.message;return false;}
        error.clear();return true;
    }
public:
    bool configureWallWorkspace(const PhysicsConfig& physics,std::string& error){
        if(!view_.gasBoundaryLayer.enabled)return true;
        auto& layer=view_.gasBoundaryLayerModel;
        ugkwp::BoundaryLayerWorkspaceSizing sizing;
        layer.workspaceCapacity=0;
        if(layer.config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport){
            layer.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(layer.config.nodes);
            const std::size_t bytes=ugkwp::gasBoundaryLayerWorkspaceBytes<Real,Ns>(layer.config.nodes);
#if defined(__CUDACC__) && defined(CUDART_VERSION)
            int device=0,multiprocessors=0;
            if(!fault_.check(cudaGetDevice(&device),"query wall workspace device")
                ||!fault_.check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device),"query wall workspace multiprocessors")){error=fault_.message;return false;}
            sizing=ugkwp::resolveBoundaryLayerWorkspace(view_.gasBoundaryLayer.count,physics.wallWorkspaceSlots,multiprocessors,0,bytes);
#else
            // Host CUDA shims do not represent a GPU. A serial worker gives a
            // deterministic functional test without pretending device sizing.
            sizing=ugkwp::resolveBoundaryLayerWorkspace(view_.gasBoundaryLayer.count,1,1,bytes,bytes);
#endif
            if(sizing.slots==0){error="boundaryLayer workspace has no compiled capacity or its byte count is not representable";return false;}
        }
        if(wallWorkspace_.size()!=sizing.bytes)wallWorkspace_.swap(previousWorkspace_);
        if(!wallWorkspace_.resize(sizing.bytes,fault_)){error=fault_.message;return false;}
        layer.workspaceCount=sizing.slots;layer.workspace=wallWorkspace_.data();
        if(!wallEvaluationOutput_.resize(sizing.slots,fault_))return false;
        layer.evaluationOutput=wallEvaluationOutput_.data();
        std::fprintf(stderr,"CHMT boundaryLayer workspace requested=%d resolved=%d capacity=%d bytes=%zu\n",
            physics.wallWorkspaceSlots,layer.workspaceCount,layer.workspaceCapacity,sizing.bytes);
        return true;
    }
    bool prepareWallGeometry(const HostMesh& mesh){
        if(!wallGeometry_||!view_.gasBoundaryLayer.enabled)return true;
        if(wallGeometryReady_&&wallGeometryVersion_==mesh.geometryVersion
            &&wallGeometry_->matchesGeometry(mesh.geometryVersion,wallFaces_,view_.gasBoundaryLayerModel.config))return true;
        if(!transfers_.drain())return false;
        std::string error;if(!wallGeometry_->prepareGeometry(mesh,wallFaces_,error,view_.gasBoundaryLayerModel.config,geometryTolerances_.diagnostics)){fault_.message=error;return false;}
        const auto& geometry=wallGeometry_->descriptors();const int count=int(geometry.size());
        std::vector<int> faceSlot(mesh.owner.size(),-1),ownerSlot(mesh.volumes.size(),-1),offsets(1,0),cells;
        std::vector<Real> distances,weights,matchingWeights;std::vector<std::size_t> qStart;
        for(int i=0;i<count;++i){const auto& d=geometry[i];
            faceSlot[d.wallFace]=i;ownerSlot[d.ownerCell]=i;qStart.push_back(distances.size());
            distances.insert(distances.end(),d.distance.begin(),d.distance.end());weights.insert(weights.end(),d.volumeWeight.begin(),d.volumeWeight.end());
            cells.insert(cells.end(),d.matchingCells.begin(),d.matchingCells.end());matchingWeights.insert(matchingWeights.end(),d.matchingWeights.begin(),d.matchingWeights.end());offsets.push_back(int(cells.size()));}
        if(!uploadBuffer(wallFaceSlot_,faceSlot)||!uploadBuffer(wallOwnerSlot_,ownerSlot)||!uploadBuffer(wallFaceList_,wallFaces_)
            ||!uploadBuffer(wallMatchOffsets_,offsets)||!uploadBuffer(wallMatchCells_,cells)||!uploadBuffer(wallMatchWeights_,matchingWeights)
            ||!uploadBuffer(wallDistance_,distances)||!uploadBuffer(wallWeights_,weights))return false;
        wallHostInput_.assign(count,{});
        for(int i=0;i<count;++i){const auto& d=geometry[i];auto& input=wallHostInput_[i];
            input.matchingDistance=d.matchingDistance;input.ownerDistance=d.ownerDistance;
            for(int k=0;k<3;++k)input.normal[k]=d.normal[k];
            input.quadrature=d.quadrature();input.quadrature.distance=d.distance.empty()?nullptr:wallDistance_.data()+qStart[i];input.quadrature.volumeWeight=d.volumeWeight.empty()?nullptr:wallWeights_.data()+qStart[i];}
        for(auto& input:wallHostInput_)input.diagnostics=view_.diagnostics;
        if(!uploadBuffer(wallInput_,wallHostInput_)||!wallOutput_.resize(count,fault_)||!wallModelStatus_.resize(count,fault_)
            ||!wallExchange_.resize(count,fault_)||!wallSst_.resize(count,fault_)||!wallStatus_.resize(count,fault_)||!wallSpecies_.resize(Ns*count,fault_))return false;
        auto& wall=view_.gasBoundaryLayer;wall.faceSlot=wallFaceSlot_.data();wall.ownerSlot=wallOwnerSlot_.data();
        wall.exchange=wallExchange_.data();wall.sst=wallSst_.data();wall.status=wallStatus_.data();wall.speciesFlux=wallSpecies_.data();
        auto& layer=view_.gasBoundaryLayerModel;layer.input=wallInput_.data();layer.output=wallOutput_.data();layer.status=wallModelStatus_.data();
        const std::size_t fluxCount=layer.volumeSstClosure&&layer.config.enableSst?std::size_t(count)*3*(layer.config.nodes+1):0;
        if(!wallFluxProfile_.resize(fluxCount,fault_))return false;
        layer.fluxProfile=wallFluxProfile_.data();
        if(layer.config.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport){
            const std::size_t values=std::size_t(count)*layer.config.nodes*(Ns+4);
            if(!wallAcceptedHistory_.resize(values,fault_)||!wallCandidateHistory_.resize(values,fault_)
                ||!wallAcceptedFrame_.resize(count,fault_)||!wallCandidateFrame_.resize(count,fault_)
                ||!wallAcceptedFrame_.zero(transfers_.stream())||!wallCandidateFrame_.zero(transfers_.stream()))return false;
            layer.acceptedHistory=wallAcceptedHistory_.data();layer.candidateHistory=wallCandidateHistory_.data();
            layer.acceptedFrame=wallAcceptedFrame_.data();layer.candidateFrame=wallCandidateFrame_.data();
        }
        layer.faces=wallFaceList_.data();layer.matchingOffsets=wallMatchOffsets_.data();layer.matchingCells=wallMatchCells_.data();layer.matchingWeights=wallMatchWeights_.data();
        wallInputDirty_=false;wallGeometryReady_=true;wallGeometryVersion_=mesh.geometryVersion;
        return wallExchange_.zero(transfers_.stream())&&wallStatus_.zero(transfers_.stream())&&wallModelStatus_.zero(transfers_.stream());
    }
    bool prepareWallBoundary(const WallKnot& knot,const SurfaceMesh& surface,std::string& error){
        if(!view_.gasBoundaryLayer.enabled){error.clear();return true;}
        if(!wallHostInput_.empty()&&(knot.faces.empty()||surface.area.empty())){
            error="boundaryLayer material wall layout mismatch";return false;}
        auto supply=[&](Real& destination,Real value){
            if(std::memcmp(&destination,&value,sizeof(Real))){destination=value;wallInputDirty_=true;}
        };
        for(std::size_t i=0;i<wallHostInput_.size();++i){auto& in=wallHostInput_[i];const auto& w=knot.faces[i];
            const auto& d=wallGeometry_->descriptors()[i];
            if(wallFaceArea_.empty()){error="boundaryLayer physical area storage missing";return false;}
            const Real area=wallFaceArea_[d.wallFace];
            supply(in.temperature,w.temperature);const Vec3 n={in.normal[0],in.normal[1],in.normal[2]};
            const Vec3 velocity=w.velocity-n*dot(w.velocity,n)+n*w.normalVelocity;
            supply(in.velocity[0],velocity.x);supply(in.velocity[1],velocity.y);supply(in.velocity[2],velocity.z);
            const Real ratio=surface.area[i]/area;
            for(int s=0;s<Ns;++s)supply(in.massFlux[s],(w.speciesRate[s]+w.poreRate[s])*ratio);
        }
        if(wallInputDirty_&&!uploadBuffer(wallInput_,wallHostInput_)){error=fault_.message;return false;}
        wallInputDirty_=false;error.clear();return true;
    }
    bool downloadWallOutputs(std::vector<ugkwp::gaswall::WallOutput<Real,Ns>>& output,std::vector<GasWallMatchingSample>& matching){
        std::vector<ugkwp::gaswall::WallInput<Real,Ns>> inputs;
        std::vector<ugkwp::GasBoundaryLayerExchange<Real>> exchange;
        std::vector<ugkwp::gaswall::WallResidentOutput<Real,Ns>> candidate;
        if(!enqueueRead(wallExchange_.data(),wallExchange_.size(),exchange)
            ||!enqueueRead(wallOutput_.data(),wallOutput_.size(),candidate)
            ||!enqueueRead(wallInput_.data(),wallInput_.size(),inputs)||!observeDiagnostics())return false;
        for(std::size_t f=0;f<exchange.size();++f)if(!exchange[f].ready){
            std::vector<int> status;std::vector<ugkwp::gaswall::WallStatus> modelStatus;
            wallModelFailed_=true;
            if(!enqueueRead(wallStatus_.data(),wallStatus_.size(),status)
                ||!enqueueRead(wallModelStatus_.data(),wallModelStatus_.size(),modelStatus)||!observeDiagnostics())return false;
            fault_.message="boundaryLayer rejected face "+std::to_string(wallFaces_[f])+" slot "+std::to_string(f)+" transport code "+std::to_string(status[f]);
            if(f<modelStatus.size())fault_.message+=" wall code "+std::to_string(int(modelStatus[f].code))+" node "+std::to_string(modelStatus[f].node)+" iteration "+std::to_string(modelStatus[f].iteration)+" residual "+(std::isfinite(modelStatus[f].residual)?std::to_string(modelStatus[f].residual):"NOT_AVAILABLE");
            return false;
        }
        std::vector<Real> flux;
        if(wallFluxProfile_.size()&&(!enqueueRead(wallFluxProfile_.data(),wallFluxProfile_.size(),flux)||!observeDiagnostics()))return false;
        output.resize(candidate.size());
        for(std::size_t i=0;i<candidate.size();++i){
            static_cast<ugkwp::gaswall::WallExchangeOutput<Real,Ns>&>(output[i])=candidate[i];
            const int n=view_.gasBoundaryLayerModel.config.nodes+1;
            for(int j=0;j<candidate[i].kFluxCount;++j){
                output[i].kFluxHeight[j]=flux[(3*i)*n+j];
                output[i].kTransportFlux[j]=flux[(3*i+1)*n+j];
                output[i].omegaDiffusiveFlux[j]=flux[(3*i+2)*n+j];
            }
        }
        matching.resize(inputs.size());for(std::size_t f=0;f<inputs.size();++f){matching[f].pressure=inputs[f].pressure;matching[f].mechanicalPressure=0;matching[f].state=inputs[f].matching;}return true;
    }
    bool uploadGeometry(const HostMesh& mesh,bool publish=true){
        const std::size_t nc=mesh.volumes.size(),nf=mesh.owner.size();
        const bool rebound=owner_V.values.size()!=nc||owner_faceOwner.values.size()!=nf
            ||owner_cellFaceId.values.size()!=mesh.cellFaces.size();
        if(rebound&&!transfers_.drain())return false;
        const bool boundaryRebound=!boundary_.view().faces||boundary_.view().count!=int(nf);



#define UGKWP_GAS_STATE_FIELD(Type,Name) if(meshField(#Name)&&!owner_##Name.allocate(view_.Name,count(#Name,nc,nf,mesh.cellFaces.size()),fault_,transfers_.stream()))return false;
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        if(view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy
            &&(!allocateSpecies(speciesBoundary_,view_.gasSpecies.boundaryMassFraction,Ns*nf)
                ||!allocateSpecies(compositionFixed_,view_.gasSpecies.compositionBoundaryFixed,nf)))return false;
        if(!prepareWallGeometry(mesh))return false;
        geometryVersion_=mesh.geometryVersion;volume_=mesh.volumes;std::vector<Real> cx(nc),cy(nc),cz(nc),length(nc),distance(nc,1);
        std::vector<int> start(nc),counts(nc);
        for(std::size_t c=0;c<nc;++c){cx[c]=mesh.cellCentres[c].x;cy[c]=mesh.cellCentres[c].y;cz[c]=mesh.cellCentres[c].z;length[c]=std::cbrt(mesh.volumes[c]);start[c]=mesh.cellFaceOffsets[c];counts[c]=mesh.cellFaceOffsets[c+1]-start[c];if(mesh.wallDistance.size()==nc)distance[c]=mesh.wallDistance[c];}
        std::vector<int> neighbours=mesh.neighbour;
        for(std::size_t f=0;f<nf;++f)if(mesh.boundaryKind[f]==BoundaryKind::Periodic){
            const int pair=mesh.periodicPartner[f];
            neighbours[f]=mesh.owner[pair];
        }
        if(!upload(view_.Cx,cx)||!upload(view_.Cy,cy)||!upload(view_.Cz,cz)||!upload(view_.V,mesh.volumes)||!upload(view_.cellLength,length)||!upload(view_.sstWallDistance,distance)
            ||!upload(view_.cellPlaneStart,start)||!upload(view_.cellPlaneCount,counts)||!upload(view_.cellFaceId,mesh.cellFaces)||!upload(view_.faceOwner,mesh.owner)||!upload(view_.faceNeighbour,neighbours)||!upload(view_.facePeriodicPair,mesh.periodicPartner))return false;
        std::vector<Real> sx(nf),sy(nf),sz(nf),area(nf),fx(nf),fy(nf),fz(nf),dx(nf),dy(nf),dz(nf),delta(nf),weight(nf,.5),Y(boundaryRebound?Ns*nf:0);
        std::vector<int> fixedY(boundaryRebound?nf:0),kMode(nf),omegaMode(nf);
        std::vector<Real> kBoundary(nf),omegaBoundary(nf);
        for(std::size_t f=0;f<nf;++f){auto a=mesh.areaVectors[f];sx[f]=a.x;sy[f]=a.y;sz[f]=a.z;area[f]=mag(a);fx[f]=mesh.faceCentres[f].x;fy[f]=mesh.faceCentres[f].y;fz[f]=mesh.faceCentres[f].z;
            const int o=mesh.owner[f],n=mesh.neighbour[f];Vec3 chord=mesh.faceCentres[f]-mesh.cellCentres[o];
            if(n>=0)chord=mesh.cellCentres[n]-mesh.cellCentres[o];
            const auto b=mesh.boundaryKind[f];
            if(b==BoundaryKind::Periodic){const int pair=mesh.periodicPartner[f];const Vec3 shift=mesh.faceCentres[f]-mesh.faceCentres[pair];dx[f]=shift.x;dy[f]=shift.y;dz[f]=shift.z;chord=mesh.cellCentres[mesh.owner[pair]]+shift-mesh.cellCentres[o];}
            delta[f]=1/maxValue(absValue(dot(chord,a/area[f])),.05*mag(chord));
            if(n>=0||b==BoundaryKind::Periodic){
                const Real own=absValue(dot(mesh.faceCentres[f]-mesh.cellCentres[o],a/area[f]));
                const Vec3 mappedNeighbour=mesh.cellCentres[o]+chord;
                const Real other=absValue(dot(mappedNeighbour-mesh.faceCentres[f],a/area[f]));
                weight[f]=(own+other)>0?other/(own+other):.5;
            }
            const auto& w=mesh.boundaryPrimitive[f];
            const bool inlet=mesh.gasBoundary?ugkp::boundary::prescribesInflow(mesh.gasBoundary->faces[f]):b==BoundaryKind::Inlet;
            if(boundaryRebound)fixedY[f]=mesh.gasBoundary&&mesh.gasBoundary->faces[f].compositionMode>=0?mesh.gasBoundary->faces[f].compositionMode:inlet;
            if(boundaryRebound)for(int s=0;s<Ns;++s)Y[s*nf+f]=w.Y[s];
            kMode[f]=omegaMode[f]=inlet?1:0;
            if(mesh.boundarySst.size()==nf){kBoundary[f]=mesh.boundarySst[f].k;omegaBoundary[f]=mesh.boundarySst[f].omega;}
        }
#define PUT(Name,Source) if(!upload(view_.Name,Source))return false
        PUT(Sfx,sx);PUT(Sfy,sy);PUT(Sfz,sz);PUT(magSf,area);PUT(faceCx,fx);PUT(faceCy,fy);PUT(faceCz,fz);PUT(facePeriodicDx,dx);PUT(facePeriodicDy,dy);PUT(facePeriodicDz,dz);PUT(deltaCoeffs,delta);PUT(faceWeight,weight);
        PUT(sstBoundaryK,kBoundary);PUT(sstBoundaryOmega,omegaBoundary);PUT(sstBoundaryKMode,kMode);PUT(sstBoundaryOmegaMode,omegaMode);
#undef PUT
        if(boundaryRebound) {
        ugkp::boundary::Definition definition=mesh.gasBoundary?*mesh.gasBoundary:ugkp::boundary::Definition{};
        if(definition.faces.empty()) {
            definition.faces.resize(nf);
            for(std::size_t f=0;f<nf;++f) {
                const auto& w=mesh.boundaryPrimitive[f];
                definition.faces[f]=ugkp::boundary::prescribed(mesh.boundaryKind[f],
                    fixedTemperature(mesh.thermalBoundary.empty()?nullptr:mesh.thermalBoundary.data(),int(f)),
                    w.rho,w.velocity.x,w.velocity.y,w.velocity.z,w.pressure,w.temperature);
            }
        }
            std::string error;
            if(!boundary_.configure(definition,transfers_.stream(),error)){fault_.message=error;return false;}
            view_.gasBoundaryProgram=boundary_.view();
        if(!ugkp::boundary::publish(definition,view_,[&](auto* dst,const auto& values){return upload(dst,values);}))return false;
        }
        if(view_.gasBoundaryLayer.enabled)wallFaceArea_=area;
        if(boundaryRebound&&view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy
            &&(!upload(view_.gasSpecies.boundaryMassFraction,Y)||!upload(view_.gasSpecies.compositionBoundaryFixed,fixedY)))return false;
        return !publish||!rebound||refreshView();
    }
    bool uploadState(const HostState& state,std::string& error){
        const std::size_t n=state.gas.size();
        if(!state.gasVoidFraction.empty()&&state.gasVoidFraction.size()!=n){error="gas phase fraction count differs from gas cells";return false;}
        if(!state.gasTemperature.empty()&&state.gasTemperature.size()!=n){error="missing supplied gas temperature backing";return false;}if(view_.sstConfigured&&n&&state.sst.empty()){error="missing supplied SST inventory";return false;}std::vector<Real> rho(n),px(n),py(n),pz(n),energy(n),species(Ns*n),k(n),omega(n);
        for(std::size_t c=0;c<n;++c){const auto& q=state.gas[c];const Real v=gasPhaseVolume(state,c);
            for(int s=0;s<Ns;++s)species[s*n+c]=q.species[s]/v;
            rho[c]=q.mass/v;px[c]=q.momentum.x/v;py[c]=q.momentum.y/v;pz[c]=q.momentum.z/v;energy[c]=q.energy/v;
            if(view_.sstConfigured){k[c]=state.sst[c].rhoK/v;omega[c]=state.sst[c].rhoOmega/v;}
        }
        if(!upload(view_.rho,rho)||!upload(view_.rhoUx,px)||!upload(view_.rhoUy,py)||!upload(view_.rhoUz,pz)||!upload(view_.rhoE,energy)||(!state.gasTemperature.empty()&&!upload(view_.Tgas,state.gasTemperature))||(view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy&&!upload(view_.gasSpecies.rho,species))||!upload(view_.rhoK,k)||!upload(view_.rhoOmega,omega)||!clearTrialStatus()){error=fault_.message;return false;}
        error.clear();return true;
    }
    bool downloadState(HostState& state,std::string& error){
        std::vector<Real> rho,px,py,pz,energy,species,k,omega,fraction;const std::size_t n=view_.nCells;
        if(view_.gasVolumeFraction&&!enqueueRead(view_.gasVolumeFraction,n,fraction)){error=fault_.message;return false;}
        if(!enqueueRead(view_.rho,n,rho)||!enqueueRead(view_.rhoUx,n,px)||!enqueueRead(view_.rhoUy,n,py)||!enqueueRead(view_.rhoUz,n,pz)||!enqueueRead(view_.rhoE,n,energy)||!readTemperature(n)||(view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy&&!enqueueRead(view_.gasSpecies.rho,Ns*n,species))){error=fault_.message;return false;}
        if(view_.sstConfigured&&(!enqueueRead(view_.rhoK,n,k)||!enqueueRead(view_.rhoOmega,n,omega))){error=fault_.message;return false;}
        if(!observeDiagnostics()){error=fault_.message;return false;}
        if(view_.gasVolumeFraction)state.gasVoidFraction=std::move(fraction);
        auto candidate=state.gas;candidate.resize(n);
        for(std::size_t c=0;c<n;++c){auto& q=candidate[c];const Real v=gasPhaseVolume(state,c);q.mass=rho[c]*v;q.momentum={px[c]*v,py[c]*v,pz[c]*v};q.energy=energy[c]*v;
            for(int s=0;s<Ns;++s)q.species[s]=view_.gasSpecies.mode==ugkwp::GasMode::SingleLegacy?(s==activeSpecies_?q.mass:0):species[s*n+c]*v;
        }
        auto sst=state.sst;
        if(view_.sstConfigured){
            sst.resize(n);for(std::size_t c=0;c<n;++c){sst[c]={k[c]*gasPhaseVolume(state,c),omega[c]*gasPhaseVolume(state,c)};
            }
        }
        state.gas=std::move(candidate);state.sst=std::move(sst);state.gasTemperature.swap(temperatureStaging_);error.clear();return true;
    }
};
}
#endif
