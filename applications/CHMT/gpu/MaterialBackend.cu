#include "ablation/SurfaceDonorAllocation.H"
#include "gpu/MaterialBackend.H"
#include "gpu/HostTransferQueue.H"
#include "gpu/MaterialTransportKernels.cuh"
#include "gpu/MaterialSourceKernels.cuh"
#include "gpu/MaterialSurfaceKernels.cuh"
#include "materials/MaterialInventoryMath.H"
#include "materials/CpuMaterialDriver.H"
#include "../../../common/GpuDiagnosticsOwner.cuh"
#include "../../../common/gasTransport/GasBoundaryLayerWorkspace.H"
#include <algorithm>
#include <memory>
namespace chmt {
namespace {
using namespace materialSourceDetail;
// All conduction kernels use the accepted material owner's stream and inventory.
// Candidate energy/EOS results are published only after every cell succeeds.
__global__ void prepareConductionKernel(int count,const SolidQ* initial,const Real* initialVolumes,
    const SolidQ* current,const Real* volumes,const Real* nativeVolumes,const PhysicsConfig* physics,
    MaterialThermalInput* input,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    status[c]={};MaterialThermalInput value;bool numerical=true;
    const auto code=makeMaterialCaloric(current[c],*physics,value.caloric,numerical);
    if(code!=MaterialCaloricCode::Success){status[c]={1,int(code),c,numerical};return;}
    MaterialPrimitive primitive;
    if(!recoverMaterial<false>(initial[c],initialVolumes[c],*physics,primitive,nullptr,-1,&numerical)){
        status[c]={2,0,c,numerical};return;
    }
    value.inventory=current[c];value.anchor=primitive.temperature;value.target=current[c].energy;
    value.nativeVolume=nativeVolumes[c];value.materialVolume=volumes[c];input[c]=value;
}
__global__ void conductionCoefficientsKernel(int count,const MaterialThermalInput* input,
    const Real* temperature,Real dt,const PhysicsConfig* physics,MaterialThermalCoefficients* output){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count)output[c]=materialThermalCoefficients(input[c],temperature[c],dt,*physics);
}
__global__ void finishConductionKernel(int count,const SolidQ* current,const Real* volumes,
    const MaterialThermalInput* input,const Real* temperature,const PhysicsConfig* physics,
    SolidQ* next,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    status[c]={};SolidQ value=current[c];value.energy=input[c].caloric.energy<false>(temperature[c]);
    MaterialPrimitive primitive;bool numerical=false;
    if(!recoverMaterial<false>(value,volumes[c],*physics,primitive,nullptr,-1,&numerical)){
        status[c]={3,0,c,numerical};return;
    }
    value.porosity=primitive.porosity;next[c]=value;
}
__global__ void selectConductionFailure(const MaterialStepResult* status,int count,MaterialStepResult* result){
    MaterialStepResult value{};
    for(int c=threadIdx.x;c<count;c+=blockDim.x)if(status[c].phase){value=status[c];break;}
    reduceMaterialFailure(value,result,false);
}

}
struct MaterialBackend {
    CudaFault fault;
    DeviceStream stream;
    HostTransferQueue transfers{fault};
    ugkp::diagnostics::GpuOwner diagnostics;
    PhysicsConfig physics{};
    std::uint64_t sequence=0,step=0;Real time=0;
    Buffer<PhysicsConfig> devicePhysics;
    Buffer<SolidQ> current,next,initial,gasChange,candidateGasChange,donor;
    Buffer<Real> initialVolumes,conductionVolumes,conductionTemperature;
    Buffer<MaterialThermalInput> conductionInput;
    Buffer<MaterialThermalCoefficients> conductionCoefficients;
    Buffer<Real> volumes,nextVolumes,sweeps,gradientSensitivity,stabilityRate,lossLeft,lossRight,cellDt;
    Buffer<Vec3> centres,areas,faceCentres,pressureGradient,poreVelocity;
    Buffer<int> owner,neighbour,periodic,cellOffsets,cellFaces,cellSigns;
    Buffer<BoundaryKind> boundary;
    Buffer<int> diffusionOffsets,diffusionFaces,diffusionSides;
    Buffer<MaterialPrimitive> primitives;
    Buffer<SolidQ> faceFlux,rates;
    Buffer<int> cellErrors,faceErrors;
    Buffer<MaterialTransportDeviceResult> transportResult;
    Buffer<MaterialStepResult> sourceStatus,stepResult;
    Buffer<ExchangePacket> packets;
    Buffer<int> packetOffsets,packetIndices,radiationOffsets,radiationFaces;
    Buffer<Real> radiation;
    Buffer<unsigned> radiationReceiver;
    Buffer<SurfacePhysicsInput> surfaceInput;
    Buffer<int> surfaceCells;
    Buffer<unsigned char> surfaceActive,wallWorkspace;
    Buffer<SurfacePhysicsResult> surfacePhysics;
    Buffer<WallFaceSample> surfaceWall;
    Buffer<MaterialSurfaceStatus> surfaceStatus;
    Buffer<Real> quadratureDistance,quadratureWeight,mechanismBasis;
    Buffer<ugkwp::GasReactionData<Real>> mechanismReactions;
    Buffer<ugkwp::GasStoichTerm<Real>> mechanismReactants,mechanismProducts;
    Buffer<ugkwp::GasColliderEfficiency<Real>> mechanismEfficiencies;
    ugkwp::GasMechanismView<Real,Ns> deviceMechanism{};
    bool mechanismReady=false,meshReady=false;
    std::uint64_t geometryVersion=0,topologyHash=0;
    int cells=0,faces=0;
    ~MaterialBackend(){stream.sync();}
    template<class T> bool allocate(Buffer<T>& buffer,std::size_t size){return buffer.resize(size,fault);}
    template<class T> bool uploadRaw(Buffer<T>& buffer,const T* data,std::size_t size){
        return allocate(buffer,size)&&transfers.upload(buffer.data(),data,size);
    }
    template<class T> bool upload(Buffer<T>& buffer,const std::vector<T>& data){return uploadRaw(buffer,data.data(),data.size());}
    template<class T> bool read(const Buffer<T>& buffer,std::vector<T>& data){
        data.resize(buffer.size());return transfers.download(data.data(),buffer.data(),buffer.size());
    }
    bool failure(std::string& error){transfers.drain();if(!fault.message.empty())error=fault.message;return false;}
    bool finish(std::string& error){
        ugkp::diagnostics::DeviceContext snapshot{};
        if(!transfers.download(&snapshot,diagnostics.device(),1)||!transfers.finish())return failure(error);
        diagnostics.consumeCompletedSnapshot(snapshot);
        if(diagnostics.dispatchPending()){error="fatal material device diagnostic";return false;}
        return true;
    }
    void operation(Real t){
        time=t;diagnostics.setOperationContext(sequence,step,time);
        ugkp::diagnostics::beginOperationKernel<<<1,1,0,stream.get()>>>(diagnostics.device(),sequence,step,time);
    }
    MaterialMeshView mesh()const{
        MaterialMeshView m;m.cells=cells;m.faces=faces;m.volumes=volumes.data();m.cellCentres=centres.data();
        m.owner=owner.data();m.neighbour=neighbour.data();m.areaVectors=areas.data();m.faceCentres=faceCentres.data();
        m.boundaryKind=boundary.data();m.periodicPartner=periodic.data();m.cellFaceOffsets=cellOffsets.data();
        m.cellFaces=cellFaces.data();m.cellFaceSigns=cellSigns.data();m.diffusionOffsets=diffusionOffsets.data();
        m.diffusionFaces=diffusionFaces.data();m.diffusionSides=diffusionSides.data();return m;
    }
    MaterialTransportWorkspace workspace()const{
        MaterialTransportWorkspace w;w.primitive=primitives.data();w.pressureGradient=pressureGradient.data();
        w.poreVelocity=poreVelocity.data();w.gradientSensitivity=gradientSensitivity.data();w.stabilityRate=stabilityRate.data();
        w.faceDiffusionLossLeft=lossLeft.data();w.faceDiffusionLossRight=lossRight.data();w.cellDtLimit=cellDt.data();
        w.faceFlux=faceFlux.data();w.rates=rates.data();w.cellErrors=cellErrors.data();w.faceErrors=faceErrors.data();return w;
    }
    bool configureMesh(const HostMesh& m,std::string& error){
        if(current.size()==0){cells=faces=0;return true;}
        if(!validateMaterialTransportMesh(m,current.size(),error))return false;
        if(meshReady&&geometryVersion==m.geometryVersion&&topologyHash==m.topologyHash)return true;
        cells=int(m.volumes.size());faces=int(m.owner.size());
        if(!upload(volumes,m.volumes)||!upload(centres,m.cellCentres)||!upload(areas,m.areaVectors)
            ||!upload(faceCentres,m.faceCentres))return failure(error);
        if(!meshReady||topologyHash!=m.topologyHash){
            if(!upload(owner,m.owner)||!upload(neighbour,m.neighbour)||!upload(periodic,m.periodicPartner)
                ||!upload(boundary,m.boundaryKind)||!upload(cellOffsets,m.cellFaceOffsets)
                ||!upload(cellFaces,m.cellFaces)||!upload(cellSigns,m.cellFaceSigns))return failure(error);
            std::vector<std::vector<std::pair<int,int>>> rows(cells);
            for(int f=0;f<faces;++f){
                if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
                int right=m.neighbour[f];
                if(m.boundaryKind[f]==BoundaryKind::Periodic){
                    if(f>m.periodicPartner[f])continue;
                    right=m.owner[m.periodicPartner[f]];
                }
                if(right<0)continue;
                rows[m.owner[f]].push_back({f,0});rows[right].push_back({f,1});
            }
            std::vector<int> offsets(cells+1),indices,sides;
            for(int c=0;c<cells;++c){for(const auto& pair:rows[c]){indices.push_back(pair.first);sides.push_back(pair.second);}offsets[c+1]=int(indices.size());}
            if(!upload(diffusionOffsets,offsets)||!upload(diffusionFaces,indices)||!upload(diffusionSides,sides))return failure(error);
        }
        if(!allocate(primitives,cells)||!allocate(pressureGradient,cells)||!allocate(poreVelocity,cells)
            ||!allocate(gradientSensitivity,cells)||!allocate(stabilityRate,cells)||!allocate(cellDt,cells)
            ||!allocate(lossLeft,faces)||!allocate(lossRight,faces)||!allocate(faceFlux,faces)||!allocate(rates,cells)
            ||!allocate(cellErrors,cells)||!allocate(faceErrors,faces)||!allocate(transportResult,1))return failure(error);
        geometryVersion=m.geometryVersion;topologyHash=m.topologyHash;meshReady=true;return true;
    }
    bool configureMechanism(const ugkwp::GasMechanismView<Real,Ns>& m,std::string& error){
        if(mechanismReady&&deviceMechanism.mechanismHash==m.mechanismHash
            &&deviceMechanism.speciesOrderHash==m.speciesOrderHash)return true;
        if(!uploadRaw(mechanismReactions,m.reactions,m.reactionCount)
            ||!uploadRaw(mechanismReactants,m.reactants,m.reactantTermCount)
            ||!uploadRaw(mechanismProducts,m.products,m.productTermCount)
            ||!uploadRaw(mechanismEfficiencies,m.efficiencies,m.efficiencyCount)
            ||!uploadRaw(mechanismBasis,m.stoichiometricBasis,Ns*m.independentRank))return failure(error);
        deviceMechanism=m;deviceMechanism.reactions=mechanismReactions.data();deviceMechanism.reactants=mechanismReactants.data();
        deviceMechanism.products=mechanismProducts.data();deviceMechanism.efficiencies=mechanismEfficiencies.data();
        deviceMechanism.stoichiometricBasis=mechanismBasis.data();mechanismReady=true;return true;
    }
};

MaterialBackend* createMaterialBackend(std::string& error){
    std::unique_ptr<MaterialBackend> b(new MaterialBackend);
    if(!b->stream.create(b->fault)){error=b->fault.message;return nullptr;}
    b->transfers.bindStream(b->stream.get());
    if(!b->fault.check(b->diagnostics.initialise(b->stream.get()),"initialize material diagnostics")){
        error=b->fault.message;return nullptr;
    }
    error.clear();return b.release();
}
void destroyMaterialBackend(MaterialBackend* b)noexcept{delete b;}
void setMaterialDiagnosticsSink(MaterialBackend& b,const ugkp::diagnostics::HostSink& sink){b.diagnostics.setSink(sink);}
bool beginMaterialWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,std::string& error){
    b.physics=physics;
    auto devicePhysics=physics;devicePhysics.diagnostics=b.diagnostics.device();
    devicePhysics.tolerances.diagnostics=b.diagnostics.device();
    b.sequence=state.commitSequence+1;b.step=state.acceptedSteps;b.time=state.time;
    if(!b.upload(b.current,state.solid)||!b.initial.copy(b.current,b.fault,b.stream.get())
        ||!b.allocate(b.next,state.solid.size())
        ||!b.allocate(b.gasChange,state.solid.size())||!b.gasChange.zero(b.stream.get())
        ||!b.allocate(b.candidateGasChange,state.solid.size())
        ||!b.allocate(b.sourceStatus,state.solid.size())||!b.allocate(b.stepResult,1)
        ||!b.uploadRaw(b.devicePhysics,&devicePhysics,1)||!b.upload(b.initialVolumes,state.solidMesh.volumes)
        ||!b.upload(b.nextVolumes,state.solidMesh.volumes))return b.failure(error);
    if(!b.configureMesh(state.solidMesh,error))return false;
    b.operation(state.time);
    return b.finish(error);
}
bool prepareResidentMaterialConduction(MaterialBackend& b,const HostMesh& mesh,
    const std::vector<Real>& nativeVolumes,std::vector<MaterialThermalInput>& input,
    std::string& error,bool& numerical){
    numerical=true;
    if(!b.configureMesh(mesh,error))return false;
    if(!b.upload(b.conductionVolumes,nativeVolumes)||!b.allocate(b.conductionInput,b.current.size())
        ||!b.allocate(b.conductionCoefficients,b.current.size()))return b.failure(error);
    b.operation(b.time);
    if(b.cells)prepareConductionKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,
        b.initial.data(),b.initialVolumes.data(),b.current.data(),b.volumes.data(),b.conductionVolumes.data(),
        b.devicePhysics.data(),b.conductionInput.data(),b.sourceStatus.data());
    selectConductionFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
    MaterialStepResult result;
    if(!b.fault.check(cudaGetLastError(),"prepare resident material conduction")
        ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.phase){numerical=result.numerical;error=result.phase==1
        ?materialCaloricError(MaterialCaloricCode(result.code)):"CPU conduction old EOS recovery failed";return false;}
    if(!b.read(b.conductionInput,input)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool evaluateResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    Real dt,std::vector<MaterialThermalCoefficients>& coefficients,std::string& error){
    if(!b.upload(b.conductionTemperature,temperature))return b.failure(error);
    b.operation(b.time);
    if(b.cells)conductionCoefficientsKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,
        b.conductionInput.data(),b.conductionTemperature.data(),dt,b.devicePhysics.data(),b.conductionCoefficients.data());
    if(!b.fault.check(cudaGetLastError(),"evaluate resident material conduction")
        ||!b.read(b.conductionCoefficients,coefficients)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool finishResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    std::vector<SolidQ>& hostMirror,std::string& error,bool& numerical){
    numerical=true;
    if(!b.upload(b.conductionTemperature,temperature))return b.failure(error);
    b.operation(b.time);
    if(b.cells)finishConductionKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,b.current.data(),
        b.volumes.data(),b.conductionInput.data(),b.conductionTemperature.data(),b.devicePhysics.data(),
        b.next.data(),b.sourceStatus.data());
    selectConductionFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
    MaterialStepResult result;
    if(!b.fault.check(cudaGetLastError(),"finish resident material conduction")
        ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.phase){numerical=result.numerical;error="implicit material endpoint EOS invalid";return false;}
    std::vector<SolidQ> candidate;
    if(!b.read(b.next,candidate)||!b.finish(error))return b.failure(error);
    b.current.swap(b.next);hostMirror.swap(candidate);error.clear();return true;
}
bool evaluateResidentMaterialTransport(MaterialBackend& b,const HostMesh& mesh,
    const std::vector<Real>& swept,Real interval,Real cfl,Real& dtLimit,std::string& error,bool& numerical){
    numerical=true;
    if(!(interval>0)||!finite(interval)||!finite(cfl)||!(cfl>0)||(!swept.empty()&&swept.size()!=mesh.owner.size())){
        error="invalid material transport interval/control";return false;
    }
    if(b.current.size()==0){dtLimit=std::numeric_limits<Real>::max();error.clear();return true;}
    if(!b.configureMesh(mesh,error))return false;
    if(!b.upload(b.sweeps,swept))return b.failure(error);
    b.operation(b.time);
    if(!b.fault.check(launchMaterialTransport(b.mesh(),b.current.data(),b.devicePhysics.data(),
        swept.empty()?nullptr:b.sweeps.data(),interval,cfl,b.workspace(),b.transportResult.data(),b.stream.get()),
        "launch resident material transport"))return b.failure(error);
    MaterialTransportDeviceResult result;
    if(!b.transfers.download(&result,b.transportResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.errorCode){numerical=result.numerical;error=materialTransport::errorMessage(MaterialTransportError(result.errorCode));
        if(result.errorCode==int(MaterialTransportError::CellEos))error+=std::to_string(result.entity);
        return false;}
    numerical=result.numerical;dtLimit=result.dtLimit;error.clear();return true;
}
bool advanceResidentMaterialSources(MaterialBackend& b,const HostMesh& endpointMesh,
    const SurfaceMesh& surface,const std::vector<ExchangePacket>& packets,
    const std::vector<Real>& radiation,const std::vector<unsigned>& receivers,
    Real dt,const CpuMaterialControls& controls,bool finalStep,std::vector<SolidQ>& shadow,
    std::vector<SolidQ>& cumulative,std::string& error,bool& numerical){
    numerical=true;
    const int count=int(b.current.size());
    if(controls.retainDonorHistory&&!b.allocate(b.donor,count))return b.failure(error);
    std::vector<std::vector<int>> packetRows(count),radiationRows(count);
    for(std::size_t i=0;i<packets.size();++i)if(requiredConsumers(packets[i].kind)&ConsumeSolid){
        const int cell=packets[i].solidCell;
        if(cell<0||cell>=count){error="material packet owner invalid";return false;}
        packetRows[cell].push_back(int(i));
    }
    for(std::size_t f=0;f<radiation.size();++f){
        if(receivers[f]!=ConsumeSolid)continue;
        const int cell=surface.solidCell[f];
        if(cell<0||cell>=count){error="radiation has no material owner";return false;}
        radiationRows[cell].push_back(int(f));
    }
    std::vector<int> packetOffsets(count+1),packetIndices,radiationOffsets(count+1),radiationFaces;
    for(int c=0;c<count;++c){
        packetIndices.insert(packetIndices.end(),packetRows[c].begin(),packetRows[c].end());
        packetOffsets[c+1]=int(packetIndices.size());
        radiationFaces.insert(radiationFaces.end(),radiationRows[c].begin(),radiationRows[c].end());
        radiationOffsets[c+1]=int(radiationFaces.size());
    }
    if(!b.upload(b.packets,packets)||!b.upload(b.packetOffsets,packetOffsets)||!b.upload(b.packetIndices,packetIndices)
        ||!b.upload(b.radiationOffsets,radiationOffsets)||!b.upload(b.radiationFaces,radiationFaces)
        ||!b.upload(b.radiation,radiation)||!b.upload(b.radiationReceiver,receivers)
        ||!b.upload(b.nextVolumes,endpointMesh.volumes))return b.failure(error);
    if(!b.fault.check(cudaMemsetAsync(b.stepResult.data(),0,sizeof(MaterialStepResult),b.stream.get()),
        "clear material source result"))return b.failure(error);
    if(count){
        applyMaterialSourcesKernel<<<(count+127)/128,128,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
            b.initial.data(),b.gasChange.data(),b.candidateGasChange.data(),b.packets.data(),b.packetOffsets.data(),b.packetIndices.data(),
            b.radiationOffsets.data(),b.radiationFaces.data(),radiation.empty()?nullptr:b.radiation.data(),
            b.radiationReceiver.data(),b.nextVolumes.data(),b.devicePhysics.data(),dt,finalStep,
            b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
    }
    selectSourceFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),count,b.stepResult.data());
    if(!b.fault.check(cudaGetLastError(),"launch resident material local sources"))return b.failure(error);
    MaterialStepResult result;std::vector<SolidQ> candidateShadow,candidateDonor;
    if(!b.transfers.download(&result,b.stepResult.data(),1)||!b.read(b.next,candidateShadow)||(controls.retainDonorHistory&&!b.read(b.donor,candidateDonor))
        ||!b.finish(error))return b.failure(error);
    if(result.phase){
        numerical=result.numerical;
        if(result.phase==1)error=materialReactionMessage(MaterialReactionCode(result.code));
        else if(result.phase==2)error=materialInventoryMessage(MaterialInventoryCode(result.code));
        else error="material endpoint inventory/EOS invalid";
        return false;
    }
    shadow=std::move(candidateShadow);cumulative=std::move(candidateDonor);
    error.clear();return true;
}
void commitResidentMaterialStep(MaterialBackend& b,Real time){
    b.current.swap(b.next);b.volumes.swap(b.nextVolumes);b.gasChange.swap(b.candidateGasChange);b.time=time;
}

bool evaluateResidentMaterialSurfaces(MaterialBackend& b,const HostState& h,
    const std::vector<MaterialSurfaceRequest>& requests,Real dt,bool useRates,bool candidate,
    std::vector<CpuSurfaceResult>& output,std::string& error,bool& numerical,
    const std::vector<Real>* sideVolume,bool* terminalWallFailure){
    numerical=true;if(terminalWallFailure)*terminalWallFailure=false;
    const auto& s=h.surface;const std::size_t nf=s.area.size();
    const auto& p=b.physics;
    const bool boundaryLayer=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer;
    const std::size_t count=nf*requests.size();
    std::vector<Real> donorShares;
    if(!surfaceGasDonorShares(h,donorShares,error))return false;
    std::vector<SurfacePhysicsInput> inputs(count);
    std::vector<int> solidCells(count,-1),quadratureOffsets(count);
    std::vector<unsigned char> active(count);
    std::vector<Real> distances,weights;
    for(std::size_t request=0;request<requests.size();++request){
        const auto& bulk=*requests[request].bulk;
        const auto& gradients=*requests[request].gradients;
        const auto* contexts=requests[request].contexts;
        const auto* accepted=requests[request].acceptedRates;
        if(accepted&&!accepted->empty()&&accepted->size()!=nf){
            numerical=false;error="accepted surface stage rate layout mismatch";return false;}
        if(!finite(dt)||dt<=0||bulk.size()!=nf||(!gradients.empty()&&gradients.size()!=nf)
            ||s.gasFace.size()!=nf||s.normal.size()!=nf||s.gasDistance.size()!=nf
            ||(!s.solidCell.empty()&&s.solidCell.size()!=nf)||(!s.baseVelocity.empty()&&s.baseVelocity.size()!=nf)){
            numerical=bulk.size()>=nf&&(gradients.empty()||gradients.size()>=nf)
                &&s.gasFace.size()>=nf&&s.normal.size()>=nf&&s.gasDistance.size()>=nf
                &&(s.solidCell.empty()||s.solidCell.size()>=nf)&&(s.baseVelocity.empty()||s.baseVelocity.size()>=nf);
            error="CPU surface input layout/interval mismatch";return false;
        }
        if(!s.gasMapOffsets.empty()||!s.gasMapFaces.empty()||!s.gasMapWeights.empty()){
            error="CPU surface requires the supported conformal interface map";return false;
        }
        if(boundaryLayer&&(!contexts||contexts->size()!=nf)){
            numerical=contexts&&contexts->size()>=nf;
            error="CPU boundaryLayer requires current matching snapshot and certified stage geometry";return false;
        }
        if(contexts)for(std::size_t f=0;f<nf;++f)if(s.gasFace[f]>=0){
            if(!b.configureMechanism((*contexts)[f].mechanism,error))return false;
            break;
        }
        for(std::size_t f=0;f<nf;++f){
            const std::size_t index=request*nf+f;
            const int gf=s.gasFace[f];if(gf<0)continue;active[index]=1;
            if(std::size_t(gf)>=h.gasMesh.owner.size()||std::size_t(gf)>=h.gasMesh.areaVectors.size()
                ||std::size_t(gf)>=h.gasMesh.faceIds.size()){error="CPU surface gas face outside real mesh";return false;}
            const int gc=h.gasMesh.owner[gf];
            if(gc<0||std::size_t(gc)>=h.gasMesh.volumes.size()){error="CPU surface gas owner outside real mesh";return false;}
            auto& in=inputs[index];in.bulk=bulk[f];if(!gradients.empty())in.gradient=gradients[f];
            in.gasInventory=conservativeGas(bulk[f],gasPhaseVolume(h,gc)*donorShares[f],p);
            if(accepted&&!accepted->empty()){
                in.useAcceptedFlux=true;const auto& rates=(*accepted)[f];
                for(int s=0;s<Ns;++s){in.acceptedGasSpecies[s]=rates.speciesRate[s];
                    in.acceptedPoreSpecies[s]=rates.poreRate[s];in.acceptedPoreSweep[s]=rates.poreSweepRate[s];}
                for(int c=0;c<Nc;++c)in.acceptedCondensed[c]=rates.condensedRate[c];
            }
            if(contexts){
                in.wallContext=(*contexts)[f];in.bulk.pressure=in.wallContext.matchingPressure;
                in.wallContext.mechanism=b.deviceMechanism;in.wallContext.workspace=nullptr;in.wallContext.failureStatus=nullptr;
                const auto& q=(*contexts)[f].quadrature;
                quadratureOffsets[index]=int(distances.size());
                for(int j=0;j<q.count;++j){distances.push_back(q.distance[j]);weights.push_back(q.volumeWeight[j]);}
            }
            in.area=s.area[f];in.normal=s.normal[f];in.gasArea=mag(h.gasMesh.areaVectors[gf]);
            in.gasNormal=-h.gasMesh.areaVectors[gf]/in.gasArea;in.gasDistance=s.gasDistance[f];in.dt=dt;
            in.aux.pressure=in.bulk.pressure;in.baseVelocity=s.baseVelocity.empty()?Vec3{}:s.baseVelocity[f];
            const int sc=s.solidCell.empty()?-1:s.solidCell[f];solidCells[index]=sc;in.hasSolid=sc>=0;
            if(in.hasSolid){
                if(std::size_t(sc)>=b.current.size()||std::size_t(sc)>=h.solidMesh.volumes.size()
                    ||s.solidFace.size()!=nf||s.solidDistance.size()!=nf){
                    numerical=s.solidFace.size()>=nf&&s.solidDistance.size()>=nf;
                    error="CPU interface material owner invalid";return false;
                }
                const int sf=s.solidFace[f];
                if(sf<0||std::size_t(sf)>=h.solidMesh.areaVectors.size()){error="CPU interface solid face invalid";return false;}
                in.solidDistance=s.solidDistance[f];in.solidArea=mag(h.solidMesh.areaVectors[sf]);
                in.solidNormal=h.solidMesh.areaVectors[sf]/in.solidArea;
            }
            if(sideVolume){
                if(sideVolume->size()!=nf){numerical=sideVolume->size()>=nf;error="surface side sweep layout mismatch";return false;}
                in.sideMeshVolumeRate=(*sideVolume)[f];
            }
        }
    }
    if(!b.upload(b.quadratureDistance,distances)||!b.upload(b.quadratureWeight,weights))return b.failure(error);
    for(std::size_t f=0;f<count;++f){
        auto& q=inputs[f].wallContext.quadrature;
        q.distance=q.count?b.quadratureDistance.data()+quadratureOffsets[f]:nullptr;
        q.volumeWeight=q.count?b.quadratureWeight.data()+quadratureOffsets[f]:nullptr;
    }
    MaterialSurfaceView view;view.count=int(count);view.workers=int(count);
    if(boundaryLayer&&p.wallModel.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport&&count){
        int device=0,sms=0;
        if(!b.fault.check(cudaGetDevice(&device),"query material wall device")
            ||!b.fault.check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"query material wall multiprocessors"))return b.failure(error);
        view.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(p.wallModel.nodes);
        view.workspaceStride=ugkwp::gasBoundaryLayerWorkspaceBytes<Real,Ns>(p.wallModel.nodes);
        const auto sizing=ugkwp::resolveBoundaryLayerWorkspace(int(count),p.wallWorkspaceSlots,sms,0,view.workspaceStride);
        if(!sizing.slots){error="boundaryLayer workspace has no compiled capacity or its byte count is not representable";return false;}
        if(!b.allocate(b.wallWorkspace,sizing.bytes))return b.failure(error);
        view.workers=sizing.slots;view.wallWorkspaces=b.wallWorkspace.data();
    }
    if(!b.upload(b.surfaceInput,inputs)||!b.upload(b.surfaceCells,solidCells)||!b.upload(b.surfaceActive,active)
        ||!b.allocate(b.surfacePhysics,count)||!b.allocate(b.surfaceWall,count)||!b.allocate(b.surfaceStatus,count))return b.failure(error);
    view.inputs=b.surfaceInput.data();view.solidCells=b.surfaceCells.data();view.solid=candidate?b.next.data():b.current.data();
    view.volumes=candidate?b.nextVolumes.data():b.volumes.data();view.materialRate=useRates?b.rates.data():nullptr;
    view.active=b.surfaceActive.data();view.physicsOut=b.surfacePhysics.data();view.wallOut=b.surfaceWall.data();view.status=b.surfaceStatus.data();
    b.operation(h.time);
    if(count)evaluateMaterialSurfaceKernel<<<(view.workers+31)/32,32,0,b.stream.get()>>>(view,b.devicePhysics.data());
    if(!b.fault.check(cudaGetLastError(),"evaluate resident material surface"))return b.failure(error);
    CpuSurfaceResult result;std::vector<MaterialSurfaceStatus> statuses;
    if(!b.read(b.surfacePhysics,result.physics)||!b.read(b.surfaceWall,result.wall)||!b.read(b.surfaceStatus,statuses)
        ||!b.finish(error))return b.failure(error);
    for(std::size_t f=0;f<count;++f)if(!statuses[f].success){
        const auto& status=statuses[f];numerical=status.numericalFailure;
        error="CPU surface constitutive solve rejected face "+std::to_string(f%nf)+" code "+std::to_string(status.device.code);
        if(boundaryLayer&&status.wall.code!=ugkwp::gaswall::WallCode::Success){
            if(terminalWallFailure)*terminalWallFailure=true;
            error+="; stationary wall profile code "+std::to_string(int(status.wall.code))+" node "+std::to_string(status.wall.node)
                +" iteration "+std::to_string(status.wall.iteration)+" residual "+(std::isfinite(status.wall.residual)?std::to_string(status.wall.residual):"NOT_AVAILABLE");
        }
        return false;
    }
    std::vector<CpuSurfaceResult> completed(requests.size());
    for(std::size_t i=0;i<requests.size();++i){
        completed[i].physics.assign(result.physics.begin()+i*nf,result.physics.begin()+(i+1)*nf);
        completed[i].wall.assign(result.wall.begin()+i*nf,result.wall.begin()+(i+1)*nf);
    }
    output=std::move(completed);error.clear();return true;
}
bool evaluateResidentMaterialSurface(MaterialBackend& b,const HostState& h,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    Real dt,bool useRates,bool candidate,CpuSurfaceResult& output,std::string& error,
    bool& numerical,const std::vector<Real>* sideVolume,
    const std::vector<GasWallClosureContext>* contexts,bool* terminalWallFailure,
    const std::vector<WallFaceSample>* acceptedRates){
    std::vector<CpuSurfaceResult> result;
    if(!evaluateResidentMaterialSurfaces(b,h,{{&bulk,&gradients,contexts,acceptedRates}},dt,useRates,candidate,
        result,error,numerical,sideVolume,terminalWallFailure))return false;
    output=std::move(result.front());return true;
}
} // namespace chmt
