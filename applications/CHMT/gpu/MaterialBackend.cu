#include "gpu/ResidentMaterialSlice.cuh"
#include "gpu/ResidentMaterialSource.cuh"
#include "../../../common/gpu/StableRowIndex.cuh"
#include "ablation/SurfaceDonorAllocation.H"
#include "gpu/MaterialBackend.H"
#include "gpu/HostTransferQueue.H"
#include "gpu/MaterialTransportKernels.cuh"
#include "gpu/MaterialSourceKernels.cuh"
#include "gpu/MaterialSurfaceKernels.cuh"
#include "ablation/MaterialSurfaceInput.H"
#include "../../../common/gpu/ScalarWorkLaunch.cuh"
#include "materials/MaterialInventoryMath.H"
#include "materials/CpuMaterialDriver.H"
#include "../../../common/GpuDiagnosticsOwner.cuh"
#include "../../../common/gasTransport/GasBoundaryLayerWorkspace.H"
#include <algorithm>
#include <memory>
#include <limits>
#include "../../../common/gpu/ImplicitDiffusion.H"
#include "../../../common/gpu/ExecutionGraph.cuh"
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
__global__ void conductionAnchorTemperature(int n,const MaterialThermalInput* input,Real* temperature){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)temperature[c]=input[c].anchor;
}
__global__ void conductionCoefficientsKernel(int count,const MaterialThermalInput* input,
    const Real* temperature,Real dt,const PhysicsConfig* physics,MaterialThermalCoefficients* output,
    Real* capacity,Real* source,Real* conductivity){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count){const auto value=materialThermalCoefficients(input[c],temperature[c],dt,*physics);
        output[c]=value;capacity[c]=value.capacity;source[c]=value.source;conductivity[c]=value.conductivity;}
}
// Initial publication diagnostics only. The repeated constitutive kernel above
// remains unchanged, including its launch arguments and per-cell work.
__global__ void inspectConductionStartupKernel(int count,const MaterialThermalCoefficients* coefficients,
    const PhysicsConfig* physics,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    const auto value=coefficients[c];
    status[c]=value.error?MaterialStepResult{1,value.error,c,true}:MaterialStepResult{};
    if(!value.error&&value.rawPorosity!=value.porosity){auto phi=value.rawPorosity;clipPhysicsMassFraction(*physics,phi,3824,c);}
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

__device__ void mergeConductionAudit(MaterialConductionAudit& a,const MaterialConductionAudit& b){
    a.residualEnergy.merge(b.residualEnergy);a.inventory.merge(b.inventory);a.stableChange.merge(b.stableChange);a.boundaryEnergy.merge(b.boundaryEnergy);
    a.change=fmax(a.change,b.change);if(!a.coefficientError)a.coefficientError=b.coefficientError;
    if(b.invalidCell>=0&&(a.invalidCell<0||b.invalidCell<a.invalidCell)){a.invalidCell=b.invalidCell;a.invalidTemperature=b.invalidTemperature;}
}
__global__ void conductionAuditParts(int cells,int boundary,const MaterialThermalInput* input,const MaterialThermalCoefficients* coefficients,
    const Real* temperature,const Real* previous,const Real* residual,const Real* flux,const int* external,Real dt,MaterialConductionAudit* parts){
    __shared__ __align__(16) unsigned char storage[128*sizeof(MaterialConductionAudit)];
    auto* values=reinterpret_cast<MaterialConductionAudit*>(storage);MaterialConductionAudit a;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<cells){const auto value=input[c];const Real T=temperature[c];a.coefficientError=coefficients[c].error;
        a.inventory.add(absValue(value.target));a.inventory.add(absValue(value.caloric.energy<false>(T)));
        if(residual)a.residualEnergy.product(dt,absValue(residual[c]));
        if(previous){if(!finite(T)||T<=0){a.invalidCell=c;a.invalidTemperature=T;}
            a.stableChange.product(value.caloric.secant<false>(T,value.anchor),T-value.anchor);
            a.stableChange.add(-value.target);a.stableChange.add(value.caloric.energy<false>(value.anchor));
            a.change=absValue(T-previous[c])/maxValue(1,maxValue(absValue(T),absValue(previous[c])));}}
    if(c<boundary&&flux&&external[c])a.boundaryEnergy.product(dt,flux[c]);
    values[threadIdx.x]=a;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)mergeConductionAudit(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x)parts[blockIdx.x]=values[0];
}
__global__ void conductionAuditFinish(int count,const MaterialConductionAudit* parts,MaterialConductionAudit* output){
    __shared__ __align__(16) unsigned char storage[128*sizeof(MaterialConductionAudit)];
    auto* values=reinterpret_cast<MaterialConductionAudit*>(storage);MaterialConductionAudit a;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)mergeConductionAudit(a,parts[i]);values[threadIdx.x]=a;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)mergeConductionAudit(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x)*output=values[0];
}

struct MaterialConductionIteration {
    int iteration=0,failure=0;bool valid=true,converged=false,stalled=false;
    Real residual=0,energyTolerance=0;
    MaterialConductionAudit audit;
    ugkp::gpu::ConjugateGradientControls linear;
};
__global__ void beginConductionLoop(MaterialConductionIteration* report,
    ugkp::gpu::ConjugateGradientControls linear,ugkp::diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;*report={};report->linear=linear;report->valid=!ugkp::diagnostics::hasFatal(context);
    cudaGraphSetConditional(handle,report->valid);
}
__global__ void prepareConductionLinearControls(MaterialConductionIteration* report,const MaterialConductionAudit* audit,
    CpuMaterialControls controls,cudaGraphConditionalHandle permitted){
    if(threadIdx.x)return;++report->iteration;report->audit=*audit;
    if(audit->coefficientError){report->valid=false;report->failure=1;cudaGraphSetConditional(permitted,0);return;}
    using ugkp::gpu::ScaledThermalSum;const auto initial=audit->residualEnergy;
    ScaledThermalSum scale;scale.add(1);if(scale.lessEqual(initial))scale=initial;
    auto tolerance=scale.multiplied(controls.energyRelativeTolerance);tolerance.add(controls.energyAbsoluteTolerance);
    tolerance.merge(audit->inventory.multiplied(64*std::numeric_limits<Real>::epsilon()));
    const Real relative=initial.populated?fmin(1.,tolerance.ratioTo(initial)*controls.linearEnergyBudgetFraction):1.;
    report->linear.tolerance=std::numeric_limits<Real>::min();report->linear.relativeTolerance=relative;
    cudaGraphSetConditional(permitted,1);
}
__global__ void finishConductionIteration(MaterialConductionIteration* report,const MaterialConductionAudit* audit,
    const ugkp::gpu::ConjugateGradientReport* linear,CpuMaterialControls controls,
    ugkp::diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;using namespace ugkp::diagnostics;
    if(!report->valid){cudaGraphSetConditional(handle,0);return;}
    report->audit=*audit;
    if(!finite(linear->finalResidual)){report->valid=false;report->failure=2;}
    else if(audit->invalidCell>=0){
        report->valid=false;report->failure=3;
        const Real temperature=audit->invalidTemperature;
        recordFatal(context,makeRecord(!finite(temperature)?Code::NonFinite:Code::NegativeTemperature,
            4103,4,0,EntityKind::Cell,audit->invalidCell,-1,temperature,0,0,Quantity::Temperature));
    }
    else if(!finite(audit->change)){
        report->valid=false;report->failure=4;
        recordFatal(context,makeRecord(Code::NonFinite,4104,4,0,EntityKind::None,-1,-1,audit->change));
    }
    if(!report->valid){cudaGraphSetConditional(handle,0);return;}
    if(!linear->converged||linear->singular)recordWarning(context,makeRecord(Code::AccuracyResidual,34020,4,report->iteration,
        EntityKind::None,-1,-1,linear->finalResidual,linear->initialResidual*report->linear.relativeTolerance,0,Quantity::LinearSystem));
    auto energy=audit->stableChange;energy.merge(audit->boundaryEnergy);report->residual=energy.binary64();
    auto scale=audit->stableChange.magnitude();const auto boundary=audit->boundaryEnergy.magnitude();if(scale.lessEqual(boundary))scale=boundary;
    ugkp::gpu::ScaledThermalSum unity;unity.add(1);if(scale.lessEqual(unity))scale=unity;
    auto tolerance=scale.multiplied(controls.energyRelativeTolerance);tolerance.add(controls.energyAbsoluteTolerance);
    tolerance.merge(audit->inventory.multiplied(64*std::numeric_limits<Real>::epsilon()));report->energyTolerance=tolerance.binary64();
    const bool energyConverged=finite(report->residual)&&absValue(report->residual)<=report->energyTolerance;
    report->stalled=audit->change==0&&linear->iterations==0;
    report->converged=audit->change<=controls.nonlinearRelativeTolerance&&energyConverged;
    const bool again=!report->converged&&!report->stalled&&report->iteration<controls.nonlinearMaxIterations;
    cudaGraphSetConditional(handle,again);
    if(!again){
        if(report->stalled&&!energyConverged)recordWarning(context,makeRecord(Code::AccuracyResidual,34021,4,report->iteration,
            EntityKind::None,-1,-1,report->residual,report->energyTolerance,0,Quantity::Accuracy));
        if(audit->change>controls.nonlinearRelativeTolerance)recordWarning(context,makeRecord(Code::AccuracyResidual,34022,4,report->iteration,
            EntityKind::None,-1,-1,audit->change,controls.nonlinearRelativeTolerance,0,Quantity::Accuracy));
        if(!energyConverged)recordWarning(context,makeRecord(Code::ConservationResidual,34023,4,report->iteration,
            EntityKind::None,-1,-1,report->residual,report->energyTolerance,0,Quantity::Conservation));
    }
}
}
namespace {
__global__ void gatherMaterialObservation(int count,const int* cells,const SolidQ* state,SolidQ* observed){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<count)observed[row]=state[cells[row]];
}
}
struct MaterialRowStorage {
    Buffer<int> keys,sortedKeys,inputIndices,counts;
};
struct PacketSolidRow {
    const ExchangePacket* packets;
    __device__ int operator()(int i)const{
        return (requiredConsumers(packets[i].kind)&ConsumeSolid)?packets[i].solidCell:-1;
    }
};
struct RadiationSolidRow {
    const unsigned* receivers;const int* cells;
    __device__ int operator()(int i)const{return receivers[i]==ConsumeSolid?cells[i]:-1;}
};
__global__ void gatherMaterialSliceReceivers(ResidentMaterialDriveHistory history,std::size_t frame,unsigned* receivers){
    const auto f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(f<history.faces)receivers[f]=history.surface[frame*history.faces+f].radiationReceiver;
}
struct MaterialBackend {
    CudaFault fault;
    DeviceStream stream;
    HostTransferQueue transfers{fault};
    ugkp::diagnostics::GpuOwner diagnostics;
    PhysicsConfig physics{};
    std::uint64_t sequence=0,step=0;Real time=0;
    Buffer<PhysicsConfig> devicePhysics,quietSurfacePhysics;
    Buffer<SolidQ> current,next,initial,gasChange,candidateGasChange,donor;
    // Geometry still consumes a published host mirror. Remember that exact
    // publication so a window handoff need not upload it back to the GPU.
    // Compare fields, not a hash/sequence: external callers may replace state.
    Buffer<int> observationCells;Buffer<SolidQ> observation;
    std::vector<int> cachedObservationCells;int observedInventorySize=-1;
    std::vector<SolidQ> publishedCurrent;
    bool publishedCurrentValid=false;
    bool retains(const std::vector<SolidQ>& input)const{
        if(!publishedCurrentValid||input.size()!=publishedCurrent.size())return false;
        for(std::size_t c=0;c<input.size();++c){
            const auto& a=input[c];const auto& v=publishedCurrent[c];
            for(int k=0;k<Nc;++k)if(a.condensed[k]!=v.condensed[k])return false;
            for(int k=0;k<Ns;++k)if(a.pore[k]!=v.pore[k])return false;
            for(int k=0;k<Nr;++k)if(a.progress[k]!=v.progress[k])return false;
            if(a.energy!=v.energy||a.porosity!=v.porosity)return false;
        }
        return true;
    }
    Buffer<Real> initialVolumes,conductionVolumes,conductionTemperature;
    Buffer<Real> conductionCapacity,conductionSource,conductionConductivity;
    Buffer<MaterialThermalInput> conductionInput;
    Buffer<MaterialThermalCoefficients> conductionCoefficients;
    Buffer<MaterialConductionIteration> conductionReport;
    struct ConductionGraph{
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        ugkp::gpu::ExecutionGraphSignature signature;
        void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);executable=nullptr;graph=nullptr;}
        ~ConductionGraph(){clear();}
    } conductionGraph;
    Buffer<MaterialConductionAudit> conductionAuditParts,conductionAuditResult;Buffer<int> conductionExternal;std::vector<int> cachedConductionExternal;
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
    ResidentMaterialDriveHistory borrowedHistory{};
    ugkp::IntervalSlicePlan pendingSlicePlan{};
    Buffer<IntervalSliceRecordPart> sliceParts;
    Real sliceExternalFraction=0;bool sliceExternalLast=false;
    bool sliceReady=false,slicePending=false;
    Buffer<ugkp::IntervalSliceCursor> sliceCursor;
    Buffer<ugkp::IntervalSlicePlan> slicePlan;
    Buffer<ExchangePacket> sliceUsedPackets,sliceCandidatePackets;
    Buffer<Real> sliceUsedRadiation,sliceCandidateRadiation,sliceExternal;
    Buffer<Real> sliceUsedExternal,sliceCandidateExternal;
    MaterialSliceAccumulators currentSlice()const{return {sliceUsedPackets.data(),sliceUsedRadiation.data(),sliceUsedExternal.data()};}
    MaterialSliceAccumulators candidateSlice()const{return {sliceCandidatePackets.data(),sliceCandidateRadiation.data(),sliceCandidateExternal.data()};}
    ResidentMaterialSlice sourceSlice()const{
        ResidentMaterialSlice slice{borrowedHistory,slicePlan.data(),currentSlice(),true};
        slice.selectedParts=sliceParts.data();slice.externalFraction=sliceExternalFraction;
        slice.externalLast=sliceExternalLast;slice.hostSelected=true;return slice;
    }

    Buffer<int> packetOffsets,packetIndices,radiationOffsets,radiationFaces,radiationCells;
    MaterialRowStorage packetRows,radiationRows;Buffer<unsigned char> rowScratch;
    std::vector<int> radiationCellMap;
    Buffer<Real> radiation;
    Buffer<unsigned> radiationReceiver;
    Buffer<SurfacePhysicsInput> surfaceInput;
    Buffer<int> surfaceCells;
    Buffer<unsigned char> surfaceActive,wallWorkspace;
    Buffer<SurfacePhysicsResult> surfacePhysics;
    Buffer<WallFaceSample> surfaceWall;
    Buffer<MaterialSurfaceStatus> surfaceStatus;
    Buffer<ugkwp::gaswall::WallOutput<Real,Ns>> surfaceTeamOutput;
    bool surfaceProbePlanReady=false;int surfaceProbeResident2=0,surfaceProbeResident3=0;
    Buffer<Real> quadratureDistance,quadratureWeight,mechanismBasis;
    Buffer<ugkwp::GasReactionData<Real>> mechanismReactions;
    Buffer<ugkwp::GasStoichTerm<Real>> mechanismReactants,mechanismProducts;
    Buffer<ugkwp::GasColliderEfficiency<Real>> mechanismEfficiencies;
    ugkwp::GasMechanismView<Real,Ns> deviceMechanism{};
    bool mechanismReady=false,meshReady=false;
    std::uint64_t geometryVersion=0,topologyHash=0;
    int cells=0,faces=0;
    cudaEvent_t conductionReady=nullptr;
    ~MaterialBackend(){stream.sync();if(conductionReady)cudaEventDestroy(conductionReady);}
    bool waitConduction(void* producer){
        if(!conductionReady&&!fault.check(cudaEventCreateWithFlags(&conductionReady,cudaEventDisableTiming),"create conduction dependency"))return false;
        return fault.check(cudaEventRecord(conductionReady,static_cast<cudaStream_t>(producer)),"record conduction dependency")
            &&fault.check(cudaStreamWaitEvent(stream.get(),conductionReady,0),"wait conduction dependency");
    }
    template<class T> bool allocate(Buffer<T>& buffer,std::size_t size){return buffer.resize(size,fault);}
    template<class T> bool uploadRaw(Buffer<T>& buffer,const T* data,std::size_t size){
        return allocate(buffer,size)&&transfers.upload(buffer.data(),data,size);
    }
    template<class T> bool upload(Buffer<T>& buffer,const std::vector<T>& data){return uploadRaw(buffer,data.data(),data.size());}
    template<class T> bool read(const Buffer<T>& buffer,std::vector<T>& data){
        data.resize(buffer.size());return transfers.download(data.data(),buffer.data(),buffer.size());
    }
    template<class Row> bool buildRows(int items,int rows,Row row,MaterialRowStorage& storage,
        Buffer<int>& offsets,Buffer<int>& indices){
        const auto reserve=[&](Buffer<int>& buffer,std::size_t count){
            return buffer.size()>=count||allocate(buffer,count);
        };
        if(!reserve(storage.keys,items)||!reserve(storage.sortedKeys,items)
            ||!reserve(storage.inputIndices,items)||!reserve(storage.counts,rows+1)
            ||!reserve(offsets,rows+1)||!reserve(indices,items))return false;
        ugkp::gpu::StableRowIndexView view{storage.keys.data(),storage.sortedKeys.data(),
            storage.inputIndices.data(),indices.data(),storage.counts.data(),offsets.data()};
        std::size_t bytes=0;
        if(!fault.check(ugkp::gpu::stableRowIndexWorkspace(items,rows,view,bytes,stream.get()),
            "size stable material row index")||(rowScratch.size()<bytes&&!allocate(rowScratch,bytes)))return false;
        return fault.check(ugkp::gpu::buildStableRowIndex(items,rows,row,view,rowScratch.data(),bytes,stream.get()),
            "build stable material row index");
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
    auto quietPhysics=physics;quietPhysics.diagnostics=nullptr;quietPhysics.tolerances.diagnostics=nullptr;
    auto devicePhysics=physics;devicePhysics.diagnostics=b.diagnostics.device();
    devicePhysics.tolerances.diagnostics=b.diagnostics.device();
    b.sequence=state.commitSequence+1;b.step=state.acceptedSteps;b.time=state.time;
    if(!b.retains(state.solid)){
        b.publishedCurrentValid=false;
        if(!b.upload(b.current,state.solid))return b.failure(error);
        b.publishedCurrent=state.solid;b.publishedCurrentValid=true;
    }
    if(!b.configureMesh(state.solidMesh,error))return false;
    if(state.solid.empty()){
        if(!b.allocate(b.initialVolumes,0)||!b.allocate(b.nextVolumes,0))return b.failure(error);
    }else if(!b.initialVolumes.copy(b.volumes,b.fault,b.stream.get())
        ||!b.nextVolumes.copy(b.volumes,b.fault,b.stream.get()))return b.failure(error);
    if(!b.initial.copy(b.current,b.fault,b.stream.get())
        ||!b.allocate(b.next,state.solid.size())
        ||!b.allocate(b.gasChange,state.solid.size())||!b.gasChange.zero(b.stream.get())
        ||!b.allocate(b.candidateGasChange,state.solid.size())
        ||!b.allocate(b.sourceStatus,state.solid.size())||!b.allocate(b.stepResult,1)
        ||!b.uploadRaw(b.devicePhysics,&devicePhysics,1)||!b.uploadRaw(b.quietSurfacePhysics,&quietPhysics,1))return b.failure(error);
    b.operation(state.time);
    return b.finish(error);
}
bool prepareResidentMaterialConduction(MaterialBackend& b,const HostMesh& mesh,
    const std::vector<Real>& nativeVolumes,std::vector<MaterialThermalInput>& input,
    std::string& error,bool& numerical,bool publishInput){
    numerical=true;
    if(!b.configureMesh(mesh,error))return false;
    if(!b.upload(b.conductionVolumes,nativeVolumes)||!b.allocate(b.conductionInput,b.current.size())
        ||!b.allocate(b.conductionCoefficients,b.current.size())||!b.allocate(b.conductionCapacity,b.current.size())
        ||!b.allocate(b.conductionSource,b.current.size())||!b.allocate(b.conductionConductivity,b.current.size()))return b.failure(error);
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
    if(!b.allocate(b.conductionTemperature,b.current.size()))return b.failure(error);
    if(b.cells)conductionAnchorTemperature<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,b.conductionInput.data(),b.conductionTemperature.data());
    if(publishInput){if(!b.read(b.conductionInput,input)||!b.finish(error))return b.failure(error);}
    else input.clear();
    error.clear();return true;
}
bool publishResidentMaterialConductionInput(MaterialBackend& b,std::vector<MaterialThermalInput>& input,std::string& error){
    if(!b.read(b.conductionInput,input)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool evaluateResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    Real dt,std::vector<MaterialThermalCoefficients>& coefficients,std::string& error,
    const Real* deviceTemperature,void* temperatureStream,bool download,bool startupDiagnostics){
    if(deviceTemperature){if(!b.waitConduction(temperatureStream))return b.failure(error);}
    else if(!b.upload(b.conductionTemperature,temperature))return b.failure(error);
    b.operation(b.time);
    if(b.cells)conductionCoefficientsKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,
        b.conductionInput.data(),deviceTemperature?deviceTemperature:b.conductionTemperature.data(),dt,b.devicePhysics.data(),b.conductionCoefficients.data(),
        b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data());
    if(!b.fault.check(cudaGetLastError(),"evaluate resident material conduction"))return b.failure(error);
    if(startupDiagnostics){
        if(b.cells)inspectConductionStartupKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(
            b.cells,b.conductionCoefficients.data(),b.devicePhysics.data(),b.sourceStatus.data());
        selectConductionFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
        MaterialStepResult result;
        if(!b.fault.check(cudaGetLastError(),"inspect resident startup coefficients")
            ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
        if(result.phase){error=result.code==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve";return false;}
    }
    if(download){if(!b.read(b.conductionCoefficients,coefficients)||!b.finish(error))return b.failure(error);}
    else coefficients.clear();
    error.clear();return true;
}
MaterialConductionDeviceFields residentMaterialConductionFields(MaterialBackend& b){
    return {b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data(),b.stream.get(),b.conductionTemperature.data(),b.current.size()};
}
namespace {
const MaterialConductionAudit* enqueueConductionAudit(MaterialBackend& b,const Real* temperature,const Real* previous,
    const Real* residual,const Real* flux,int boundary,Real dt,cudaStream_t stream){
    const int blocks=(std::max(b.cells,boundary)+127)/128;
    if(blocks)conductionAuditParts<<<blocks,128,0,stream>>>(b.cells,boundary,b.conductionInput.data(),b.conductionCoefficients.data(),
        temperature,previous,residual,flux,b.conductionExternal.data(),dt,b.conductionAuditParts.data());
    conductionAuditFinish<<<1,128,0,stream>>>(blocks,b.conductionAuditParts.data(),b.conductionAuditResult.data());
    return b.conductionAuditResult.data();
}
}
bool solveResidentMaterialConduction(MaterialBackend& b,ugkp::gpu::ImplicitDiffusion& diffusion,
    ugkp::gpu::SparseConjugateGradient& sparse,const std::vector<int>& external,Real dt,const CpuMaterialControls& controls,
    Real normalizationSmall,Real singularSmall,std::string& error){
    if(!b.waitConduction(diffusion.deviceValues().readyStream))return b.failure(error);
    if(b.cachedConductionExternal!=external){if(!b.upload(b.conductionExternal,external))return b.failure(error);b.cachedConductionExternal=external;}
    const int blocks=(std::max(b.cells,static_cast<int>(external.size()))+127)/128;
    auto& report=b.conductionReport;
    if(!b.allocate(b.conductionAuditParts,blocks)||!b.allocate(b.conductionAuditResult,1)||!b.allocate(report,1))return b.failure(error);
    // Reuse the complete nonlinear/sparse graph while every captured argument
    // remains valid. Matrix values and geometry coefficients update in place.
    // Changed extents, storage addresses, interval or controls recapture it.
    auto& graph=b.conductionGraph;
    auto signature=diffusion.residentExecutionSignature();
    signature.append(sparse.residentExecutionSignature());
    signature.add(b.cells);signature.add(external.size());signature.add(dt);
    signature.add(normalizationSmall);signature.add(singularSmall);
    signature.add(controls.nonlinearMaxIterations);signature.add(controls.nonlinearRelativeTolerance);
    signature.add(controls.energyAbsoluteTolerance);signature.add(controls.energyRelativeTolerance);
    signature.add(controls.linearMaxIterations);signature.add(controls.linearEnergyBudgetFraction);
    for(const void* pointer:{static_cast<const void*>(report.data()),static_cast<const void*>(b.diagnostics.device()),
        static_cast<const void*>(b.conductionInput.data()),static_cast<const void*>(b.devicePhysics.data()),
        static_cast<const void*>(b.conductionCoefficients.data()),static_cast<const void*>(b.conductionCapacity.data()),
        static_cast<const void*>(b.conductionSource.data()),static_cast<const void*>(b.conductionConductivity.data()),
        static_cast<const void*>(b.conductionAuditParts.data()),static_cast<const void*>(b.conductionAuditResult.data()),
        static_cast<const void*>(b.conductionExternal.data())})signature.add(pointer);
    if(!(graph.signature==signature)){graph.clear();graph.signature=std::move(signature);}
    using namespace ugkp::gpu;using executionGraphDetail::appendCaptured;
    try{
        if(!graph.executable){
        executionGraphDetail::checked(cudaGraphCreate(&graph.graph,0));cudaGraphConditionalHandle handle,permitted;
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&handle,graph.graph,0,cudaGraphCondAssignDefault));
        ConjugateGradientControls linear;linear.maximumIterations=controls.linearMaxIterations;linear.normalizationSmall=normalizationSmall;linear.singularSmall=singularSmall;
        const auto initialized=appendCaptured(graph.graph,{},b.stream.get(),[&]{
            beginConductionLoop<<<1,1,0,b.stream.get()>>>(report.data(),linear,b.diagnostics.device(),handle);});
        cudaGraph_t body=nullptr;executionGraphDetail::addWhile(graph.graph,handle,{initialized},body);
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&permitted,body,0,cudaGraphCondAssignDefault));
        const auto assembled=appendCaptured(body,{},b.stream.get(),[&]{
            diffusion.enqueueResidentSnapshot(b.stream.get());
            if(b.cells)conductionCoefficientsKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,b.conductionInput.data(),
                diffusion.deviceTemperature(),dt,b.devicePhysics.data(),b.conductionCoefficients.data(),b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data());
            diffusion.enqueueResidentAssembly(b.stream.get());
        });
        const auto residual=sparse.appendResidentLocalResidual(body,assembled,diffusion.deviceSolution(),diffusion.deviceResidual());
        const auto budget=appendCaptured(body,{residual},b.stream.get(),[&]{
            const auto audit=enqueueConductionAudit(b,diffusion.deviceTemperature(),nullptr,diffusion.deviceResidual(),nullptr,external.size(),dt,b.stream.get());
            prepareConductionLinearControls<<<1,1,0,b.stream.get()>>>(report.data(),audit,controls,permitted);
        });
        cudaGraph_t solveBody=nullptr;const auto solve=executionGraphDetail::addIf(body,permitted,{budget},solveBody);
        const auto solved=sparse.appendResidentLocalLdu(solveBody,nullptr,diffusion.deviceSolution(),&report.data()->linear);
        appendCaptured(solveBody,{solved},b.stream.get(),[&]{
            diffusion.enqueueResidentBoundaryTemperature(b.stream.get());diffusion.enqueueResidentBoundaryFlux(b.stream.get());
            enqueueConductionAudit(b,diffusion.deviceTemperature(),diffusion.devicePreviousSolution(),nullptr,diffusion.deviceBoundaryFlux(),external.size(),dt,b.stream.get());
        });
        appendCaptured(body,{solve},b.stream.get(),[&]{
            finishConductionIteration<<<1,1,0,b.stream.get()>>>(report.data(),b.conductionAuditResult.data(),sparse.deviceReport(),controls,b.diagnostics.device(),handle);
        });
        executionGraphDetail::checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
        }
        executionGraphDetail::checked(cudaGraphLaunch(graph.executable,b.stream.get()));
        MaterialConductionIteration result;
        if(!b.transfers.download(&result,report.data(),1)||!b.finish(error))return b.failure(error);
        if(!result.valid){
            error=result.failure==1?(result.audit.coefficientError==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve")
                :result.failure==2?"nonfinite OpenFOAM implicit material sparse residual"
                :result.failure==3?"nonfinite or nonpositive implicit material temperature":"nonfinite implicit material residual";
            return false;
        }
        error.clear();return true;
    }catch(const std::exception& exception){graph.clear();error=exception.what();return false;}
}
bool auditResidentMaterialConduction(MaterialBackend& b,const Real* temperature,const Real* previous,
    const Real* residual,const Real* flux,const std::vector<int>& external,Real dt,void* stream,
    MaterialConductionAudit& audit,std::string& error){
    if(!b.waitConduction(stream))return b.failure(error);
    if(b.cachedConductionExternal!=external){if(!b.upload(b.conductionExternal,external))return b.failure(error);b.cachedConductionExternal=external;}
    const int count=std::max(b.cells,static_cast<int>(external.size())),blocks=(count+127)/128;
    if(!b.allocate(b.conductionAuditParts,blocks)||!b.allocate(b.conductionAuditResult,1))return b.failure(error);
    if(blocks)conductionAuditParts<<<blocks,128,0,b.stream.get()>>>(b.cells,external.size(),b.conductionInput.data(),b.conductionCoefficients.data(),
        temperature,previous,residual,flux,b.conductionExternal.data(),dt,b.conductionAuditParts.data());
    conductionAuditFinish<<<1,128,0,b.stream.get()>>>(blocks,b.conductionAuditParts.data(),b.conductionAuditResult.data());
    if(!b.fault.check(cudaGetLastError(),"reduce resident material conduction")
        ||!b.transfers.download(&audit,b.conductionAuditResult.data(),1)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool finishResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    std::vector<SolidQ>& hostMirror,std::string& error,bool& numerical,const Real* deviceTemperature,void* temperatureStream){
    numerical=true;
    if(deviceTemperature){if(!b.waitConduction(temperatureStream))return b.failure(error);}
    else if(!b.upload(b.conductionTemperature,temperature))return b.failure(error);
    b.operation(b.time);
    if(b.cells)finishConductionKernel<<<(b.cells+127)/128,128,0,b.stream.get()>>>(b.cells,b.current.data(),
        b.volumes.data(),b.conductionInput.data(),deviceTemperature?deviceTemperature:b.conductionTemperature.data(),b.devicePhysics.data(),
        b.next.data(),b.sourceStatus.data());
    selectConductionFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
    MaterialStepResult result;
    if(!b.fault.check(cudaGetLastError(),"finish resident material conduction")
        ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.phase){numerical=result.numerical;error="implicit material endpoint EOS invalid";return false;}
    std::vector<SolidQ> candidate;
    if(!b.read(b.next,candidate)||!b.finish(error))return b.failure(error);
    b.current.swap(b.next);hostMirror.swap(candidate);
    b.publishedCurrent=hostMirror;b.publishedCurrentValid=true;
    error.clear();return true;
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
bool beginResidentMaterialHistory(MaterialBackend& b,const ResidentMaterialDriveHistory* supplied,
    const IntervalHistory& host,std::string& error){
    b.sliceReady=false;b.slicePending=false;b.borrowedHistory={};
    // Optional borrowed view is an optimization only. External material callers
    // and histories without the producer layout keep the original upload path.
    if(!supplied||!supplied->surface||!supplied->interval||!supplied->faces
        ||supplied->frames!=host.records().size())return true;
    for(const auto& record:host.records())if(record.radiationEnergy.size()!=supplied->faces
        ||record.radiationReceiver.size()!=supplied->faces)return true;
    b.borrowedHistory=*supplied;b.pendingSlicePlan={};
    const ugkp::IntervalSliceCursor initialCursor{};
    if(!b.waitConduction(supplied->producerStream)
        ||!b.uploadRaw(b.sliceCursor,&initialCursor,1)||!b.allocate(b.slicePlan,1)
        ||!b.allocate(b.sliceUsedPackets,3*supplied->faces)||!b.allocate(b.sliceCandidatePackets,3*supplied->faces)
        ||!b.allocate(b.sliceUsedRadiation,supplied->faces)||!b.allocate(b.sliceCandidateRadiation,supplied->faces)
        ||!b.upload(b.sliceExternal,host.externalRadiationEnergy())
        ||!b.allocate(b.sliceUsedExternal,b.sliceExternal.size())||!b.allocate(b.sliceCandidateExternal,b.sliceExternal.size())
        ||!b.sliceUsedPackets.zero(b.stream.get())||!b.sliceUsedRadiation.zero(b.stream.get())
        ||!b.sliceUsedExternal.zero(b.stream.get()))return b.failure(error);
    b.sliceReady=true;return true;
}
bool advanceResidentMaterialSources(MaterialBackend& b,const HostMesh& endpointMesh,
    const SurfaceMesh& surface,const std::vector<ExchangePacket>& packets,
    const std::vector<Real>& radiation,const std::vector<unsigned>& receivers,
    Real dt,const CpuMaterialControls& controls,bool finalStep,std::vector<SolidQ>& shadow,
    std::vector<SolidQ>& cumulative,std::string& error,bool& numerical,const std::vector<int>* observedCells,const IntervalSliceMetadata* selectedSlice){
    numerical=true;
    const int count=int(b.current.size());
    if(controls.retainDonorHistory&&!b.allocate(b.donor,count))return b.failure(error);
    // Keep existing owner admission errors, while constructing row schedules
    // from resident packets and receiver maps on device in stable input order.
    for(const auto& packet:packets)if(requiredConsumers(packet.kind)&ConsumeSolid){
        if(packet.solidCell<0||packet.solidCell>=count){error="material packet owner invalid";return false;}
    }
    for(std::size_t f=0;f<radiation.size();++f){
        if(receivers[f]!=ConsumeSolid)continue;
        const int cell=surface.solidCell[f];
        if(cell<0||cell>=count){error="radiation has no material owner";return false;}
    }
    if(b.radiationCellMap!=surface.solidCell){
        if(!b.upload(b.radiationCells,surface.solidCell))return b.failure(error);
        b.radiationCellMap=surface.solidCell;
    }
    b.slicePending=b.sliceReady&&selectedSlice;
    if(b.slicePending){
        // These decisions were emitted by the same host cursor that produced
        // the geometry/ledger packets. No device time comparison is repeated.
        b.pendingSlicePlan={};b.pendingSlicePlan.begin.record=selectedSlice->beginRecord;
        b.pendingSlicePlan.past=selectedSlice->beginRecord+selectedSlice->parts.size();
        b.pendingSlicePlan.end.record=selectedSlice->endRecord;
        b.sliceExternalFraction=selectedSlice->externalFraction;b.sliceExternalLast=selectedSlice->externalLast;
    }
    if(!b.upload(b.nextVolumes,endpointMesh.volumes))return b.failure(error);
    if(b.slicePending){
        if(!b.allocate(b.radiation,radiation.size())||!b.allocate(b.radiationReceiver,receivers.size())
            ||!b.uploadRaw(b.slicePlan,&b.pendingSlicePlan,1)||!b.upload(b.sliceParts,selectedSlice->parts))return b.failure(error);
        if(!b.fault.check(prepareHostSelectedMaterialSlice(b.sourceSlice(),b.candidateSlice(),
            b.sliceExternal.data(),b.sliceExternal.size(),b.radiation.data(),b.stream.get()),
            "prepare borrowed material source slice"))return b.failure(error);
        gatherMaterialSliceReceivers<<<material_slice_detail::blocks(b.borrowedHistory.faces),128,0,b.stream.get()>>>(
            b.borrowedHistory,b.pendingSlicePlan.begin.record,b.radiationReceiver.data());
        if(!b.fault.check(cudaGetLastError(),"borrow material radiation receiver map"))return b.failure(error);
        const auto slots=(b.pendingSlicePlan.past-b.pendingSlicePlan.begin.record)*b.borrowedHistory.faces*3;
        if(!b.buildRows(int(slots),count,ResidentMaterialSliceRow{b.sourceSlice()},b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
    }else{
        if(!b.upload(b.packets,packets)||!b.upload(b.radiation,radiation)||!b.upload(b.radiationReceiver,receivers)
            ||!b.buildRows(int(packets.size()),count,PacketSolidRow{b.packets.data()},b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
    }
    if(!b.buildRows(int(radiation.size()),count,RadiationSolidRow{b.radiationReceiver.data(),b.radiationCells.data()},
        b.radiationRows,b.radiationOffsets,b.radiationFaces))return b.failure(error);
    if(!b.fault.check(cudaMemsetAsync(b.stepResult.data(),0,sizeof(MaterialStepResult),b.stream.get()),
        "clear material source result"))return b.failure(error);
    if(count){
        if(b.slicePending){
        applyMaterialSliceSourcesKernel<<<(count+127)/128,128,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
            b.initial.data(),b.gasChange.data(),b.candidateGasChange.data(),b.sourceSlice(),b.packetOffsets.data(),b.packetIndices.data(),
            b.radiationOffsets.data(),b.radiationFaces.data(),radiation.empty()?nullptr:b.radiation.data(),
            b.radiationReceiver.data(),b.nextVolumes.data(),b.devicePhysics.data(),dt,finalStep,
            b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
        }else{
        applyMaterialSourcesKernel<<<(count+127)/128,128,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
            b.initial.data(),b.gasChange.data(),b.candidateGasChange.data(),b.packets.data(),b.packetOffsets.data(),b.packetIndices.data(),
            b.radiationOffsets.data(),b.radiationFaces.data(),radiation.empty()?nullptr:b.radiation.data(),
            b.radiationReceiver.data(),b.nextVolumes.data(),b.devicePhysics.data(),dt,finalStep,
            b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
        }
    }
    selectSourceFailure<<<1,256,0,b.stream.get()>>>(b.sourceStatus.data(),count,b.stepResult.data());
    if(!b.fault.check(cudaGetLastError(),"launch resident material local sources"))return b.failure(error);
    MaterialStepResult result;std::vector<SolidQ> candidateShadow,candidateDonor;
    if(observedCells){
        if(b.cachedObservationCells!=*observedCells||b.observedInventorySize!=count){
            for(int cell:*observedCells)if(cell<0||cell>=count){error="material observation cell outside inventory";return false;}
            if(!b.upload(b.observationCells,*observedCells))return b.failure(error);
            b.cachedObservationCells=*observedCells;b.observedInventorySize=count;
        }
        if(!b.allocate(b.observation,observedCells->size()))return b.failure(error);
        const int rows=observedCells->size();
        if(rows)gatherMaterialObservation<<<(rows+127)/128,128,0,b.stream.get()>>>(rows,b.observationCells.data(),b.next.data(),b.observation.data());
        if(!b.fault.check(cudaGetLastError(),"gather material geometry observation")
            ||!b.read(b.observation,candidateShadow))return b.failure(error);
    }else if(!b.read(b.next,candidateShadow))return b.failure(error);
    if(!b.transfers.download(&result,b.stepResult.data(),1)||(controls.retainDonorHistory&&!b.read(b.donor,candidateDonor))
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
bool commitResidentMaterialStep(MaterialBackend& b,Real time,std::string& error){
    if(b.slicePending){
        // Advance accepted history only with the whole material transaction;
        // a failed endpoint surface solve never consumes this candidate slice.
        if(!b.fault.check(commitResidentMaterialSlice(b.slicePlan.data(),b.candidateSlice(),b.currentSlice(),
            b.borrowedHistory.faces,b.sliceExternal.size(),b.sliceCursor.data(),nullptr,b.stream.get()),
            "commit borrowed material source slice"))return b.failure(error);
        b.slicePending=false;
    }
    b.current.swap(b.next);b.volumes.swap(b.nextVolumes);b.gasChange.swap(b.candidateGasChange);b.time=time;
    // A local-source commit changes current before the conduction publication.
    b.publishedCurrentValid=false;return true;
}

void commitResidentMaterialStep(MaterialBackend& b,Real time){
    // Preserve the old external entry point. Its default source path does not
    // bind resident history, so no new cursor commit kernel is launched.
    // The production driver uses the error-reporting overload above.
    std::string error;(void)commitResidentMaterialStep(b,time,error);
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
            auto& in=inputs[index];in.ablating=s.ablating.empty()||s.ablating.at(f);
            setMaterialSurfaceGasDrive(in,bulk[f],gradients.empty()?nullptr:&gradients[f],gasPhaseVolume(h,gc),donorShares[f],
                accepted&&!accepted->empty()?&(*accepted)[f]:nullptr,p);
            if(contexts){
                setMaterialSurfaceMatching(in,(*contexts)[f]);
                in.wallContext.mechanism=b.deviceMechanism;in.wallContext.workspace=nullptr;in.wallContext.failureStatus=nullptr;
                in.wallContext.teamOutput=nullptr;in.wallContext.teamDiagnostics=nullptr;in.wallContext.quietPhysics=nullptr;
                const auto& q=(*contexts)[f].quadrature;
                quadratureOffsets[index]=int(distances.size());
                for(int j=0;j<q.count;++j){distances.push_back(q.distance[j]);weights.push_back(q.volumeWeight[j]);}
            }
            setMaterialSurfaceGasGeometry(in,s.area[f],s.normal[f],h.gasMesh.areaVectors[gf],s.gasDistance[f],
                s.baseVelocity.empty()?Vec3{}:s.baseVelocity[f],dt);
            const int sc=s.solidCell.empty()?-1:s.solidCell[f];solidCells[index]=sc;in.hasSolid=sc>=0;
            if(in.hasSolid){
                if(std::size_t(sc)>=b.current.size()||std::size_t(sc)>=h.solidMesh.volumes.size()
                    ||s.solidFace.size()!=nf||s.solidDistance.size()!=nf){
                    numerical=s.solidFace.size()>=nf&&s.solidDistance.size()>=nf;
                    error="CPU interface material owner invalid";return false;
                }
                const int sf=s.solidFace[f];
                if(sf<0||std::size_t(sf)>=h.solidMesh.areaVectors.size()){error="CPU interface solid face invalid";return false;}
                setMaterialSurfaceSolidGeometry(in,s.solidDistance[f],h.solidMesh.areaVectors[sf]);
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
    int surfaceProbeDepth=1,surfaceProbes=1;
    const bool cooperativeWall=boundaryLayer&&p.wallModel.model==ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
    if(boundaryLayer&&p.wallModel.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport&&count){
        int device=0,sms=0;
        if(!b.fault.check(cudaGetDevice(&device),"query material wall device")
            ||!b.fault.check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"query material wall multiprocessors"))return b.failure(error);
        view.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(p.wallModel.nodes);
        view.workspaceStride=ugkwp::gasBoundaryLayerWorkspaceBytes<Real,Ns>(p.wallModel.nodes);
        auto sizing=ugkwp::resolveBoundaryLayerWorkspace(int(count),p.wallWorkspaceSlots,sms,0,view.workspaceStride);
        if(!sizing.slots){error="boundaryLayer workspace has no compiled capacity or its byte count is not representable";return false;}
        if(cooperativeWall){
            if(!b.surfaceProbePlanReady){
                using Two=MaterialSurfaceRootTeam<2>;using Three=MaterialSurfaceRootTeam<3>;
                if(!b.fault.check(ugkp::gpu::residentProbeBlocks(
                        evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Two>,Two::width,
                        sizeof(Two::Storage<SurfacePhysicsResult>),sms,b.surfaceProbeResident2),"query two-round root residency")
                    ||!b.fault.check(ugkp::gpu::residentProbeBlocks(
                        evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Three>,Three::width,
                        sizeof(Three::Storage<SurfacePhysicsResult>),sms,b.surfaceProbeResident3),"query three-round root residency"))return b.failure(error);
                b.surfaceProbePlanReady=true;
            }
            // Extra root probes use otherwise idle subteams. With enough
            // independent faces, keep the ordinary one-warp traversal.
            if(sizing.slots<=b.surfaceProbeResident3)surfaceProbeDepth=3;
            else if(sizing.slots<=b.surfaceProbeResident2)surfaceProbeDepth=2;
            surfaceProbes=(1<<surfaceProbeDepth)-1;
            if(view.workspaceStride>std::numeric_limits<std::size_t>::max()/std::size_t(surfaceProbes)){
                error="boundaryLayer probe workspace byte count is not representable";return false;
            }
            sizing=ugkwp::resolveBoundaryLayerWorkspace(int(count),p.wallWorkspaceSlots,sms,0,view.workspaceStride*surfaceProbes);
            if(!sizing.slots){error="boundaryLayer probe workspace byte count is not representable";return false;}
        }
        if(!b.allocate(b.wallWorkspace,sizing.bytes))return b.failure(error);
        view.workers=sizing.slots;view.wallWorkspaces=b.wallWorkspace.data();
    }
    if(cooperativeWall){
        if(!b.allocate(b.surfaceTeamOutput,std::size_t(view.workers)*surfaceProbes))return b.failure(error);
        view.wallTeamOutput=b.surfaceTeamOutput.data();view.quietPhysics=b.quietSurfacePhysics.data();
    }
    if(!b.upload(b.surfaceInput,inputs)||!b.upload(b.surfaceCells,solidCells)||!b.upload(b.surfaceActive,active)
        ||!b.allocate(b.surfacePhysics,count)||!b.allocate(b.surfaceWall,count)||!b.allocate(b.surfaceStatus,count))return b.failure(error);
    view.inputs=b.surfaceInput.data();view.solidCells=b.surfaceCells.data();view.solid=candidate?b.next.data():b.current.data();
    view.volumes=candidate?b.nextVolumes.data():b.volumes.data();view.materialRate=useRates?b.rates.data():nullptr;
    view.active=b.surfaceActive.data();view.physicsOut=b.surfacePhysics.data();view.wallOut=b.surfaceWall.data();view.status=b.surfaceStatus.data();
    b.operation(h.time);
    if(count&&cooperativeWall){
        if(surfaceProbeDepth==3){
            using Team=MaterialSurfaceRootTeam<3>;
            evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),b.stream.get()>>>(view,b.devicePhysics.data());
        }else if(surfaceProbeDepth==2){
            using Team=MaterialSurfaceRootTeam<2>;
            evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),b.stream.get()>>>(view,b.devicePhysics.data());
        }else evaluateMaterialSurfaceTeamKernel<<<view.workers,ugkwp::gaswall::detail::wallWarpWidth,0,b.stream.get()>>>(view,b.devicePhysics.data());
    }else if(count){
        ugkp::gpu::ScalarWorkLaunch launch;
        if(!b.fault.check(ugkp::gpu::scalarWorkLaunch(evaluateMaterialSurfaceKernel<MaterialSurfaceView>,view.workers,launch),
            "schedule independent material surface workers"))return b.failure(error);
        evaluateMaterialSurfaceKernel<<<launch.blocks,launch.threads,0,b.stream.get()>>>(view,b.devicePhysics.data());
    }
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
