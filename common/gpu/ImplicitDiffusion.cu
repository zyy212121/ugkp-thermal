#include <cstring>
#include "ImplicitDiffusion.H"
#include "ThermalFieldOperators.H"
#include "ThermalGradientKernels.cuh"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA implicit diffusion: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t n){if(n<=capacity)return;T* next=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&next),n*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=n;}
    void upload(const std::vector<T>& input,PinnedTransferQueue& queue){reserve(input.size());queue.upload(ptr,input.data(),input.size()*sizeof(T));}
    void download(std::vector<T>& output,int n,PinnedTransferQueue& queue){output.resize(n);queue.download(output.data(),ptr,n*sizeof(T));}
};
__global__ void interpolateResidentConductivity(int nf,int ni,const int* owner,const int* neighbour,
    const double* cells,const double* weights,const double* boundary,const double* adjacent,
    const int* coupled,const int* adjacentCells,const double* jump,const double* boundaryDevice,bool harmonic,double* gamma){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    ScalarFaceInterpolation value;
    if(f<ni)value={cells[owner[f]],cells[neighbour[f]],weights[f],0};
    else{const int b=f-ni;
        if(coupled[b]==1)value={cells[owner[f]],adjacentCells[b]>=0?cells[adjacentCells[b]]-jump[b]:adjacent[b],weights[f],1};
        else value={coupled[b]==2?cells[owner[f]]:boundaryDevice?boundaryDevice[b]:boundary[b],0,0,2};}
    gamma[f]=thermalScalarInterpolation(value,harmonic);
}
__global__ void residentTemperatureFaces(int nf,int ni,const int* owner,const int* neighbour,
    const double* temperature,const double* weight,const int* boundaryNeighbour,
    const double* valueInternal,const double* valueBoundary,const double* jump,double* faceTemperature){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    const double own=temperature[owner[f]];
    if(f<ni)faceTemperature[f]=thermalLinearInterpolation(own,temperature[neighbour[f]],weight[f],0);
    else{const int b=f-ni,other=boundaryNeighbour[b];
        faceTemperature[f]=other>=0?thermalLinearInterpolation(own,temperature[other]-jump[b],weight[f],1)
            :valueInternal[b]*own+valueBoundary[b];}
}
__global__ void residentNormalCorrection(int nf,int ni,const int* owner,const int* neighbour,
    const double* temperature,const ThermalVector* gradient,const double* weight,const ThermalVector* area,
    const double* magnitude,const ThermalVector* direction,const int* boundaryNeighbour,
    const double* normalInternal,const double* normalBoundary,const ThermalVector* transform,double* correction){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    const int c=owner[f];const auto a=gradient[c];ThermalVector face;
    if(f<ni){const auto b=gradient[neighbour[f]];face={thermalLinearInterpolation(a.x,b.x,weight[f],0),
        thermalLinearInterpolation(a.y,b.y,weight[f],0),thermalLinearInterpolation(a.z,b.z,weight[f],0)};}
    else{const int b=f-ni,other=boundaryNeighbour[b];
        if(other>=0){const auto v=gradient[other];const auto x=transform[3*b],y=transform[3*b+1],z=transform[3*b+2];
            const ThermalVector adjacent{x.x*v.x+y.x*v.y+z.x*v.z,x.y*v.x+y.y*v.y+z.y*v.z,x.z*v.x+y.z*v.y+z.z*v.z};
            face={thermalLinearInterpolation(a.x,adjacent.x,weight[f],1),thermalLinearInterpolation(a.y,adjacent.y,weight[f],1),thermalLinearInterpolation(a.z,adjacent.z,weight[f],1)};
        }else{const auto sf=area[f];const ThermalVector normal{sf.x/magnitude[f],sf.y/magnitude[f],sf.z/magnitude[f]};
            const auto x=transform[3*b],y=transform[3*b+1],z=transform[3*b+2];
            const ThermalVector boundary{x.x*a.x+y.x*a.y+z.x*a.z,x.y*a.x+y.y*a.y+z.y*a.z,x.z*a.x+y.z*a.y+z.z*a.z};
            const double snGrad=normalInternal[b]*temperature[c]+normalBoundary[b];
            const double change=snGrad-(normal.x*boundary.x+normal.y*boundary.y+normal.z*boundary.z);
            face={boundary.x+normal.x*change,boundary.y+normal.y*change,boundary.z+normal.z*change};}}
    const auto d=direction[f];correction[f]=d.x*face.x+d.y*face.y+d.z*face.z;
}
__global__ void scatterInterfaceEnergy(int n,const int* boundaryFaces,const double* incoming,double* energy){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)energy[boundaryFaces[i]]=incoming[i];
}
__global__ void prescribeEnergyGradient(int nb,const int* selected,const double* energy,const double* area,
    const double* conductivity,double interval,double* gradientBoundary,double* valueBoundary,double* normalBoundary,const double* valueDelta,const double* deviceInterval=nullptr){
    const int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb||!selected[b])return;
    if(deviceInterval)interval=*deviceInterval;
    const double denominator=area[b]*interval*conductivity[b];
    const double gradient=energy[b]/denominator;gradientBoundary[b]=gradient;
    if(valueBoundary){valueBoundary[b]=gradient/valueDelta[b];normalBoundary[b]=gradient;}
}
__global__ void faceFactors(int nf,int ni,const double* gamma,const double* area,const double* delta,
    const double* gradientInternal,const double* gradientBoundary,const double* correction,bool corrected,
    double* upper,double* internal,double* boundary,double* flux){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    const double g=gamma[f]*area[f];
    if(f<ni)upper[f]=delta[f]*g;
    else {const int b=f-ni;internal[b]=g*gradientInternal[b];boundary[b]=(-g)*gradientBoundary[b];}
    flux[f]=corrected?g*correction[f]:0;
}
__global__ void transientCells(int n,const double* volume,const double* capacity,const double* rhs,double dt,
    bool sourceFromCapacity,double* diagonal,double* source,const double* deviceInterval=nullptr){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    if(deviceInterval)dt=*deviceInterval;
    const double weight=capacity[c]/dt;
    const double forcing=sourceFromCapacity?weight*rhs[c]:rhs[c];
    diagonal[c]=volume[c]*weight-diagonal[c];
    source[c]=(0-source[c])+volume[c]*forcing;
}
__global__ void negate(int n,double* field,bool subtractFromZero){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)field[i]=subtractFromZero?0-field[i]:-field[i];
}
__global__ void cellAssembly(int n,int ni,const int* offsets,const int* faces,const int* signs,
    const double* upper,const double* flux,const double* volume,bool corrected,double* diagonal,double* source){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double d=0,sum=0;
    for(int i=offsets[c];i<offsets[c+1];++i){const int f=faces[i];
        if(f<ni)d-=upper[f];
        if(signs[i]>0)sum+=flux[f];else sum-=flux[f];
    }
    diagonal[c]=d;
    // Preserve the native surfaceIntegrate -> divide by V -> matrix source
    // integration, including its floating-point rounding; do not cancel V.
    source[c]=corrected?0-volume[c]*(sum/volume[c]):0;
}
__global__ void matrixFaceFlux(int nf,int ni,int start,const int* owner,const int* neighbour,
    const double* upper,const double* lower,const double* internal,const double* boundary,
    const double* correction,const double* temperature,const double* neighbourTemperature,
    const int* coupled,const int* adjacentCells,const double* jump,double* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x,f=i+start;if(f>=nf)return;
    if(f<ni)output[i]=upper[f]*temperature[neighbour[f]]-lower[f]*temperature[owner[f]];
    else {const int b=f-ni;const double adjacent=adjacentCells[b]>=0?temperature[adjacentCells[b]]-jump[b]:neighbourTemperature[b];
        const double remote=coupled[b]?boundary[b]*adjacent:boundary[b];
        output[i]=internal[b]*temperature[owner[f]]-remote;}
    output[i]+=correction[f];
}
}
struct ImplicitDiffusion::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;int cells=-1,internal=-1;
    std::vector<int> owner,neighbour;
    std::vector<double> cachedVolume,cachedArea,cachedDelta,cachedWeights,cachedRhs,hostSolution,hostAnchor;bool hostSolutionCurrent=false;
    void geometry(Array<double>& target,std::vector<double>& cache,const std::vector<double>& values){
        if(cache!=values){target.upload(values,transfers);cache=values;}
    }
    // These arrays are host-authored read-only kernel inputs. Compare exact
    // bytes so changes in signed zero, NaNs or moving-mesh geometry invalidate
    // the device copy as well. Kernel-written boundary arrays are not cached.
    template<class T> void immutableInput(Array<T>& target,std::vector<T>& cache,const std::vector<T>& values){
        if(cache.size()!=values.size() || (!values.empty() &&
            std::memcmp(cache.data(),values.data(),values.size()*sizeof(T))!=0)){
            target.upload(values,transfers);cache=values;
        }
    }
    std::vector<int> cachedConductivityCoupled,cachedConductivityAdjacentCells,cachedGradientNeighbour;
    std::vector<double> cachedConductivityJump,cachedGradientWeights;
    std::vector<ThermalVector> cachedFaceArea,cachedCorrectionDirection,cachedNeighbourTransform;
    Array<int> offsets,faces,signs,deviceOwner,deviceNeighbour,coupled,fluxAdjacentCells;
    Array<double> fluxJump;
    Array<boundary::Face> jumpScheduleFaces;Array<boundary::Function> jumpScheduleFunctions;
    Array<boundary::Term> jumpScheduleTerms;Array<int> jumpScheduleSigns;Array<double> jumpScheduleTime;
    int jumpScheduleCount=0;
    Array<double> solution,anchorSolution,previousSolution,residual,neighbourTemperature,faceFlux;
    Array<double> windowSolution,windowBoundary;
    bool asymmetric=false,assemblyTopologyReady=false,residentBoundary=false,residentConductivity=false;
    Array<double> conductivityWeights,conductivityBoundary,conductivityNeighbour;
    Array<int> conductivityCoupled,conductivityAdjacentCells;
    Array<double> conductivityJump;
    Array<double> gradientWeights,valueInternal,valueBoundary,normalInternal,normalBoundary,jump,faceTemperature,boundaryValueDelta;
    Array<double> prescribedEnergy;Array<int> prescribedSelected,interfaceBoundaryFaces;
    ThermalInterfaceEnergyDeviceView interfaceEnergy[2];int interfaceEnergyOffset[2]={0,0};std::vector<int> cachedInterfaceBoundaryFaces;
    std::vector<double> cachedPrescribedEnergy;std::vector<int> cachedPrescribedSelected;
    const double* prescribedConductivity=nullptr;double prescribedInterval=0;
    Array<int> gradientNeighbour;
    Array<ThermalVector> faceArea,correctionDirection,neighbourTransform,gradient;

    cudaEvent_t inputReady=nullptr;
    void waitFor(void* producer){
        if(!inputReady)checked(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));
        checked(cudaEventRecord(inputReady,static_cast<cudaStream_t>(producer)));
        checked(cudaStreamWaitEvent(stream,inputReady,0));
    }
    Array<double> volume,gamma,area,delta,gradientInternal,gradientBoundary,correction;
    Array<double> capacity,rhs,diagonal,source,upper,lower,boundaryDiagonal,boundarySource,flux;
    struct PreparedAssembly {
        int faces=0;
        bool valid=false,corrected=false,harmonic=false,transient=false,sourceFromCapacity=false;
        const double *conductivity=nullptr,*boundaryConductivity=nullptr,*temperature=nullptr,*capacity=nullptr,*rhs=nullptr;
        double interval=0;
    } prepared;
    void assemblePrepared(cudaStream_t target,const double* deviceInterval=nullptr){
        const int n=cells,ni=internal,nf=prepared.faces,nb=nf-ni;
        if(prepared.conductivity&&nf)interpolateResidentConductivity<<<(nf+127)/128,128,0,target>>>(nf,ni,deviceOwner.ptr,deviceNeighbour.ptr,
            prepared.conductivity,conductivityWeights.ptr,conductivityBoundary.ptr,conductivityNeighbour.ptr,conductivityCoupled.ptr,conductivityAdjacentCells.ptr,conductivityJump.ptr,prepared.boundaryConductivity,prepared.harmonic,gamma.ptr);
        if(prepared.temperature){
            if(nb&&prescribedConductivity)prescribeEnergyGradient<<<(nb+127)/128,128,0,target>>>(nb,prescribedSelected.ptr,prescribedEnergy.ptr,
                area.ptr+ni,prescribedConductivity,prescribedInterval,gradientBoundary.ptr,valueBoundary.ptr,normalBoundary.ptr,boundaryValueDelta.ptr,deviceInterval);
            if(nf)residentTemperatureFaces<<<(nf+127)/128,128,0,target>>>(nf,ni,deviceOwner.ptr,deviceNeighbour.ptr,
                prepared.temperature,gradientWeights.ptr,gradientNeighbour.ptr,valueInternal.ptr,valueBoundary.ptr,jump.ptr,faceTemperature.ptr);
            if(prepared.corrected&&n)thermalGradientDetail::cellGradient<<<(n+127)/128,128,0,target>>>(n,offsets.ptr,faces.ptr,signs.ptr,
                faceTemperature.ptr,faceArea.ptr,volume.ptr,gradient.ptr);
            if(prepared.corrected&&nf)residentNormalCorrection<<<(nf+127)/128,128,0,target>>>(nf,ni,deviceOwner.ptr,deviceNeighbour.ptr,
                prepared.temperature,gradient.ptr,gradientWeights.ptr,faceArea.ptr,area.ptr,correctionDirection.ptr,
                gradientNeighbour.ptr,normalInternal.ptr,normalBoundary.ptr,neighbourTransform.ptr,correction.ptr);
        }else if(nb&&prescribedConductivity)prescribeEnergyGradient<<<(nb+127)/128,128,0,target>>>(nb,prescribedSelected.ptr,prescribedEnergy.ptr,
            area.ptr+ni,prescribedConductivity,prescribedInterval,gradientBoundary.ptr,nullptr,nullptr,nullptr,deviceInterval);
        if(nf)faceFactors<<<(nf+127)/128,128,0,target>>>(nf,ni,gamma.ptr,area.ptr,delta.ptr,gradientInternal.ptr,gradientBoundary.ptr,correction.ptr,prepared.corrected,upper.ptr,boundaryDiagonal.ptr,boundarySource.ptr,flux.ptr);
        if(n)cellAssembly<<<(n+127)/128,128,0,target>>>(n,ni,offsets.ptr,faces.ptr,signs.ptr,upper.ptr,flux.ptr,volume.ptr,prepared.corrected,diagonal.ptr,source.ptr);
        if(prepared.transient){
            if(n)transientCells<<<(n+127)/128,128,0,target>>>(n,volume.ptr,prepared.capacity,prepared.rhs,
                prepared.interval,prepared.sourceFromCapacity,diagonal.ptr,source.ptr,deviceInterval);
            if(ni)negate<<<(ni+127)/128,128,0,target>>>(ni,upper.ptr,false);
            if(nb){negate<<<(nb+127)/128,128,0,target>>>(nb,boundaryDiagonal.ptr,true);
                negate<<<(nb+127)/128,128,0,target>>>(nb,boundarySource.ptr,true);}
            if(nf)negate<<<(nf+127)/128,128,0,target>>>(nf,flux.ptr,false);
        }
    }
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}if(inputReady)cudaEventDestroy(inputReady);}
};
ImplicitDiffusion::ImplicitDiffusion():data_(new Implementation){}
ImplicitDiffusion::~ImplicitDiffusion()=default;
ExecutionGraphSignature ImplicitDiffusion::residentExecutionSignature()const{
    const auto& d=*data_;ExecutionGraphSignature key;key.add(this);
    for(const void* p:{
        static_cast<const void*>(d.offsets.ptr),static_cast<const void*>(d.faces.ptr),static_cast<const void*>(d.signs.ptr),
        static_cast<const void*>(d.deviceOwner.ptr),static_cast<const void*>(d.deviceNeighbour.ptr),static_cast<const void*>(d.coupled.ptr),
        static_cast<const void*>(d.fluxAdjacentCells.ptr),static_cast<const void*>(d.fluxJump.ptr),static_cast<const void*>(d.solution.ptr),
        static_cast<const void*>(d.previousSolution.ptr),static_cast<const void*>(d.residual.ptr),static_cast<const void*>(d.neighbourTemperature.ptr),
        static_cast<const void*>(d.faceFlux.ptr),static_cast<const void*>(d.conductivityWeights.ptr),static_cast<const void*>(d.conductivityBoundary.ptr),
        static_cast<const void*>(d.conductivityNeighbour.ptr),static_cast<const void*>(d.conductivityCoupled.ptr),static_cast<const void*>(d.conductivityAdjacentCells.ptr),
        static_cast<const void*>(d.conductivityJump.ptr),static_cast<const void*>(d.gradientWeights.ptr),static_cast<const void*>(d.valueInternal.ptr),
        static_cast<const void*>(d.valueBoundary.ptr),static_cast<const void*>(d.normalInternal.ptr),static_cast<const void*>(d.normalBoundary.ptr),
        static_cast<const void*>(d.jump.ptr),static_cast<const void*>(d.faceTemperature.ptr),static_cast<const void*>(d.boundaryValueDelta.ptr),
        static_cast<const void*>(d.prescribedEnergy.ptr),static_cast<const void*>(d.prescribedSelected.ptr),static_cast<const void*>(d.gradientNeighbour.ptr),
        static_cast<const void*>(d.faceArea.ptr),static_cast<const void*>(d.correctionDirection.ptr),static_cast<const void*>(d.neighbourTransform.ptr),
        static_cast<const void*>(d.gradient.ptr),static_cast<const void*>(d.volume.ptr),static_cast<const void*>(d.gamma.ptr),
        static_cast<const void*>(d.area.ptr),static_cast<const void*>(d.delta.ptr),static_cast<const void*>(d.gradientInternal.ptr),
        static_cast<const void*>(d.gradientBoundary.ptr),static_cast<const void*>(d.correction.ptr),static_cast<const void*>(d.capacity.ptr),
        static_cast<const void*>(d.rhs.ptr),static_cast<const void*>(d.diagonal.ptr),static_cast<const void*>(d.source.ptr),
        static_cast<const void*>(d.upper.ptr),static_cast<const void*>(d.lower.ptr),static_cast<const void*>(d.boundaryDiagonal.ptr),
        static_cast<const void*>(d.boundarySource.ptr),static_cast<const void*>(d.flux.ptr)
    })key.add(p);
    key.add(d.cells);key.add(d.internal);key.add(d.owner.size());
    key.add(d.asymmetric);key.add(d.residentBoundary);key.add(d.residentConductivity);
    key.add(d.prescribedConductivity);key.add(d.prescribedInterval);
    key.add(d.prepared.faces);key.add(d.prepared.valid);key.add(d.prepared.corrected);
    key.add(d.prepared.harmonic);key.add(d.prepared.transient);key.add(d.prepared.sourceFromCapacity);
    key.add(d.prepared.conductivity);key.add(d.prepared.boundaryConductivity);key.add(d.prepared.temperature);
    key.add(d.prepared.capacity);key.add(d.prepared.rhs);key.add(d.prepared.interval);
    return key;
}
void ImplicitDiffusion::assembleResident(const DiffusionInput& input,const DiffusionTransientInput* transient){
    auto& d=*data_;const int n=input.cells,ni=input.internalFaces,nf=input.owner.size(),nb=nf-ni;
    d.asymmetric=false;d.residentBoundary=input.deviceTemperature;d.residentConductivity=input.deviceConductivity;
    if(!d.assemblyTopologyReady||d.cells!=n||d.internal!=ni||d.owner!=input.owner||d.neighbour!=input.neighbour){
        std::vector<std::vector<std::pair<int,int>>> rows(n);
        for(int f=0;f<nf;++f){rows[input.owner[f]].push_back({f,1});if(f<ni)rows[input.neighbour[f]].push_back({f,-1});}
        std::vector<int> offsets(1,0),faces,signs;
        for(const auto& row:rows){for(auto entry:row){faces.push_back(entry.first);signs.push_back(entry.second);}offsets.push_back(faces.size());}
        d.offsets.upload(offsets,d.transfers);d.faces.upload(faces,d.transfers);d.signs.upload(signs,d.transfers);
        d.deviceOwner.upload(input.owner,d.transfers);d.deviceNeighbour.upload(input.neighbour,d.transfers);
        d.transfers.finish();d.cells=n;d.internal=ni;d.owner=input.owner;d.neighbour=input.neighbour;d.assemblyTopologyReady=true;
    }
    d.geometry(d.volume,d.cachedVolume,input.volume);d.geometry(d.area,d.cachedArea,input.area);d.geometry(d.delta,d.cachedDelta,input.delta);
    if(input.deviceConductivity){
        d.waitFor(input.conductivityStream);d.gamma.reserve(nf);
        d.geometry(d.conductivityWeights,d.cachedWeights,input.conductivityWeights);
        d.conductivityBoundary.upload(input.conductivityBoundary,d.transfers);
        d.conductivityNeighbour.upload(input.conductivityNeighbour,d.transfers);
        d.immutableInput(d.conductivityCoupled,d.cachedConductivityCoupled,input.conductivityCoupled);
        d.immutableInput(d.conductivityAdjacentCells,d.cachedConductivityAdjacentCells,input.conductivityAdjacentCells);d.immutableInput(d.conductivityJump,d.cachedConductivityJump,input.conductivityJump);
    }else d.gamma.upload(input.gamma,d.transfers);
    d.gradientInternal.upload(input.gradientInternal,d.transfers);d.gradientBoundary.upload(input.gradientBoundary,d.transfers);
    if(input.deviceTemperature){
        d.immutableInput(d.gradientWeights,d.cachedGradientWeights,input.gradientWeights);d.valueInternal.upload(input.valueInternal,d.transfers);
        d.valueBoundary.upload(input.valueBoundary,d.transfers);d.normalInternal.upload(input.normalInternal,d.transfers);
        d.normalBoundary.upload(input.normalBoundary,d.transfers);d.jump.upload(input.jump,d.transfers);
        d.boundaryValueDelta.upload(input.boundaryValueDelta,d.transfers);
        d.immutableInput(d.gradientNeighbour,d.cachedGradientNeighbour,input.gradientNeighbour);d.immutableInput(d.faceArea,d.cachedFaceArea,input.faceArea);
        d.immutableInput(d.correctionDirection,d.cachedCorrectionDirection,input.correctionDirection);d.immutableInput(d.neighbourTransform,d.cachedNeighbourTransform,input.neighbourTransform);
        d.gradient.reserve(n);d.faceTemperature.reserve(nf);d.correction.reserve(nf);
    }else d.correction.upload(input.correction,d.transfers);
    d.diagonal.reserve(n);d.source.reserve(n);d.upper.reserve(ni);d.boundaryDiagonal.reserve(nb);d.boundarySource.reserve(nb);d.flux.reserve(nf);
    if(transient){
        if(transient->deviceCapacity||transient->deviceRhs)d.waitFor(transient->readyStream);
        if(!transient->deviceCapacity)d.capacity.upload(*transient->capacity,d.transfers);
        if(!transient->deviceRhs)d.geometry(d.rhs,d.cachedRhs,*transient->rhs);
    }
    d.solution.reserve(n);d.residual.reserve(n);d.previousSolution.reserve(n);
    d.prepared={nf,true,input.corrected,input.harmonicConductivity,transient!=nullptr,
        transient?transient->sourceFromCapacity:false,input.deviceConductivity,input.deviceBoundaryConductivity,input.deviceTemperature,
        transient?(transient->deviceCapacity?transient->deviceCapacity:d.capacity.ptr):nullptr,
        transient?(transient->deviceRhs?transient->deviceRhs:d.rhs.ptr):nullptr,transient?transient->dt:0};
    d.assemblePrepared(d.stream);
}
void ImplicitDiffusion::enqueueResidentAssembly(void* executionStream,const double* deviceInterval){
    auto& d=*data_;
    if(!d.prepared.valid)throw std::logic_error("resident diffusion must be configured before enqueue");
    d.assemblePrepared(static_cast<cudaStream_t>(executionStream),deviceInterval);
}
void ImplicitDiffusion::assemble(const DiffusionInput& input,DiffusionMatrix& output,const DiffusionTransientInput* transient){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    assembleResident(input,transient);
    const int n=input.cells,ni=input.internalFaces,nf=input.owner.size(),nb=nf-ni;
    d.diagonal.download(output.diagonal,n,d.transfers);d.source.download(output.source,n,d.transfers);d.upper.download(output.upper,ni,d.transfers);
    d.boundaryDiagonal.download(output.boundaryDiagonal,nb,d.transfers);d.boundarySource.download(output.boundarySource,nb,d.transfers);d.flux.download(output.fluxCorrection,nf,d.transfers);
    d.transfers.finish();
}
SparseDeviceLduValues ImplicitDiffusion::deviceValues()const{
    const auto& d=*data_;return {d.diagonal.ptr,d.upper.ptr,d.asymmetric?d.lower.ptr:d.upper.ptr,
        d.source.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.stream,d.residentBoundary?d.jump.ptr:nullptr};
}
double* ImplicitDiffusion::deviceSolution(){data_->hostSolutionCurrent=false;return data_->solution.ptr;}
const double* ImplicitDiffusion::deviceTemperature()const{return data_->solution.ptr;}
double* ImplicitDiffusion::deviceResidual(){return data_->residual.ptr;}
const double* ImplicitDiffusion::devicePreviousSolution()const{return data_->previousSolution.ptr;}
void ImplicitDiffusion::snapshotSolution(){
    auto& d=*data_;d.previousSolution.reserve(d.cells);enqueueResidentSnapshot(d.stream);
}
void ImplicitDiffusion::enqueueResidentSnapshot(void* stream){
    auto& d=*data_;if(d.cells)checked(cudaMemcpyAsync(d.previousSolution.ptr,d.solution.ptr,d.cells*sizeof(double),cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream)));
}
void ImplicitDiffusion::prepareResidentWindowSnapshot(){
    auto& d=*data_;d.windowSolution.reserve(d.cells);d.windowBoundary.reserve(d.owner.size()-d.internal);
}
void ImplicitDiffusion::enqueueResidentWindowSnapshot(void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);const int nb=d.owner.size()-d.internal;
    if(d.cells)checked(cudaMemcpyAsync(d.windowSolution.ptr,d.solution.ptr,d.cells*sizeof(double),cudaMemcpyDeviceToDevice,stream));
    if(nb)checked(cudaMemcpyAsync(d.windowBoundary.ptr,d.faceTemperature.ptr+d.internal,nb*sizeof(double),cudaMemcpyDeviceToDevice,stream));
}
void ImplicitDiffusion::enqueueResidentWindowRestore(void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);const int nb=d.owner.size()-d.internal;
    if(d.cells)checked(cudaMemcpyAsync(d.solution.ptr,d.windowSolution.ptr,d.cells*sizeof(double),cudaMemcpyDeviceToDevice,stream));
    if(nb)checked(cudaMemcpyAsync(d.faceTemperature.ptr+d.internal,d.windowBoundary.ptr,nb*sizeof(double),cudaMemcpyDeviceToDevice,stream));
}
namespace {
__global__ void advanceBoundaryScheduleTime(double* time,const double* interval){if(!threadIdx.x)*time+=*interval;}
__global__ void evaluateBoundaryJumpSchedule(boundary::View program,const int* signs,const double* time,double* jump,double* fluxJump){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=program.count)return;
    const int function=program.faces[face].function[boundary::Temperature];if(function<0)return;
    const double value=signs[face]*boundary::evaluate(program,function,*time,0);
    jump[face]=value;if(fluxJump)fluxJump[face]=value;
}
}
void ImplicitDiffusion::configureBoundaryJumpSchedule(const boundary::Definition& definition,const std::vector<int>& signs,double initialTime){
    auto& d=*data_;boundary::validate(definition);
    if(definition.faces.size()!=signs.size()||definition.faces.size()!=std::size_t(d.owner.size()-d.internal))
        throw std::runtime_error("solid jump schedule extent mismatch");
    d.jumpScheduleCount=definition.faces.size();
    d.jumpScheduleFaces.upload(definition.faces,d.transfers);d.jumpScheduleFunctions.upload(definition.functions,d.transfers);
    d.jumpScheduleTerms.upload(definition.terms,d.transfers);d.jumpScheduleSigns.upload(signs,d.transfers);
    d.jumpScheduleTime.upload(std::vector<double>{initialTime},d.transfers);d.transfers.finish();
}
void ImplicitDiffusion::enqueueBoundaryJumpSchedule(void* stream,const double* interval){
    auto& d=*data_;if(!d.jumpScheduleCount)return;
    const auto execution=static_cast<cudaStream_t>(stream);
    advanceBoundaryScheduleTime<<<1,1,0,execution>>>(d.jumpScheduleTime.ptr,interval);
    const boundary::View view{d.jumpScheduleFaces.ptr,d.jumpScheduleFunctions.ptr,d.jumpScheduleTerms.ptr,d.jumpScheduleCount};
    evaluateBoundaryJumpSchedule<<<(d.jumpScheduleCount+127)/128,128,0,execution>>>(view,d.jumpScheduleSigns.ptr,d.jumpScheduleTime.ptr,d.jump.ptr,d.fluxJump.ptr);
}
void ImplicitDiffusion::enqueueResidentInterfaceEnergy(void* stream){
    auto& d=*data_;
    for(int segment=0;segment<2;++segment){const auto& view=d.interfaceEnergy[segment];if(view.enabled&&view.faces)
        scatterInterfaceEnergy<<<(view.faces+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(view.faces,
            d.interfaceBoundaryFaces.ptr+d.interfaceEnergyOffset[segment],view.energy,d.prescribedEnergy.ptr);}
}
void ImplicitDiffusion::prescribeBoundaryEnergy(const std::vector<double>& energy,const std::vector<int>& selected,
    double interval,const double* conductivity,void* propertyStream){
    auto& d=*data_;std::vector<double> hostEnergy(energy);
    // Device segments replace their physical input slots; never upload their
    // downloaded audit mirrors merely to overwrite them with the same ledger.
    for(int segment=0;segment<2;++segment){const auto& view=d.interfaceEnergy[segment];if(view.enabled)for(int i=0;i<view.faces;++i)
        hostEnergy[d.cachedInterfaceBoundaryFaces[d.interfaceEnergyOffset[segment]+i]]=0;}
    d.geometry(d.prescribedEnergy,d.cachedPrescribedEnergy,hostEnergy);
    if(d.cachedPrescribedSelected!=selected){d.prescribedSelected.upload(selected,d.transfers);d.cachedPrescribedSelected=selected;}
    for(int segment=0;segment<2;++segment){const auto& view=d.interfaceEnergy[segment];if(view.enabled&&view.faces){
        d.waitFor(view.producerStream);
        scatterInterfaceEnergy<<<(view.faces+127)/128,128,0,d.stream>>>(view.faces,
            d.interfaceBoundaryFaces.ptr+d.interfaceEnergyOffset[segment],view.energy,d.prescribedEnergy.ptr);
    }}
    d.prescribedConductivity=conductivity;d.prescribedInterval=interval;if(conductivity)d.waitFor(propertyStream);
}
void ImplicitDiffusion::setInterfaceEnergy(const std::vector<int>& boundaryFaces,const ThermalInterfaceEnergyDeviceView& view,
    const ThermalInterfaceEnergyDeviceView& appended){
    auto& d=*data_;const ThermalInterfaceEnergyDeviceView next[2]={view,appended};
    const int offsets[2]={0,static_cast<int>(boundaryFaces.size())-appended.faces};
    bool changed=d.cachedInterfaceBoundaryFaces!=boundaryFaces;
    for(int i=0;i<2;++i)changed=changed||d.interfaceEnergy[i].enabled!=next[i].enabled
        ||d.interfaceEnergy[i].faces!=next[i].faces||d.interfaceEnergyOffset[i]!=offsets[i];
    if(changed)d.cachedPrescribedEnergy.clear();
    d.interfaceEnergy[0]=view;d.interfaceEnergy[1]=appended;d.interfaceEnergyOffset[0]=offsets[0];d.interfaceEnergyOffset[1]=offsets[1];
    for(const auto& input:d.interfaceEnergy)if(input.enabled&&(input.faces<0||input.faces>static_cast<int>(boundaryFaces.size())))
        throw std::runtime_error("solid interface device energy map is incomplete");
    if(view.enabled&&appended.enabled&&view.faces+appended.faces>static_cast<int>(boundaryFaces.size()))
        throw std::runtime_error("solid interface device energy segments overlap");
    if((view.enabled||appended.enabled)&&d.cachedInterfaceBoundaryFaces!=boundaryFaces){d.interfaceBoundaryFaces.upload(boundaryFaces,d.transfers);d.cachedInterfaceBoundaryFaces=boundaryFaces;}
}
void ImplicitDiffusion::refreshBoundaryTemperature(){enqueueResidentBoundaryTemperature(data_->stream);}
void ImplicitDiffusion::enqueueResidentBoundaryTemperature(void* stream){
    auto& d=*data_;if(!d.residentBoundary)return;const int nf=d.owner.size();
    if(nf)residentTemperatureFaces<<<(nf+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(nf,d.internal,d.deviceOwner.ptr,d.deviceNeighbour.ptr,
        d.solution.ptr,d.gradientWeights.ptr,d.gradientNeighbour.ptr,d.valueInternal.ptr,d.valueBoundary.ptr,d.jump.ptr,d.faceTemperature.ptr);
}
void ImplicitDiffusion::initializeBoundaryTemperature(const std::vector<double>& temperature){
    auto& d=*data_;const int nb=d.owner.size()-d.internal;
    if(!d.residentBoundary||temperature.size()!=static_cast<std::size_t>(nb))throw std::invalid_argument("initial resident boundary temperature extent mismatch");
    if(nb)d.transfers.upload(d.faceTemperature.ptr+d.internal,temperature.data(),nb*sizeof(double));
}
const double* ImplicitDiffusion::deviceBoundaryTemperature()const{
    const auto& d=*data_;return d.residentBoundary&&d.faceTemperature.ptr?d.faceTemperature.ptr+d.internal:nullptr;
}
bool ImplicitDiffusion::residentIteration()const{return data_->residentBoundary&&data_->residentConductivity;}
void ImplicitDiffusion::uploadSolution(const double* input,int count){
    auto& d=*data_;const bool current=d.hostSolutionCurrent&&d.hostSolution.size()==static_cast<std::size_t>(count)
        &&std::equal(d.hostSolution.begin(),d.hostSolution.end(),input);
    d.solution.reserve(count);d.residual.reserve(count);d.anchorSolution.reserve(count);
    if(!current){
        if(d.hostAnchor.size()==static_cast<std::size_t>(count)&&std::equal(d.hostAnchor.begin(),d.hostAnchor.end(),input)){
            if(count)checked(cudaMemcpyAsync(d.solution.ptr,d.anchorSolution.ptr,count*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
        }else d.transfers.upload(d.solution.ptr,input,count*sizeof(double));
    }
    if(count){d.hostAnchor.assign(input,input+count);checked(cudaMemcpyAsync(d.anchorSolution.ptr,d.solution.ptr,count*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));}
    else d.hostAnchor.clear();
    d.hostSolutionCurrent=current;
}
void ImplicitDiffusion::copySolution(const double* input,int count,void* stream){
    auto& d=*data_;d.hostSolutionCurrent=false;d.waitFor(stream);d.solution.reserve(count);d.residual.reserve(count);
    if(count)checked(cudaMemcpyAsync(d.solution.ptr,input,count*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
}
void ImplicitDiffusion::downloadSolution(double* output,int count){
    auto& d=*data_;d.transfers.download(output,d.solution.ptr,count*sizeof(double));d.transfers.finish();d.hostSolution.assign(output,output+count);d.hostSolutionCurrent=true;
}
void ImplicitDiffusion::downloadResidual(double* output,int count){
    auto& d=*data_;d.transfers.download(output,d.residual.ptr,count*sizeof(double));d.transfers.finish();
}
void ImplicitDiffusion::downloadBoundaryState(std::vector<double>& temperature,std::vector<double>& gradient){
    auto& d=*data_;if(!d.residentBoundary){temperature.clear();gradient.clear();return;}
    const int n=d.owner.size()-d.internal;temperature.resize(n);gradient.resize(n);
    if(n){d.transfers.download(temperature.data(),d.faceTemperature.ptr+d.internal,n*sizeof(double));
        d.transfers.download(gradient.data(),d.normalBoundary.ptr,n*sizeof(double));}d.transfers.finish();
}
const double* ImplicitDiffusion::deviceBoundaryFlux()const{return data_->faceFlux.ptr;}
const double* ImplicitDiffusion::deviceVolume()const{return data_->volume.ptr;}
const double* ImplicitDiffusion::deviceBoundaryArea()const{return data_->area.ptr?data_->area.ptr+data_->internal:nullptr;}
const double* ImplicitDiffusion::deviceBoundaryGradient()const{return data_->normalBoundary.ptr;}
const double* ImplicitDiffusion::deviceBoundaryEnergy()const{return data_->prescribedEnergy.ptr;}
void ImplicitDiffusion::downloadBoundaryCoefficients(std::vector<double>& diagonal,std::vector<double>& source){
    auto& d=*data_;const int nb=d.owner.size()-d.internal;
    d.boundaryDiagonal.download(diagonal,nb,d.transfers);d.boundarySource.download(source,nb,d.transfers);d.transfers.finish();
}
void ImplicitDiffusion::retain(const DiffusionInput& input,const DiffusionMatrix& matrix){
    auto& d=*data_;d.prepared.valid=false;d.assemblyTopologyReady=false;d.residentBoundary=false;d.residentConductivity=false;d.cells=input.cells;d.internal=input.internalFaces;d.owner=input.owner;d.neighbour=input.neighbour;
    d.deviceOwner.upload(input.owner,d.transfers);d.deviceNeighbour.upload(input.neighbour,d.transfers);
    d.diagonal.upload(matrix.diagonal,d.transfers);d.source.upload(matrix.source,d.transfers);
    d.upper.upload(matrix.upper,d.transfers);d.lower.upload(matrix.lower,d.transfers);d.asymmetric=!matrix.lower.empty();
    d.boundaryDiagonal.upload(matrix.boundaryDiagonal,d.transfers);d.boundarySource.upload(matrix.boundarySource,d.transfers);
    d.flux.upload(matrix.fluxCorrection,d.transfers);
    if(matrix.fluxCorrection.empty()){d.flux.reserve(input.owner.size());checked(cudaMemsetAsync(d.flux.ptr,0,input.owner.size()*sizeof(double),d.stream));}
    d.solution.reserve(input.cells);d.residual.reserve(input.cells);
}
void ImplicitDiffusion::flux(const std::vector<double>& neighbour,const std::vector<int>& coupled,
    const std::vector<int>& adjacentCells,const std::vector<double>& jump,std::vector<double>& output,bool boundaryOnly,bool download){
    auto& d=*data_;const int nf=d.owner.size(),start=boundaryOnly?d.internal:0,count=nf-start;
    d.neighbourTemperature.upload(neighbour,d.transfers);d.coupled.upload(coupled,d.transfers);d.faceFlux.reserve(count);
    d.fluxAdjacentCells.upload(adjacentCells,d.transfers);d.fluxJump.upload(jump,d.transfers);
    if(count)matrixFaceFlux<<<(count+127)/128,128,0,d.stream>>>(nf,d.internal,start,d.deviceOwner.ptr,d.deviceNeighbour.ptr,
        d.upper.ptr,d.asymmetric?d.lower.ptr:d.upper.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.flux.ptr,
        d.solution.ptr,d.neighbourTemperature.ptr,d.coupled.ptr,d.fluxAdjacentCells.ptr,d.fluxJump.ptr,d.faceFlux.ptr);
    if(download){d.faceFlux.download(output,count,d.transfers);d.transfers.finish();}
}

void ImplicitDiffusion::enqueueResidentBoundaryFlux(void* stream){
    auto& d=*data_;const int nf=d.owner.size(),count=nf-d.internal;
    if(count)matrixFaceFlux<<<(count+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(nf,d.internal,d.internal,d.deviceOwner.ptr,d.deviceNeighbour.ptr,
        d.upper.ptr,d.asymmetric?d.lower.ptr:d.upper.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.flux.ptr,
        d.solution.ptr,d.neighbourTemperature.ptr,d.coupled.ptr,d.fluxAdjacentCells.ptr,d.fluxJump.ptr,d.faceFlux.ptr);
}

void ImplicitDiffusion::transient(const std::vector<double>& volume,const std::vector<double>& capacity,
    const std::vector<double>& rhs,double dt,bool sourceFromCapacity,DiffusionMatrix& matrix){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=volume.size(),ni=matrix.upper.size(),nl=matrix.lower.size(),nb=matrix.boundaryDiagonal.size(),nf=matrix.fluxCorrection.size();
    d.prepared.valid=false;d.cachedVolume.clear();d.cachedRhs.clear();
    d.volume.upload(volume,d.transfers);d.capacity.upload(capacity,d.transfers);d.rhs.upload(rhs,d.transfers);
    d.diagonal.upload(matrix.diagonal,d.transfers);d.source.upload(matrix.source,d.transfers);d.upper.upload(matrix.upper,d.transfers);d.lower.upload(matrix.lower,d.transfers);
    d.boundaryDiagonal.upload(matrix.boundaryDiagonal,d.transfers);d.boundarySource.upload(matrix.boundarySource,d.transfers);d.flux.upload(matrix.fluxCorrection,d.transfers);
    if(n)transientCells<<<(n+127)/128,128,0,d.stream>>>(n,d.volume.ptr,d.capacity.ptr,d.rhs.ptr,dt,sourceFromCapacity,d.diagonal.ptr,d.source.ptr);
    if(ni)negate<<<(ni+127)/128,128,0,d.stream>>>(ni,d.upper.ptr,false);
    if(nl)negate<<<(nl+127)/128,128,0,d.stream>>>(nl,d.lower.ptr,false);
    if(nb){negate<<<(nb+127)/128,128,0,d.stream>>>(nb,d.boundaryDiagonal.ptr,true);negate<<<(nb+127)/128,128,0,d.stream>>>(nb,d.boundarySource.ptr,true);}
    if(nf)negate<<<(nf+127)/128,128,0,d.stream>>>(nf,d.flux.ptr,false);
    d.diagonal.download(matrix.diagonal,n,d.transfers);d.source.download(matrix.source,n,d.transfers);d.upper.download(matrix.upper,ni,d.transfers);d.lower.download(matrix.lower,nl,d.transfers);
    d.boundaryDiagonal.download(matrix.boundaryDiagonal,nb,d.transfers);d.boundarySource.download(matrix.boundarySource,nb,d.transfers);d.flux.download(matrix.fluxCorrection,nf,d.transfers);
    d.transfers.finish();
}

}}
