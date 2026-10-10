#pragma once
// Application-owned allocation only. All physical evaluations remain common.
// Include after the resident allocation/copy/error helpers.
template<class T>
void releaseSharedGasPointer(const T*& pointer)
{
    if(pointer) { cudaFree(const_cast<T*>(pointer)); pointer=nullptr; }
}
template<class T>
void releaseSharedGasPointer(T*& pointer)
{
    if(pointer) { cudaFree(pointer); pointer=nullptr; }
}
template<class Real,int Ns>
void releaseSharedGasSpecies(ugkwp::GasSpeciesState<Real,Ns>& state)
{
#define RELEASE_SHARED(Name) releaseSharedGasPointer(state.Name)
    RELEASE_SHARED(rho); RELEASE_SHARED(initial); RELEASE_SHARED(flux);
    RELEASE_SHARED(gradX); RELEASE_SHARED(gradY); RELEASE_SHARED(gradZ);
    RELEASE_SHARED(limiter); RELEASE_SHARED(boundaryMassFraction);
    RELEASE_SHARED(compositionBoundaryFixed); RELEASE_SHARED(positivityScale);
    RELEASE_SHARED(diffusivity); RELEASE_SHARED(soundSpeed); RELEASE_SHARED(heatCapacity);
    RELEASE_SHARED(gasConstant); RELEASE_SHARED(cellStatus); RELEASE_SHARED(faceStatus);
    RELEASE_SHARED(chemistryAudit); RELEASE_SHARED(chemistryStatus);
    RELEASE_SHARED(thermo.species); RELEASE_SHARED(thermo.coefficients);
    RELEASE_SHARED(thermo.elementComposition); RELEASE_SHARED(mechanism.reactions);
    RELEASE_SHARED(mechanism.reactants); RELEASE_SHARED(mechanism.products);
    RELEASE_SHARED(mechanism.efficiencies); RELEASE_SHARED(mechanism.stoichiometricBasis);
#undef RELEASE_SHARED
    state=ugkwp::GasSpeciesState<Real,Ns>{};
}
template<class T>
int allocateSharedGasZero(T*& pointer,std::size_t count)
{
    if(allocate(pointer,count,"allocate shared gas storage")) return 1;
    if(count && cudaMemset(pointer,0,count*sizeof(T))!=cudaSuccess)
    { setLastErrorText("failed to initialize shared gas storage"); return 1; }
    return 0;
}
template<class T>
int uploadSharedGasTable(const T*& pointer,const T* source,std::size_t count)
{
    T* allocation=nullptr;
    const int rc=allocate(allocation,count,"allocate immutable gas model table");
    pointer=allocation;
    return rc?rc:copyToDevice(allocation,source,count,"upload immutable gas model table");
}
inline int syncSharedGasSpecies(DeviceState* state)
{
    const cudaError_t err=cudaMemcpy
    (
        reinterpret_cast<char*>(state->deviceState)+offsetof(DeviceState,gasSpecies),
        &state->gasSpecies,sizeof(state->gasSpecies),cudaMemcpyHostToDevice
    );
    if(err!=cudaSuccess){setLastError("upload shared gas view",err);return 1;}
    return 0;
}
