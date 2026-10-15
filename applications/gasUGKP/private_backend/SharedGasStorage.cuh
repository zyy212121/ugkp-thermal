#pragma once
// Application-owned allocation only. All physical evaluations remain common.
// Include after the resident allocation/copy/error helpers.
#include <limits>
#include <type_traits>
#include <utility>
template<class State,class=void>struct SharedGasGraphStorage:std::false_type{};
template<class State>struct SharedGasGraphStorage<State,std::void_t<decltype(std::declval<State>().gasGraphDt)>>:std::true_type{};
template<class State,class=void>struct SharedGasSnapshotStorage:std::false_type{};
template<class State>struct SharedGasSnapshotStorage<State,std::void_t<decltype(std::declval<State>().gasTrial.trial.valid),decltype(std::declval<State>().gasTrial.interval.valid)>>:std::true_type{};
template<class State>void invalidateSharedGasConfiguration(State* state,bool recapture=true)
{



    if constexpr(SharedGasGraphStorage<State>::value)
        if(recapture)state->gasGraphDt=std::numeric_limits<double>::quiet_NaN();
    if constexpr(SharedGasSnapshotStorage<State>::value)
        state->gasTrial.trial.valid=state->gasTrial.interval.valid=false;
    if constexpr(ugkwp::GasBoundaryLayerCapability<State>::value)
        state->gasBoundaryLayer.preparedFirstStage=false;
}
template<class T>
int releaseSharedGasPointer(const T*& pointer)
{
    if(pointer){const auto error=cudaFree(const_cast<T*>(pointer));
        if(error!=cudaSuccess){setLastError("free shared gas storage",error);return 1;}pointer=nullptr;}
    return 0;
}
template<class T>
int releaseSharedGasPointer(T*& pointer)
{
    if(pointer){const auto error=cudaFree(pointer);
        if(error!=cudaSuccess){setLastError("free shared gas storage",error);return 1;}pointer=nullptr;}
    return 0;
}
template<class Real,int Ns>
int releaseSharedGasSpecies(ugkwp::GasSpeciesState<Real,Ns>& state)
{
    int result=0;
#define RELEASE_SHARED(Name) result|=releaseSharedGasPointer(state.Name)
    RELEASE_SHARED(rho); RELEASE_SHARED(initial); RELEASE_SHARED(flux);
    RELEASE_SHARED(gradX); RELEASE_SHARED(gradY); RELEASE_SHARED(gradZ);
    RELEASE_SHARED(limiter); RELEASE_SHARED(boundaryMassFraction);
    RELEASE_SHARED(compositionBoundaryFixed); RELEASE_SHARED(positivityScale);
    RELEASE_SHARED(diffusivity); RELEASE_SHARED(soundSpeed); RELEASE_SHARED(heatCapacity);
    RELEASE_SHARED(gasConstant); RELEASE_SHARED(cellStatus); RELEASE_SHARED(faceStatus);
    RELEASE_SHARED(chemistryStatus);
    RELEASE_SHARED(thermo.species); RELEASE_SHARED(thermo.coefficients);
    RELEASE_SHARED(thermo.elementComposition); RELEASE_SHARED(mechanism.reactions);
    RELEASE_SHARED(mechanism.reactants); RELEASE_SHARED(mechanism.products);
    RELEASE_SHARED(mechanism.efficiencies); RELEASE_SHARED(mechanism.stoichiometricBasis);
#undef RELEASE_SHARED
    if(!result)state=ugkwp::GasSpeciesState<Real,Ns>{};
    return result;
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
