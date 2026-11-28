#pragma once

#include "gpu/ParticleRadiationResponse.H"
#include "GpuParticleThermalCarry.H"
#include "wall/GpuColdWallThermalCarry.cuh"

__device__ ugkp::gpu::ParticleRadiationColdProfiles residentRadiationColdProfiles(const DeviceState& state)
{
    return {state.nFaces,state.pStuckFaceId,state.particleStuckCandidateMask,state.pColdHandle,
        state.pColdNodeSpecificEnthalpy,state.pCold2DNodeSpecificEnthalpy,
        ugkwp::coldWallCarryCaloric(state.coldWallSolidificationParameters)};
}

template<class State>
__device__ auto residentRadiationCarry(const State& state,int particle,int)
    -> decltype(GpuReal(state.pThermalEnthalpyResidual[particle]))
{
    return state.pThermalEnthalpyResidual[particle];
}

template<class State>
__device__ GpuReal residentRadiationCarry(const State&,int,long)
{
    return GpuReal(0);
}

template<class State>
__device__ auto writeResidentRadiationCarry(State& state,int particle,GpuReal carry,int)
    -> decltype(state.pThermalEnthalpyResidual[particle]=carry,void())
{
    state.pThermalEnthalpyResidual[particle]=carry;
}

template<class State>
__device__ void writeResidentRadiationCarry(State&,int,GpuReal,long)
{}

template<class Thermal>
__device__ auto setRadiationCarry(Thermal& thermal,GpuReal carry,int)
    -> decltype(thermal.specificEnthalpyResidual=carry,void())
{
    thermal.specificEnthalpyResidual=carry;
}

template<class Thermal>
__device__ void setRadiationCarry(Thermal&,GpuReal,long)
{}

template<class Thermal>
__device__ auto getRadiationCarry(const Thermal& thermal,int)
    -> decltype(GpuReal(thermal.specificEnthalpyResidual))
{
    return thermal.specificEnthalpyResidual;
}

template<class Thermal>
__device__ GpuReal getRadiationCarry(const Thermal&,long)
{
    return GpuReal(0);
}

__device__ GpuReal coldRadiationCarry(const DeviceState& state,int particle,ugkp::gpu::ColdRadiationProfile profile)
{
    return profile.nodes==Foam::gpuThermal::coldWallAxialNodeCount
        ? GpuReal(0) : residentRadiationCarry(state,particle,0);
}

__device__ ugkp::gpu::ColdRadiationSample publishColdRadiationHeat
(
    DeviceState& state,int particle,ugkp::gpu::ColdRadiationProfile profile,
    GpuReal bound,bool heating,GpuReal fraction,const ugkwp::ParticleCaloric& law
)
{
    using namespace ugkp::gpu;
    ColdRadiationSample sample;
    if (profile.nodes==Foam::gpuThermal::coldWallAxialNodeCount)
    {
        auto* nodes=const_cast<GpuReal*>(static_cast<const GpuReal*>(profile.enthalpy));
        sample=sampleColdRadiationNodes(nodes,profile.nodes,GpuReal(0),bound,heating,fraction,law,nodes);
    }
    else
    {
        auto* nodes=const_cast<float*>(static_cast<const float*>(profile.enthalpy));
        sample=sampleColdRadiationNodes(nodes,profile.nodes,coldRadiationCarry(state,particle,profile),
            bound,heating,fraction,law,nodes);
    }
    if (sample.valid)
    {
        state.pT[particle]=sample.temperature;
        writeResidentRadiationCarry(state,particle,sample.carry,0);
    }
    return sample;
}

__device__ void invalidRadiationProfile(DeviceState& state,int particle,GpuReal value)
{
    using namespace ugkp::diagnostics;
    recordFatal(state.diagnostics,makeRecord(Code::InvalidPhysicalState,30911,0,0,
        EntityKind::Particle,particle,-1,value));
}

__global__ void prepareResidentRadiationHeat(DeviceState* sp,const GpuReal* heat,
    Foam::gpuThermal::RadiationAffineTemperatureUpdate* requests)
{
    auto& state=*sp;
    if (ugkp::diagnostics::hasFatal(state.diagnostics)) return;
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=state.nCells) return;
    const GpuReal value=heat[cell];
    requests[cell]={value,GpuReal(0)};
    if (!std::isfinite(value))
    {
        using namespace ugkp::diagnostics;
        recordFatal(state.diagnostics,makeRecord(Code::NonFinite,30910,0,0,EntityKind::Cell,cell,-1,value));
    }
}

__global__ void radiationEnthalpyCapacityKernel
(DeviceState* sp,const Foam::gpuThermal::RadiationAffineTemperatureUpdate* requests)
{
    using namespace ugkp::gpu;
    auto& state=*sp;
    if (ugkp::diagnostics::hasFatal(state.diagnostics)) return;
    const auto cold=residentRadiationColdProfiles(state);
    const ugkwp::ParticleCaloric liquid;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<*state.particleCountDevice; i+=blockDim.x*gridDim.x)
    {
        const auto wallState=state.pStuck[i];
        if (!radiationParticleEligible(state.pStatus[i],wallState)) continue;
        const int cell=state.pCellId[i];
        const bool heating=requests[cell].scale>=0;
        const auto profile=cold.profile(i,wallState);
        const GpuReal capacity=profile.nodes
            ? coldProfileSpecificCapacity(profile,coldRadiationCarry(state,i,profile),
                cold.caloric.enthalpy(heating?state.TpMax:state.TpMin),heating)
            : coldProfileCapacityToBound(liquid.enthalpy(state.pT[i]),
                liquid.enthalpy(heating?state.TpMax:state.TpMin),heating,residentRadiationCarry(state,i,0));
        if (!std::isfinite(capacity))
        {
            invalidRadiationProfile(state,i,capacity);
            return;
        }
        atomicAdd(state.radiationMobileMass+cell,state.pm[i]*std::abs(capacity));
    }
}

__global__ void radiationEnthalpyApplyKernel
(
    DeviceState* sp,const Foam::gpuThermal::RadiationAffineTemperatureUpdate* requests,
    const GpuReal* publicationFraction
)
{
    using namespace ugkp::gpu;
    auto& state=*sp;
    if (ugkp::diagnostics::hasFatal(state.diagnostics)) return;
    const auto cold=residentRadiationColdProfiles(state);
    const ugkwp::ParticleCaloric liquid;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<*state.particleCountDevice; i+=blockDim.x*gridDim.x)
    {
        const auto wallState=state.pStuck[i];
        if (!radiationParticleEligible(state.pStatus[i],wallState)) continue;
        const int cell=state.pCellId[i];
        const GpuReal heat=requests[cell].scale;
        const GpuReal capacity=state.radiationMobileMass[cell];
        const GpuReal fraction=publicationFraction?publicationFraction[cell]
            :(capacity>0?fmin(GpuReal(1),std::abs(heat)/capacity):GpuReal(0));
        if (fraction==0) continue;
        const bool heating=heat>=0;
        const auto profile=cold.profile(i,wallState);
        GpuReal accepted=0;
        if (profile.nodes)
        {
            const auto sample=publishColdRadiationHeat(state,i,profile,
                cold.caloric.enthalpy(heating?state.TpMax:state.TpMin),heating,fraction,cold.caloric);
            if (!sample.valid)
            {
                invalidRadiationProfile(state,i,heat);
                return;
            }
            accepted=sample.heat;
        }
        else
        {
            const GpuReal carry=residentRadiationCarry(state,i,0);
            ugkwp::ParticleThermalCarry<GpuReal> candidate{state.pT[i]};
            setRadiationCarry(candidate,carry,0);
            const GpuReal delta=coldProfileCapacityToBound(liquid.enthalpy(candidate.temperature),
                liquid.enthalpy(heating?state.TpMax:state.TpMin),heating,carry);
            const auto result=ugkwp::publishParticleThermalHeat(candidate,fraction*delta,state.TpMin,state.TpMax,liquid);
            if (!result.valid)
            {
                invalidRadiationProfile(state,i,heat);
                return;
            }
            state.pT[i]=candidate.temperature;
            writeResidentRadiationCarry(state,i,getRadiationCarry(candidate,0),0);
            accepted=result.acceptedSpecificHeat;
        }
        atomicAdd(state.radiationMobileTemperatureMass+cell,state.pm[i]*accepted);
    }
}
