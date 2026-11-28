#pragma once

#include "RadiationPrecision.H"
#include "../GpuParticleCaloric.H"
#include "../wall/ParticleWallState.H"
#include <cmath>

#if defined(__CUDACC__)
#define UGKP_RADIATION_PROFILE_HD __host__ __device__
#else
#define UGKP_RADIATION_PROFILE_HD
#endif

namespace ugkp { namespace gpu {

UGKP_RADIATION_PROFILE_HD inline bool radiationParticleEligible(int status, unsigned char wallState)
{
    using namespace Foam::gpuThermal;
    return status == 1 && (wallState == particleWallMobile || wallState == particleWallDeposited);
}

struct ColdRadiationProfile
{
    const void* enthalpy = nullptr;
    int nodes = 0;
};

struct ParticleRadiationColdProfiles
{
    int faces = 0;
    const int* face = nullptr;
    const unsigned char* interaction = nullptr;
    const int* axialSlot = nullptr;
    const GpuReal* axialEnthalpy = nullptr;
    const float* axisymmetricEnthalpy = nullptr;
    ugkwp::ParticleCaloric caloric{};

    UGKP_RADIATION_PROFILE_HD ColdRadiationProfile profile(int particle, unsigned char wallState) const
    {
        using namespace Foam::gpuThermal;
        if (wallState != particleWallDeposited || !interaction || !face) return {};
        const int wallFace = face[particle];
        if (wallFace < 0 || wallFace >= faces) return {};
        if (interaction[wallFace] == particleWallSolidifyingDeposition)
        {
            const int slot = axialSlot ? axialSlot[particle] : particle;
            return {axialEnthalpy && slot >= 0 ? axialEnthalpy + slot*coldWallAxialNodeCount : nullptr,
                coldWallAxialNodeCount};
        }
        if (interaction[wallFace] == particleWallColdWall2D)
        {
            constexpr int nodes = coldWallAxialNodeCount*coldWallRadialRingCount;
            return {axisymmetricEnthalpy ? axisymmetricEnthalpy + particle*nodes : nullptr, nodes};
        }
        return {};
    }
};

UGKP_RADIATION_PROFILE_HD inline RadiationReal coldProfileCapacityToBound
(RadiationReal oldEnthalpy, RadiationReal bound, bool heating, RadiationReal carry=0)
{
    const RadiationReal available = (bound-oldEnthalpy)-carry;
    return (heating ? available > 0 : available < 0) ? available : RadiationReal(0);
}

template<class Storage>
UGKP_RADIATION_PROFILE_HD RadiationReal coldNodeSpecificCapacity
(const Storage* enthalpy, int count, RadiationReal carry, RadiationReal bound, bool heating)
{
    if (!enthalpy || count <= 0) return RadiationReal(NAN);
    RadiationReal total=0;
    RadiationReal minimum=enthalpy[0];
    for (int node=0; node<count; ++node)
    {
        const RadiationReal value=enthalpy[node];
        total+=value;
        if (value<minimum) minimum=value;
    }
    const RadiationReal capacity=coldProfileCapacityToBound(total/RadiationReal(count),bound,heating,carry);
    if (heating) return capacity;
    const RadiationReal nodeCapacity=coldProfileCapacityToBound(minimum,RadiationReal(0),false,carry);
    return capacity>nodeCapacity?capacity:nodeCapacity;
}

UGKP_RADIATION_PROFILE_HD inline RadiationReal coldProfileSpecificCapacity
(ColdRadiationProfile profile, RadiationReal carry, RadiationReal bound, bool heating)
{
    if (profile.nodes == Foam::gpuThermal::coldWallAxialNodeCount)
        return coldNodeSpecificCapacity(static_cast<const GpuReal*>(profile.enthalpy),
            profile.nodes,carry,bound,heating);
    return coldNodeSpecificCapacity(static_cast<const float*>(profile.enthalpy),
        profile.nodes,carry,bound,heating);
}

struct ColdRadiationSample
{
    RadiationReal temperature = 0;
    RadiationReal heat = 0;
    RadiationReal grid = 0;
    RadiationReal carry = 0;
    bool valid = false;
};

template<class Storage>
UGKP_RADIATION_PROFILE_HD ColdRadiationSample sampleColdRadiationNodes
(
    const Storage* enthalpy, int count, RadiationReal carry, RadiationReal bound, bool heating,
    RadiationReal fraction, const ugkwp::ParticleCaloric& law, Storage* publish = nullptr
)
{
    if (!enthalpy || count <= 0 || !std::isfinite(fraction) || fraction < 0 || fraction > 1) return {};
    constexpr bool compensated = sizeof(Storage) < sizeof(RadiationReal);
    const RadiationReal heat = fraction == 0 ? RadiationReal(0)
        : fraction*coldNodeSpecificCapacity(enthalpy,count,carry,bound,heating);
    RadiationReal mean = 0, change = 0, grid = 0;
    for (int node=0; node<count; ++node)
    {
        const RadiationReal old = enthalpy[node];
        const Storage stored = fraction == 0 ? enthalpy[node] : Storage(old+(carry+heat));
        if (!std::isfinite(stored)) return {};
        mean += stored;
        change += RadiationReal(stored)-old;
        if constexpr (!compensated)
        {
            const Storage adjacent=std::nextafter(stored,heating?Storage(INFINITY):-Storage(INFINITY));
            grid+=std::abs(RadiationReal(adjacent)-stored);
        }
    }
    const RadiationReal inverseCount = RadiationReal(1)/RadiationReal(count);
    const RadiationReal storageChange = change*inverseCount;
    const RadiationReal nextCarry = compensated ? (carry-storageChange)+heat : RadiationReal(0);
    const RadiationReal effectiveEnthalpy=mean*inverseCount+nextCarry;
    if constexpr (compensated)
        grid=std::abs(std::nextafter(effectiveEnthalpy,heating
            ? RadiationReal(INFINITY) : -RadiationReal(INFINITY))-effectiveEnthalpy);
    else
        grid*=inverseCount;
    const RadiationReal temperature=law.temperature(effectiveEnthalpy);
    const RadiationReal accepted = storageChange+(nextCarry-carry);
    if (!std::isfinite(temperature) || !std::isfinite(accepted) || !std::isfinite(grid)) return {};
    if (publish && fraction != 0)
    {
        for (int node=0; node<count; ++node)
        {
            const RadiationReal old = enthalpy[node];
            publish[node] = Storage(old+(carry+heat));
        }
    }
    return {temperature,accepted,grid,nextCarry,true};
}

UGKP_RADIATION_PROFILE_HD inline ColdRadiationSample sampleColdRadiationProfile
(
    ColdRadiationProfile profile, RadiationReal carry, RadiationReal bound, bool heating,
    RadiationReal fraction, const ugkwp::ParticleCaloric& law
)
{
    if (profile.nodes == Foam::gpuThermal::coldWallAxialNodeCount)
        return sampleColdRadiationNodes(static_cast<const GpuReal*>(profile.enthalpy),
            profile.nodes,RadiationReal(0),bound,heating,fraction,law);
    return sampleColdRadiationNodes(static_cast<const float*>(profile.enthalpy),
        profile.nodes,carry,bound,heating,fraction,law);
}

template<class View,class Sample>
UGKP_RADIATION_PROFILE_HD void accumulateColdRadiationResponse
(
    const View& view,int particle,ColdRadiationProfile profile,
    RadiationReal carry,RadiationReal fraction,bool heating,Sample& result
)
{
    const auto& law=view.coldProfiles.caloric;
    const RadiationReal bound=law.enthalpy(heating?view.maximumTemperature:view.minimumTemperature);
    if (profile.nodes==Foam::gpuThermal::coldWallAxialNodeCount) carry=0;
    const auto trial=sampleColdRadiationProfile(profile,carry,bound,heating,fraction,law);
    const RadiationReal capacity=coldProfileSpecificCapacity(profile,carry,bound,heating);
    result.mass+=view.mass[particle];
    result.temperature+=view.mass[particle]*(trial.valid?trial.temperature:RadiationReal(NAN));
    result.energy+=view.mass[particle]*fraction*capacity;
}

}}
#undef UGKP_RADIATION_PROFILE_HD
