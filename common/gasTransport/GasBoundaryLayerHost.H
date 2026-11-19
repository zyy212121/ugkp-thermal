#pragma once
#include "GasBoundaryLayerState.H"
namespace ugkwp
{
template<class Policy,class Host,class Time,class=void> struct GasBoundaryLayerPrepareCapability:std::false_type{};
template<class Policy,class Host,class Time> struct GasBoundaryLayerPrepareCapability<Policy,Host,Time,
    std::void_t<decltype(Policy::prepareBoundaryLayer(static_cast<Host*>(nullptr),Time{},Time{}))>>:std::true_type{};


template<class Host> inline void invalidateGasBoundaryLayerPreparation(Host& host)
{
    if constexpr(GasBoundaryLayerCapability<Host>::value)
    {
        host.gasBoundaryLayer.preparedFirstStage=false;
        ++host.gasBoundaryLayer.generation;
    }
}
template<class Host,class Time> inline bool gasBoundaryLayerPreparationReusable(const Host& host,Time dt,Time time)
{
    if constexpr(GasBoundaryLayerCapability<Host>::value)
    {
        const auto& wall=host.gasBoundaryLayer;
        return wall.enabled && wall.preparedFirstStage && wall.preparedGeneration==wall.generation
            && wall.preparedTime==double(time) && wall.preparedInterval==double(dt);
    }
    return false;
}
template<class Policy,class Host,class Time>
inline int prepareGasBoundaryLayerStage(Host* host,Time dt,Time time,bool preflight)
{
    if constexpr(GasBoundaryLayerCapability<Host>::value)
    {
        auto& wall=host->gasBoundaryLayer;
        if(!wall.enabled)return 0;
        const bool reuse=!preflight && gasBoundaryLayerPreparationReusable(*host,dt,time);
        wall.preparedFirstStage=false;
        if(reuse)return 0;
        if constexpr(GasBoundaryLayerPrepareCapability<Policy,Host,Time>::value)
        {
            const int result=Policy::prepareBoundaryLayer(host,dt,time);
            if(result)return result;
            if(preflight)
            {
                wall.preparedFirstStage=true;wall.preparedGeneration=wall.generation;
                wall.preparedTime=double(time);wall.preparedInterval=double(dt);
            }
            return 0;
        }
        return 1;
    }
    return 0;
}
}
