#pragma once
#include "gpu/AdvanceApi.H"
#include "ProgressLogging.H"
#include "FoamPhysicalAdvanceTime.H"

namespace ugkp { namespace gpu {
inline int collectFoamAdvance(void* handle,AdvanceState& state,const Foam::Time* time)
{
    if (!time) return ugkwpGpuResidentStrictCollectAdvance(handle,&state);
    AdvanceProgress progress;
    const int status=ugkwpGpuResidentStrictCollectAdvanceProgress(handle,&state,&progress,nullptr,nullptr);
    for (unsigned i=0;i<progress.count;++i) {
        const auto& point=progress.points[i];
        if (!status && point.step==state.steps) continue;
        ugkwp::logging::writeProgressLog(*time,Foam::label(point.step),
            Foam::Time::timeName(time->timeToUserTime(point.time)),point.dt,true);
    }
    return status;
}
}}
