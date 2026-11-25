#pragma once
#include "core/Types.H"
#include "../../../common/gpu/PhysicalEventClock.H"
#include <string>

namespace chmt {
struct IntervalIdentity { std::uint64_t sequence=0,epoch=0; };
struct CouplingInterval { IntervalIdentity identity{}; Real begin=0,end=0; };
struct CouplingControls {
    Real gasMaxDt=0,materialMaxSubstep=0;
    double scheduleOrigin=0;
    ugkp::gpu::OutputCouplingClock clock{};
};

class CouplingWindowController {
    CouplingInterval interval_{};
public:
    const CouplingInterval& interval() const {return interval_;}
    static bool outputDue(const CouplingInterval& interval,const CouplingControls& controls){
        const double period=controls.clock.outputPeriod;
        return period>0 && ugkp::gpu::reachedEventTime(interval.end,
            ugkp::gpu::nextPeriodicTime(controls.scheduleOrigin,period,interval.begin),interval.end-interval.begin);
    }
    static std::uint64_t outputIndex(Real time,const CouplingControls& controls){
        return static_cast<std::uint64_t>(std::round((time-controls.scheduleOrigin)/controls.clock.outputPeriod));
    }
    bool begin(Real time,Real target,std::uint64_t sequence,const CouplingControls& controls,std::string& error){
        if(!(controls.clock.contactPeriod>0) || !std::isfinite(controls.clock.contactPeriod) ||
           !std::isfinite(time) || !std::isfinite(target) || !std::isfinite(controls.scheduleOrigin)){
            error="invalid coupling window period or physical time";return false;
        }
        const double tick=ugkp::gpu::nextFullOrContactEvent(controls.clock,controls.scheduleOrigin,time);
        const double end=ugkp::gpu::eventDeadline(time,tick,target);
        if(!(end>time) || !std::isfinite(tick)){
            error="coupling window has no representable advancing endpoint";return false;
        }
        interval_={{sequence,0},time,end};
        error.clear();return true;
    }
};
}
