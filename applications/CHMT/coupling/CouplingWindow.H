#pragma once
#include "core/Types.H"
#include "../../../common/gpu/PhysicalEventClock.H"
#include <string>

namespace chmt {
struct IntervalIdentity { std::uint64_t sequence=0,epoch=0; };
struct CouplingInterval { IntervalIdentity identity{}; Real begin=0,end=0; };
struct CouplingControls {
    Real interval=0,gasMaxDt=0,materialMaxSubstep=0;
    double scheduleOrigin=0;
};

class CouplingWindowController {
    CouplingInterval interval_{};
public:
    const CouplingInterval& interval() const {return interval_;}
    bool begin(Real time,Real target,std::uint64_t sequence,const CouplingControls& controls,std::string& error){
        if(!(controls.interval>0) || !std::isfinite(controls.interval) ||
           !std::isfinite(time) || !std::isfinite(target) || !std::isfinite(controls.scheduleOrigin)){
            error="invalid coupling window period or physical time";return false;
        }
        const double tick=ugkp::gpu::nextPeriodicTime(controls.scheduleOrigin,controls.interval,time);
        const double end=ugkp::gpu::eventDeadline(time,tick,target);
        if(!(end>time) || !std::isfinite(tick)){
            error="coupling window has no representable advancing endpoint";return false;
        }
        interval_={{sequence,0},time,end};
        error.clear();return true;
    }
};
}
