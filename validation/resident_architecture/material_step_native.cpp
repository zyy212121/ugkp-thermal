#include "material_step_probe.H"
#include "materials/MaterialDrive.H"
#include <iostream>
#include <cassert>
#include <limits>
#include <iomanip>
using namespace chmt;
WindowTime native(ugkp::CompensatedTime t){WindowTime x(t.sum);x.add(t.correction);return x;}
int main(){std::vector<materialStepProbe::Test> tests;
    for(int start=0;start<8;++start)for(double maximum:{0.,.01,.1,.5,1.,3.})for(int mode=0;mode<10;++mode){
        materialStepProbe::Test t;t.begin=ugkp::CompensatedTime(start==0?0:start==1?.1:start==2?.75:start==3?1:start==4?1.25:start==5?2:start==6?-1:1);
        if(start==7)t.begin.add(1e-20);t.controls.maxSubstep=maximum;t.changed=mode%2;
        if(mode>=2){for(double k:{0.,.25,.75,1.,1.5,2.})t.knots.push_back({ugkp::CompensatedTime(k),k});}
        if(mode==3){t.knots[3].local.add(-1e-20);}
        if(mode==4)t.controls.driveRelativeTolerance=-1;
        if(mode==5)t.transportError=7;
        if(mode==6)t.transportLimit=.003;
        if(mode==7)t.transportLimit=0;
        if(mode==8)t.transportLimit=std::numeric_limits<double>::quiet_NaN();
        if(mode==9)t.transportLimit=1e-25;
        tests.push_back(t);
    }
    const auto actual=materialStepProbe::evaluate(tests);std::size_t errors=0,finished=0;double maximumError=0;long double maximumClockError=0;
    for(std::size_t i=0;i<tests.size();++i){const auto& t=tests[i];const auto& out=actual[i];const auto elapsed=native(t.begin),end=native(t.total);
        assert(out.active==(elapsed<end));if(!out.active){++finished;assert(out.error==MaterialStepSelectionError::None);continue;}
        const auto& controls=t.controls;
        double dt=std::min(boundedWindowDuration(end,elapsed),controls.maxSubstep>0?controls.maxSubstep:2.);
        IntervalHistory history;CouplingInterval window;window.end=2;std::string message;history.begin(window,message);
        for(int r=0;r<2;++r){GasIntervalRecord record;record.localBegin=WindowTime(r);record.localEnd=WindowTime(r+1);record.actualDt=1;
            GasPrimitive gas;gas.temperature=r&&t.changed?600:300;record.gasTrace={gas};record.gasGradient.resize(1);record.gasTraction.resize(1);record.radiationReceiver={0};history.appendAccepted(record,message);}
        MaterialDriveCursor drive(history);WindowTime boundedEnd=elapsed+dt;
        MaterialStepSelectionError error=MaterialStepSelectionError::None;
        if(!drive.slabEndLocal(elapsed,boundedEnd,controls.driveRelativeTolerance,controls.driveTractionScale,boundedEnd,message))error=MaterialStepSelectionError::Drive;
        else {dt=boundedWindowDuration(boundedEnd,elapsed);
            for(const auto& knot:t.knots)if(native(knot.local)>elapsed&&native(knot.local)<elapsed+dt)dt=boundedWindowDuration(native(knot.local),elapsed);
            if(t.transportError)error=MaterialStepSelectionError::Transport;
            else {dt=std::min(dt,t.transportLimit);if(!std::isfinite(dt)||!(dt>0)||elapsed+dt==elapsed)error=MaterialStepSelectionError::NonAdvancing;}}
        assert(out.error==error);if(error!=MaterialStepSelectionError::None){++errors;continue;}
        maximumError=std::max(maximumError,std::abs(dt-out.dt));
        if(dt!=out.dt){std::cerr<<std::setprecision(25)<<"timing mismatch i="<<i<<" begin="<<t.begin.sum<<" + "<<t.begin.correction<<" maximum="<<t.controls.maxSubstep<<" limit="<<t.transportLimit<<" expected dt="<<dt<<" device dt="<<out.dt<<" end="<<out.end.sum<<" + "<<out.end.correction<<"\n";}
        assert(dt==out.dt);
        const long double clockError=std::abs(windowDifference(native(out.end),elapsed+dt));
        maximumClockError=std::max(maximumClockError,clockError);
        // The oracle uses two binary80 components; the resident clock uses two
        // FP64 components. Require exact chosen dt and unchanged endpoint
        // classification, with only double-double roundoff in the clock tail.
        const long double clockBound=std::numeric_limits<double>::epsilon()*std::numeric_limits<double>::epsilon()
            *std::max(1.,std::abs(t.begin.value())+std::abs(dt));
        assert(clockError<=clockBound);assert(windowAbsolute(0,native(out.end))==windowAbsolute(0,elapsed+dt));
        assert(out.finalStep==(elapsed+dt==end));
    }
    assert(materialStepProbe::evaluate({}).empty());
    std::cout<<"PASS "<<tests.size()<<" native/device material step selections, "<<errors<<" original failures, "<<finished<<" completed clocks; exact dt/output stamps and endpoint classifications, maximum compensated tail error "<<std::setprecision(17)<<maximumClockError<<"; three kernel-only graph replays\n";
}
