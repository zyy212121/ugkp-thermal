#ifndef UGKP_FOAM_ADVANCE_BOUNDARY_H
#define UGKP_FOAM_ADVANCE_BOUNDARY_H
#include "gpu/AdvanceState.H"
#include "gpu/PhysicalEventClock.H"
#include "FoamGasTimeStepControls.H"
#include "Time.H"
#include "functionObjectList.H"
#include "timeControlFunctionObject.H"
#include <cmath>
#include <algorithm>
#include <limits>
namespace ugkp { namespace gpu {
inline bool foamRequiresStepObservation(const Foam::Time& time,const Foam::word& mode)
{
    return mode=="cpuTime" || mode=="clockTime" ||
        time.controlDict().lookupOrDefault<Foam::word>("stopAt","endTime")!="endTime";
}

inline std::uint64_t functionStepLimit(const Foam::Time& time)
{
    using Mode=Foam::timeControl::timeControls;
    const auto& objects=time.functionObjects();
    std::uint64_t limit=std::numeric_limits<std::uint64_t>::max();
    if(!objects.size())return limit;
    const auto* entries=time.controlDict().subDictPtr("functions");
    for(Foam::label i=0;i<objects.size();++i){
        const auto* scheduled=dynamic_cast<const Foam::functionObjects::timeControl*>(&objects[i]);
        if(!scheduled || Foam::functionObject::postProcess)return 1;
        const auto* definition=entries ? entries->subDictPtr(objects[i].name()) : nullptr;
        const auto cap=[&](const Foam::timeControl& control,const char* interval){
            const auto mode=control.control();
            if(mode==Mode::none || mode==Mode::writeTime || mode==Mode::outputTime)return;
            if(mode==Mode::timeStep && definition){
                const auto n=definition->lookupOrDefault<Foam::label>(interval,0);
                if(n>1){limit=std::min(limit,std::uint64_t(n-time.timeIndex()%n));return;}
            }
            limit=1;
        };
        cap(scheduled->executeControl(),"executeInterval");
        cap(scheduled->writeControl(),definition&&definition->found("outputControl")?"outputInterval":"writeInterval");
    }
    return limit;
}

inline bool foamAdvancePrefetchAllowed(const Foam::Time& time)
{
    if(!time.running())return false;
    const auto& controls=time.controlDict();
    return !time.functionObjects().size() && !controls.lookupOrDefault<bool>("runTimeModifiable",false)
        && !foamRequiresStepObservation(time,controls.lookupOrDefault<Foam::word>("writeControl","timeStep"));
}

inline AdvanceRequest foamAdvanceBoundary(Foam::Time& time,AdvanceRequest request,Foam::label logInterval,
    AdvanceClock clock){
    const bool exactPhysical=clock==AdvanceClock::physical;
    const auto& controls=time.controlDict();
    const auto mode=controls.lookupOrDefault<Foam::word>("writeControl","timeStep");
    const double interval=controls.lookupOrDefault<Foam::scalar>("writeInterval",1);
    request.boundary=time.endTime().value();request.endTime=request.boundary;
    request.stopOffset=exactPhysical?0:0.5;request.exactBoundary=exactPhysical;
    request.scheduleOrigin=time.beginTime().value();
    const bool mutableControls=controls.lookupOrDefault<bool>("runTimeModifiable",false);
    const auto functionSteps=functionStepLimit(time);
    const bool observe=functionSteps==1 || foamRequiresStepObservation(time,mode) || (mutableControls&&!exactPhysical);
    request.watchControlFiles=mutableControls&&exactPhysical&&!observe;
    if(exactPhysical&&mutableControls)
        request.adjustTimeStep=solverControls::readAdaptiveSwitch(controls,request.adjustTimeStep);
    const auto due=std::uint64_t(logInterval-time.timeIndex()%logInterval);
    const auto stride=std::uint64_t(logInterval);
    const auto count=observe ? std::uint64_t(1) : std::max(std::uint64_t(1),advanceProgressCapacity/stride);
    request.maximumSteps=std::min(due+(count-1)*stride,functionSteps);
    request.thermalOutputTime=request.endTime;
    request.progressInterval=count>1 ? static_cast<unsigned>(logInterval) : 0;
    request.maximumSteps=std::min(request.maximumSteps,std::uint64_t(Foam::labelMax-time.timeIndex()));
    request.adjustWrite=false;
    if(mode=="timeStep"&&interval>0){
        const auto steps=std::max(Foam::label(1),Foam::label(interval+0.5));
        request.thermalOutputStep=std::uint64_t(time.timeIndex())+steps-time.timeIndex()%steps;
        request.maximumSteps=std::min(request.maximumSteps,std::uint64_t(steps-time.timeIndex()%steps));
    }else if((mode=="runTime"||mode=="adjustableRunTime")&&interval>0){
        const double writeInterval=time.userTimeToTime(interval);
        const double begin=request.scheduleOrigin;
        if(exactPhysical)request.nextWrite=nextPeriodicTime(begin,writeInterval,time.value());
        else{
            const double index=std::floor(((time.value()-begin)+0.5*time.deltaTValue())/writeInterval);
            request.nextWrite=begin+(index+1)*writeInterval;
        }
        request.boundary=eventDeadline(time.value(),request.nextWrite,request.boundary);
        request.thermalOutputTime=request.boundary;
        request.adjustWrite=mode=="adjustableRunTime";
    }


    if(observe){
        request.maximumSteps=1;
        if(request.adjustWrite){
            const double functionTime=time.value()+time.functionObjects().timeToNextWrite();
            if(functionTime>time.value()){
                request.nextWrite=std::min(request.nextWrite,functionTime);
                if(exactPhysical)request.boundary=std::min(request.boundary,functionTime);
            }
        }
    }
    if(request.maximumSteps<=due)request.progressInterval=0;
    return request;
}
inline void publishFoamAdvance(Foam::Time& time,const AdvanceState& report,bool preserveRequestedWrite=false){
    if(!report.submissionSteps)return;
    const auto index=Foam::label(report.steps);
    const double dt=report.clockDt,previousDt=report.previousClockDt;


    if(report.submissionSteps>1){
        time.setTime(report.previousTime-previousDt,index-2);
        time.setDeltaTNoAdjust(previousDt);time++;
        if(preserveRequestedWrite&&time.writeTime())time.writeOnce();
    }
    time.setTime(report.previousTime,index-1);time.setDeltaTNoAdjust(dt);time++;
    time.setTime(report.time,index);
}
}}
#endif
