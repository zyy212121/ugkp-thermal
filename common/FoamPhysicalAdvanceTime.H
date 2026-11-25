#pragma once
#include "FoamAdvanceBoundary.H"
#include "DiagnosticsHost.H"
#include "timeIOdictionary.H"
#include "fileOperation.H"

namespace ugkp { namespace gpu {


class PhysicalAdvanceTime : public Foam::Time
{
    double nominalDt_ = 0;
public:
    using Foam::Time::Time;
    using Foam::Time::operator++;

    AdvanceRequest advanceControls(const solverControls::GasTimeStepControls& gas) const
    {
        AdvanceRequest request;
        request.adjustTimeStep=gas.adjustTimeStep;
        request.targetCourant=gas.maxCo;
        request.maximumDt=gas.maxDeltaT;
        request.growthLimit=gas.maximumTimeStepGrowth;
        request.courantInterval=gas.courantUpdateInterval;
        request.small=Foam::SMALL;
        return request;
    }

    bool observesExternalWrite() const
    {
        return writeControl_==writeControl::cpuTime || writeControl_==writeControl::clockTime ||
            stopAt_!=stopAtControl::endTime || sigWriteNow_.active() || sigStopAtWriteNow_.active();
    }

    AdvanceState initialAdvance(const AdvanceRequest& controls) const
    {
        AdvanceState initial;
        initial.time=value();
        initial.dt=initialPhysicalDt(controls.adjustTimeStep);
        initial.previousDt=deltaT0Value();
        initial.steps=timeIndex();
        initial.acceptedSequence=initial.steps;
        initial.stepsSinceCourant=controls.courantInterval;
        return initial;
    }

    AdvanceRequest advanceBoundary(AdvanceRequest controls,Foam::label logInterval)
    {
        auto request=foamAdvanceBoundary(*this,controls,logInterval,AdvanceClock::physical);
        if(observesExternalWrite() || writeOnce_)request.maximumSteps=1;
        if(writeOnce_)request.thermalOutputStep=timeIndex()+1;
        return request;
    }

    double initialPhysicalDt(bool adaptive) const
    {
        if (!adaptive) return deltaTValue();
        const Foam::timeIOdictionary saved(Foam::IOobject("time",timeName(),"uniform",*this,
            Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false));
        const auto* entry = saved.lookupEntryPtr("cflDeltaT",false,false);
        if (!entry) return deltaTValue();
        if (entry->isStream()) {
            const auto& tokens=entry->stream();
            if (tokens.size()==1 && tokens[0].isNumber()) {
                const double dt=userTimeToTime(tokens[0].number());
                if (std::isfinite(dt) && dt>0) return dt;
            }
        }
        diagnostics::hostWarning(diagnostics::Code::RestartMetadata,
            {35018,__FILE__,__func__,__LINE__,"invalid optional cflDeltaT, using the accepted restart step"});
        return deltaTValue();
    }

    void publishAdvance(const AdvanceState& state)
    {
        if (state.submissionSteps) nominalDt_=state.dt;
        publishFoamAdvance(*this,state,true);
    }

    bool writeTimeDict() const override
    {
        if (nominalDt_==deltaTValue() || !(nominalDt_>0)
            || !solverControls::readAdaptiveSwitch(controlDict(),false))
            return Foam::Time::writeTimeDict();



        Foam::timeIOdictionary saved(Foam::IOobject("time",timeName(),"uniform",*this,
            Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false));
        saved.add("beginTime",timeToUserTime(beginTime_));
        saved.add("value",timeName(timeToUserTime(value()),maxPrecision_));
        saved.add("name",Foam::string(timeName()));
        saved.add("index",timeIndex());
        saved.add("deltaT",timeToUserTime(deltaTValue()));
        saved.add("deltaT0",timeToUserTime(deltaT0Value()));
        saved.add("cflDeltaT",timeToUserTime(nominalDt_));
        return saved.regIOobject::writeObject(Foam::IOstream::ASCII,Foam::IOstream::currentVersion,
            Foam::IOstream::UNCOMPRESSED,true);
    }

    bool running() const override
    {
        return stopAt_ == stopAtControl::endTime ? value() < endTime_ : Foam::Time::running();
    }

    Foam::Time& operator++() override
    {
        const double end = value()+deltaTValue();
        if (stopAt_ == stopAtControl::endTime && reachedEventTime(end,endTime_,deltaTValue()))
            writeOnce();
        const auto mode = writeControl_;
        const auto interval = writeInterval_;
        if ((mode != writeControl::runTime && mode != writeControl::adjustableRunTime) || interval <= 0)
            return Foam::Time::operator++();

        const double target = nextPeriodicTime(beginTime_,interval,value());



        if (reachedEventTime(end,target,deltaTValue())) {
            writeControl_ = writeControl::timeStep;
            writeInterval_ = 1;
        } else {
            writeInterval_ = Foam::great;
        }
        try {
            Foam::Time::operator++();
        } catch (...) {
            writeControl_ = mode;
            writeInterval_ = interval;
            throw;
        }
        writeControl_ = mode;
        writeInterval_ = Foam::great;
        setWriteInterval(interval);
        return *this;
    }
};
inline bool foamAdvancePrefetchAllowed(const PhysicalAdvanceTime& time)
{
    return !time.observesExternalWrite() && foamAdvancePrefetchAllowed(static_cast<const Foam::Time&>(time));
}
}}
