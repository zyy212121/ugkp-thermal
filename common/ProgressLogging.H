#ifndef UGKWP_PROGRESSLOGGING_H
#define UGKWP_PROGRESSLOGGING_H

#include "Time.H"
#include "dictionary.H"
#include "label.H"

namespace ugkwp
{
namespace logging
{

constexpr Foam::label defaultProgressLogInterval = 5000;

inline Foam::label progressLogInterval(const Foam::dictionary& controlDict)
{
    const Foam::label configured =
        controlDict.lookupOrDefault<Foam::label>
        (
            "logInterval",
            defaultProgressLogInterval
        );

    return configured > 0 ? configured : defaultProgressLogInterval;
}

template<class Step>
inline bool progressLogDue(const Step step, const Foam::label interval)
{
    return step > Step(0)
        && interval > 0
        && step % static_cast<Step>(interval) == Step(0);
}

inline void writeProgressLog(const Foam::Time& runTime,Foam::label step,const Foam::word& timeName,double dt,bool collected=false)
{
    Foam::Info<< "UGKP_REPRO step=" << step
        << " time=" << timeName
        << " dt=" << dt
        << (collected?" batchElapsed=":" elapsed=") << runTime.elapsedClockTime() << Foam::endl;
}

inline void writeProgressLog(const Foam::Time& runTime)
{
    writeProgressLog(runTime,runTime.timeIndex(),runTime.timeName(),runTime.deltaTValue());
}

}
}

#endif
