#pragma once

#include <cmath>
#include <limits>

namespace ugkpcht
{

template<class Scalar>
inline Scalar thermalScheduleTolerance
(
    const Scalar elapsed,
    const Scalar interval
)
{
    using std::abs;
    Scalar scale = abs(elapsed) > abs(interval) ? abs(elapsed) : abs(interval);
    if (scale < Scalar(1))
    {
        scale = Scalar(1);
    }
    return Scalar(64)*std::numeric_limits<Scalar>::epsilon()*scale;
}

template<class Scalar>
inline bool thermalEventIsDue
(
    const Scalar elapsed,
    const Scalar interval
)
{
    return elapsed + thermalScheduleTolerance(elapsed, interval) >= interval;
}

}                     
