#pragma once
#include "PrecisionRestart.H"
#include "IOstream.H"
#include <algorithm>
#include <limits>

namespace ugkp { namespace precision {


class ScopedRestartWritePrecision
{
    unsigned int previous_;
public:
    ScopedRestartWritePrecision()
    : previous_(Foam::IOstream::defaultPrecision())
    {
        Foam::IOstream::defaultPrecision(std::max(
            previous_, unsigned(std::numeric_limits<double>::max_digits10)));
    }
    ~ScopedRestartWritePrecision()
    {
        Foam::IOstream::defaultPrecision(previous_);
    }
    ScopedRestartWritePrecision(const ScopedRestartWritePrecision&) = delete;
    ScopedRestartWritePrecision& operator=(const ScopedRestartWritePrecision&) = delete;
};
}}
