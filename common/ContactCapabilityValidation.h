#pragma once
#include <cmath>
template<class Real>
inline const char* validateContactCapabilities
(const int candidateCount, const bool, const bool,
 const int, const Real maximumCoverage)
{
    if (candidateCount > 0 && !std::isfinite(maximumCoverage))
        return "mechanical particle contact requires finite maximumCoverage";
    return nullptr;
}
