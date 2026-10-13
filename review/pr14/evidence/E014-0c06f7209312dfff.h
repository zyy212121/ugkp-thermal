#pragma once
#include <cmath>
template<class Real>
inline const char* validateContactCapabilities
(const int candidateCount, const bool cold1D, const bool cold2D,
 const int heatTransferEnabled, const Real maximumCoverage)
{
    if ((cold1D || cold2D) && heatTransferEnabled == 0)
        return "cold-wall conduction requires enabled particle-wall heat transfer";
    if (candidateCount > 0 && (!std::isfinite(maximumCoverage)
        || maximumCoverage <= Real(0) || maximumCoverage > Real(1)))
        return "mechanical particle contact requires maximumCoverage in (0, 1]";
    return nullptr;
}
