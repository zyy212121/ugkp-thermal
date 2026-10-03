#pragma once
#include <cstddef>
// Schema7 has one theta wire slot. Mobile particles store physical theta;
// wall-bound particles store contact age. CHT keeps the two resident arrays separate.
template<class Real, class Time>
inline void unpackParticleRestartTheta(const std::size_t n,
    const unsigned char* stuck, const double* wireTheta, Real* physicalTheta, Time* contactAge)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        physicalTheta[i] = stuck[i] == 0 ? static_cast<Real>(wireTheta[i]) : Real(0);
        contactAge[i] = stuck[i] != 0 ? wireTheta[i] : Time(0);
    }
}
template<class Time>
inline void packParticleRestartTheta(const std::size_t n,
    const unsigned char* stuck, const Time* contactAge, double* wireTheta)
{
    for (std::size_t i = 0; i < n; ++i)
        if (stuck[i] != 0) wireTheta[i] = contactAge[i];
}
