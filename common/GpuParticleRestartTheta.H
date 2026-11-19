#pragma once
#include <cstddef>


#ifdef __CUDACC__
__host__ __device__
#endif
inline double particleRestartTheta(unsigned char stuck, double physicalTheta, double contactAge)
{
    return stuck ? contactAge : physicalTheta;
}

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
        wireTheta[i] = particleRestartTheta(stuck[i],wireTheta[i],contactAge[i]);
}
