#pragma once
#include "GpuParticleCaloric.H"
namespace ugkwp {
#if defined(__CUDACC__)
#define UGKP_CALORIC_RESPONSE_HD __host__ __device__
#else
#define UGKP_CALORIC_RESPONSE_HD
#endif
struct ParticleCaloricHeatResponse {
    ParticleCaloric law;
    double initialTemperature,fraction,minimumTemperature,maximumTemperature;
    // The relaxation fraction is frozen at the source epoch, exactly as in
    // the reference gas particle exchange. Particle bounds are independent
    // of the gas thermo table. Clipping is part of the evaluated equation.
    UGKP_CALORIC_RESPONSE_HD void operator()(double gasTemperature,
        double& specificHeat,double& derivative)const noexcept {
        const double rawDelta=fraction*(gasTemperature-initialTemperature);
        const double lower=minimumTemperature-initialTemperature;
        const double upper=maximumTemperature-initialTemperature;
        const double delta=rawDelta<lower?lower:(rawDelta>upper?upper:rawDelta);
        specificHeat=law.enthalpyIncrement(initialTemperature,delta);
        derivative=rawDelta<lower || rawDelta>upper?0.:
            fraction*law.enthalpyDerivative(initialTemperature+delta);
    }
};
#undef UGKP_CALORIC_RESPONSE_HD
}
