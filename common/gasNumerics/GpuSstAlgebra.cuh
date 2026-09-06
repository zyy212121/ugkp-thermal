#include "GpuPrecisionTypes.H"
#ifndef UGKWP_GPU_SST_ALGEBRA_CUH
#define UGKWP_GPU_SST_ALGEBRA_CUH

#include <cmath>

#if defined(__CUDACC__)
#define UGKWP_SST_HD __host__ __device__
#else
#define UGKWP_SST_HD
#endif

namespace ugkwp
{

struct SstCoefficients
{
    GpuReal alphaK1;
    GpuReal alphaK2;
    GpuReal alphaOmega1;
    GpuReal alphaOmega2;
    GpuReal beta1;
    GpuReal beta2;
    GpuReal betaStar;
    GpuReal gamma1;
    GpuReal gamma2;
    GpuReal a1;
    GpuReal b1;
    GpuReal c1;
};

UGKWP_SST_HD inline SstCoefficients defaultSstCoefficients()
{
    return SstCoefficients{
        GPU_R(0.85),
        GPU_R(1.0),
        GPU_R(0.5),
        GPU_R(0.856),
        GPU_R(0.075),
        GPU_R(0.0828),
        GPU_R(0.09),
        GPU_R(5.0)/GPU_R(9.0),
        GPU_R(0.44),
        GPU_R(0.31),
        GPU_R(1.0),
        GPU_R(10.0)
    };
}

UGKWP_SST_HD inline GpuReal sstMaximum(const GpuReal a, const GpuReal b)
{
    return a > b ? a : b;
}

UGKWP_SST_HD inline GpuReal sstMinimum(const GpuReal a, const GpuReal b)
{
    return a < b ? a : b;
}

UGKWP_SST_HD inline GpuReal sstPositive(const GpuReal value, const GpuReal floor)
{
    return sstMaximum(value, floor);
}

UGKWP_SST_HD inline GpuReal sstBlend
(
    const GpuReal f1,
    const GpuReal value1,
    const GpuReal value2
)
{
    return f1*(value1 - value2) + value2;
}

UGKWP_SST_HD inline GpuReal sstAlphaK
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.alphaK1, coefficients.alphaK2);
}

UGKWP_SST_HD inline GpuReal sstAlphaOmega
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(
        f1,
        coefficients.alphaOmega1,
        coefficients.alphaOmega2
    );
}

UGKWP_SST_HD inline GpuReal sstBeta
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.beta1, coefficients.beta2);
}

UGKWP_SST_HD inline GpuReal sstGamma
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.gamma1, coefficients.gamma2);
}

UGKWP_SST_HD inline GpuReal sstCrossDiffusion
(
    const GpuReal omega,
    const GpuReal gradKDotGradOmega,
    const SstCoefficients& coefficients
)
{
    const GpuReal omegaSafe = sstPositive(omega, GPU_TINY(1.0e-300));
    return
        GPU_R(2.0)*coefficients.alphaOmega2*gradKDotGradOmega/omegaSafe;
}

UGKWP_SST_HD inline GpuReal sstF1
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal nu,
    const GpuReal wallDistance,
    const GpuReal cdKOmega,
    const SstCoefficients& coefficients
)
{
    const GpuReal kSafe = sstPositive(k, GPU_R(0.0));
    const GpuReal omegaSafe = sstPositive(omega, GPU_TINY(1.0e-300));
    const GpuReal ySafe = sstPositive(wallDistance, GPU_TINY(1.0e-300));
    const GpuReal ySquared = ySafe*ySafe;
    const GpuReal cdPlus = sstMaximum(cdKOmega, GPU_R(1.0e-10));
    const GpuReal viscousArgument = GPU_R(500.0)*nu/(ySquared*omegaSafe);
    const GpuReal turbulentArgument =
        std::sqrt(kSafe)/(coefficients.betaStar*omegaSafe*ySafe);
    const GpuReal crossDiffusionArgument =
        GPU_R(4.0)*coefficients.alphaOmega2*kSafe/(cdPlus*ySquared);
    const GpuReal argument = sstMinimum(
        sstMinimum(
            sstMaximum(turbulentArgument, viscousArgument),
            crossDiffusionArgument
        ),
        GPU_R(10.0)
    );
    const GpuReal argumentSquared = argument*argument;
    return std::tanh(argumentSquared*argumentSquared);
}

UGKWP_SST_HD inline GpuReal sstF2
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal nu,
    const GpuReal wallDistance,
    const SstCoefficients& coefficients
)
{
    const GpuReal kSafe = sstPositive(k, GPU_R(0.0));
    const GpuReal omegaSafe = sstPositive(omega, GPU_TINY(1.0e-300));
    const GpuReal ySafe = sstPositive(wallDistance, GPU_TINY(1.0e-300));
    const GpuReal ySquared = ySafe*ySafe;
    const GpuReal argument = sstMinimum(
        sstMaximum(
            GPU_R(2.0)*std::sqrt(kSafe)
           /(coefficients.betaStar*omegaSafe*ySafe),
            GPU_R(500.0)*nu/(ySquared*omegaSafe)
        ),
        GPU_R(100.0)
    );
    return std::tanh(argument*argument);
}

UGKWP_SST_HD inline GpuReal sstNut
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal s2,
    const GpuReal f2,
    const SstCoefficients& coefficients
)
{
    const GpuReal denominator = sstMaximum(
        coefficients.a1*sstPositive(omega, GPU_TINY(1.0e-300)),
        coefficients.b1*f2*std::sqrt(sstPositive(s2, GPU_R(0.0)))
    );
    return coefficients.a1*sstPositive(k, GPU_R(0.0))/denominator;
}

UGKWP_SST_HD inline GpuReal sstKProduction
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal nut,
    const GpuReal gByNu,
    const SstCoefficients& coefficients
)
{
    return sstMinimum(
        nut*gByNu,
        coefficients.c1*coefficients.betaStar
       *sstPositive(k, GPU_R(0.0))*sstPositive(omega, GPU_R(0.0))
    );
}

UGKWP_SST_HD inline GpuReal sstKSource
(
    const GpuReal rho,
    const GpuReal k,
    const GpuReal omega,
    const GpuReal divU,
    const GpuReal nut,
    const GpuReal gByNu,
    const SstCoefficients& coefficients
)
{
    return rho*(
        sstKProduction(k, omega, nut, gByNu, coefficients)
      - (GPU_R(2.0)/GPU_R(3.0))*divU*k
      - coefficients.betaStar*k*omega
    );
}

UGKWP_SST_HD inline GpuReal sstOmegaProductionLimit
(
    const GpuReal omega,
    const GpuReal s2,
    const GpuReal f2,
    const SstCoefficients& coefficients
)
{
    return
        (coefficients.c1/coefficients.a1)
       *coefficients.betaStar*omega
       *sstMaximum(
            coefficients.a1*omega,
            coefficients.b1*f2*std::sqrt(sstPositive(s2, GPU_R(0.0)))
        );
}

UGKWP_SST_HD inline GpuReal sstOmegaSource
(
    const GpuReal rho,
    const GpuReal k,
    const GpuReal omega,
    const GpuReal divU,
    const GpuReal gByNu,
    const GpuReal s2,
    const GpuReal f1,
    const GpuReal f2,
    const GpuReal cdKOmega,
    const SstCoefficients& coefficients
)
{
    (void)k;
    const GpuReal gamma = sstGamma(f1, coefficients);
    const GpuReal beta = sstBeta(f1, coefficients);
    const GpuReal production = gamma*sstMinimum(
        gByNu,
        sstOmegaProductionLimit(omega, s2, f2, coefficients)
    );
    return rho*(
        production
      - (GPU_R(2.0)/GPU_R(3.0))*gamma*divU*omega
      - beta*omega*omega
      + (GPU_R(1.0) - f1)*cdKOmega
    );
}

UGKWP_SST_HD inline GpuReal sstLowReWallOmega
(
    const GpuReal nu,
    const GpuReal wallDistance,
    const SstCoefficients& coefficients
)
{
    const GpuReal ySafe = sstPositive(wallDistance, GPU_TINY(1.0e-300));
    return GPU_R(6.0)*nu/(coefficients.beta1*ySafe*ySafe);
}

}                   

#undef UGKWP_SST_HD

#endif
