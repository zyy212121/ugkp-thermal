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
    double alphaK1;
    double alphaK2;
    double alphaOmega1;
    double alphaOmega2;
    double beta1;
    double beta2;
    double betaStar;
    double gamma1;
    double gamma2;
    double a1;
    double b1;
    double c1;
};

UGKWP_SST_HD inline SstCoefficients defaultSstCoefficients()
{
    return SstCoefficients{
        0.85,
        1.0,
        0.5,
        0.856,
        0.075,
        0.0828,
        0.09,
        5.0/9.0,
        0.44,
        0.31,
        1.0,
        10.0
    };
}

UGKWP_SST_HD inline double sstMaximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKWP_SST_HD inline double sstMinimum(const double a, const double b)
{
    return a < b ? a : b;
}

UGKWP_SST_HD inline double sstPositive(const double value, const double floor)
{
    return sstMaximum(value, floor);
}

UGKWP_SST_HD inline double sstBlend
(
    const double f1,
    const double value1,
    const double value2
)
{
    return f1*(value1 - value2) + value2;
}

UGKWP_SST_HD inline double sstAlphaK
(
    const double f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.alphaK1, coefficients.alphaK2);
}

UGKWP_SST_HD inline double sstAlphaOmega
(
    const double f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(
        f1,
        coefficients.alphaOmega1,
        coefficients.alphaOmega2
    );
}

UGKWP_SST_HD inline double sstBeta
(
    const double f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.beta1, coefficients.beta2);
}

UGKWP_SST_HD inline double sstGamma
(
    const double f1,
    const SstCoefficients& coefficients
)
{
    return sstBlend(f1, coefficients.gamma1, coefficients.gamma2);
}

UGKWP_SST_HD inline double sstCrossDiffusion
(
    const double omega,
    const double gradKDotGradOmega,
    const SstCoefficients& coefficients
)
{
    const double omegaSafe = sstPositive(omega, 1.0e-300);
    return
        2.0*coefficients.alphaOmega2*gradKDotGradOmega/omegaSafe;
}

UGKWP_SST_HD inline double sstF1
(
    const double k,
    const double omega,
    const double nu,
    const double wallDistance,
    const double cdKOmega,
    const SstCoefficients& coefficients
)
{
    const double kSafe = sstPositive(k, 0.0);
    const double omegaSafe = sstPositive(omega, 1.0e-300);
    const double ySafe = sstPositive(wallDistance, 1.0e-300);
    const double ySquared = ySafe*ySafe;
    const double cdPlus = sstMaximum(cdKOmega, 1.0e-10);
    const double viscousArgument = 500.0*nu/(ySquared*omegaSafe);
    const double turbulentArgument =
        std::sqrt(kSafe)/(coefficients.betaStar*omegaSafe*ySafe);
    const double crossDiffusionArgument =
        4.0*coefficients.alphaOmega2*kSafe/(cdPlus*ySquared);
    const double argument = sstMinimum(
        sstMinimum(
            sstMaximum(turbulentArgument, viscousArgument),
            crossDiffusionArgument
        ),
        10.0
    );
    const double argumentSquared = argument*argument;
    return std::tanh(argumentSquared*argumentSquared);
}

UGKWP_SST_HD inline double sstF2
(
    const double k,
    const double omega,
    const double nu,
    const double wallDistance,
    const SstCoefficients& coefficients
)
{
    const double kSafe = sstPositive(k, 0.0);
    const double omegaSafe = sstPositive(omega, 1.0e-300);
    const double ySafe = sstPositive(wallDistance, 1.0e-300);
    const double ySquared = ySafe*ySafe;
    const double argument = sstMinimum(
        sstMaximum(
            2.0*std::sqrt(kSafe)
           /(coefficients.betaStar*omegaSafe*ySafe),
            500.0*nu/(ySquared*omegaSafe)
        ),
        100.0
    );
    return std::tanh(argument*argument);
}

UGKWP_SST_HD inline double sstNut
(
    const double k,
    const double omega,
    const double s2,
    const double f2,
    const SstCoefficients& coefficients
)
{
    const double denominator = sstMaximum(
        coefficients.a1*sstPositive(omega, 1.0e-300),
        coefficients.b1*f2*std::sqrt(sstPositive(s2, 0.0))
    );
    return coefficients.a1*sstPositive(k, 0.0)/denominator;
}

UGKWP_SST_HD inline double sstKProduction
(
    const double k,
    const double omega,
    const double nut,
    const double gByNu,
    const SstCoefficients& coefficients
)
{
    return sstMinimum(
        nut*gByNu,
        coefficients.c1*coefficients.betaStar
       *sstPositive(k, 0.0)*sstPositive(omega, 0.0)
    );
}

UGKWP_SST_HD inline double sstKSource
(
    const double rho,
    const double k,
    const double omega,
    const double divU,
    const double nut,
    const double gByNu,
    const SstCoefficients& coefficients
)
{
    return rho*(
        sstKProduction(k, omega, nut, gByNu, coefficients)
      - (2.0/3.0)*divU*k
      - coefficients.betaStar*k*omega
    );
}

UGKWP_SST_HD inline double sstOmegaProductionLimit
(
    const double omega,
    const double s2,
    const double f2,
    const SstCoefficients& coefficients
)
{
    return
        (coefficients.c1/coefficients.a1)
       *coefficients.betaStar*omega
       *sstMaximum(
            coefficients.a1*omega,
            coefficients.b1*f2*std::sqrt(sstPositive(s2, 0.0))
        );
}

UGKWP_SST_HD inline double sstOmegaSource
(
    const double rho,
    const double k,
    const double omega,
    const double divU,
    const double gByNu,
    const double s2,
    const double f1,
    const double f2,
    const double cdKOmega,
    const SstCoefficients& coefficients
)
{
    (void)k;
    const double gamma = sstGamma(f1, coefficients);
    const double beta = sstBeta(f1, coefficients);
    const double production = gamma*sstMinimum(
        gByNu,
        sstOmegaProductionLimit(omega, s2, f2, coefficients)
    );
    return rho*(
        production
      - (2.0/3.0)*gamma*divU*omega
      - beta*omega*omega
      + (1.0 - f1)*cdKOmega
    );
}

UGKWP_SST_HD inline double sstLowReWallOmega
(
    const double nu,
    const double wallDistance,
    const SstCoefficients& coefficients
)
{
    const double ySafe = sstPositive(wallDistance, 1.0e-300);
    return 6.0*nu/(coefficients.beta1*ySafe*ySafe);
}

}                   

#undef UGKWP_SST_HD

#endif
