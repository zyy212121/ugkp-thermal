#ifndef UGKWP_GPU_PRECISION_TYPES_H
#include "GpuPrecisionTypes.H"
#endif
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

template<class Real> struct SstCoefficientsT
{
    Real alphaK1;
    Real alphaK2;
    Real alphaOmega1;
    Real alphaOmega2;
    Real beta1;
    Real beta2;
    Real betaStar;
    Real gamma1;
    Real gamma2;
    Real a1;
    Real b1;
    Real c1;
    template<class Other> UGKWP_SST_HD SstCoefficientsT& operator=(const SstCoefficientsT<Other>& other){
        alphaK1=Real(other.alphaK1);
        alphaK2=Real(other.alphaK2);
        alphaOmega1=Real(other.alphaOmega1);
        alphaOmega2=Real(other.alphaOmega2);
        beta1=Real(other.beta1);
        beta2=Real(other.beta2);
        betaStar=Real(other.betaStar);
        gamma1=Real(other.gamma1);
        gamma2=Real(other.gamma2);
        a1=Real(other.a1);
        b1=Real(other.b1);
        c1=Real(other.c1);
        return *this;
    }
};
using SstCoefficients=SstCoefficientsT<GpuReal>;
template<class Real> UGKWP_SST_HD inline SstCoefficientsT<Real> defaultSstCoefficientsT()
{
    return SstCoefficientsT<Real>{
        Real(0.85),
        Real(1.0),
        Real(0.5),
        Real(0.856),
        Real(0.075),
        Real(0.0828),
        Real(0.09),
        Real(5.0)/Real(9.0),
        Real(0.44),
        Real(0.31),
        Real(1.0),
        Real(10.0)
    };
}

UGKWP_SST_HD inline SstCoefficients defaultSstCoefficients()
{ return defaultSstCoefficientsT<GpuReal>(); }

template<class Real> UGKWP_SST_HD inline Real sstMaximum(const Real a, const Real b)
{
    return a > b ? a : b;
}

UGKWP_SST_HD inline GpuReal sstMaximum(const GpuReal a, const GpuReal b)
{ return sstMaximum<GpuReal>(a, b); }

template<class Real> UGKWP_SST_HD inline Real sstMinimum(const Real a, const Real b)
{
    return a < b ? a : b;
}

UGKWP_SST_HD inline GpuReal sstMinimum(const GpuReal a, const GpuReal b)
{ return sstMinimum<GpuReal>(a, b); }

template<class Real> UGKWP_SST_HD inline Real sstPositive(const Real value, const Real floor)
{
    return sstMaximum(value, floor);
}

UGKWP_SST_HD inline GpuReal sstPositive(const GpuReal value, const GpuReal floor)
{ return sstPositive<GpuReal>(value, floor); }

template<class Real> UGKWP_SST_HD inline Real sstBlend
(
    const Real f1,
    const Real value1,
    const Real value2
)
{
    return f1*(value1 - value2) + value2;
}

UGKWP_SST_HD inline GpuReal sstBlend
(
    const GpuReal f1,
    const GpuReal value1,
    const GpuReal value2
)
{ return sstBlend<GpuReal>(f1, value1, value2); }

template<class Real> UGKWP_SST_HD inline Real sstAlphaK
(
    const Real f1,
    const SstCoefficientsT<Real>& coefficients
)
{
    return sstBlend(f1, coefficients.alphaK1, coefficients.alphaK2);
}

UGKWP_SST_HD inline GpuReal sstAlphaK
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{ return sstAlphaK<GpuReal>(f1, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstAlphaOmega
(
    const Real f1,
    const SstCoefficientsT<Real>& coefficients
)
{
    return sstBlend(
        f1,
        coefficients.alphaOmega1,
        coefficients.alphaOmega2
    );
}

UGKWP_SST_HD inline GpuReal sstAlphaOmega
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{ return sstAlphaOmega<GpuReal>(f1, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstBeta
(
    const Real f1,
    const SstCoefficientsT<Real>& coefficients
)
{
    return sstBlend(f1, coefficients.beta1, coefficients.beta2);
}

UGKWP_SST_HD inline GpuReal sstBeta
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{ return sstBeta<GpuReal>(f1, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstGamma
(
    const Real f1,
    const SstCoefficientsT<Real>& coefficients
)
{
    return sstBlend(f1, coefficients.gamma1, coefficients.gamma2);
}

UGKWP_SST_HD inline GpuReal sstGamma
(
    const GpuReal f1,
    const SstCoefficients& coefficients
)
{ return sstGamma<GpuReal>(f1, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstCrossDiffusion
(
    const Real omega,
    const Real gradKDotGradOmega,
    const SstCoefficientsT<Real>& coefficients
)
{
    const Real omegaSafe = sstPositive(omega, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    return
        Real(2.0)*coefficients.alphaOmega2*gradKDotGradOmega/omegaSafe;
}

UGKWP_SST_HD inline GpuReal sstCrossDiffusion
(
    const GpuReal omega,
    const GpuReal gradKDotGradOmega,
    const SstCoefficients& coefficients
)
{ return sstCrossDiffusion<GpuReal>(omega, gradKDotGradOmega, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstF1
(
    const Real k,
    const Real omega,
    const Real nu,
    const Real wallDistance,
    const Real cdKOmega,
    const SstCoefficientsT<Real>& coefficients
)
{
    const Real kSafe = sstPositive(k, Real(0.0));
    const Real omegaSafe = sstPositive(omega, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    const Real ySafe = sstPositive(wallDistance, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    const Real ySquared = ySafe*ySafe;
    const Real cdPlus = sstMaximum(cdKOmega, Real(1.0e-10));
    const Real viscousArgument = Real(500.0)*nu/(ySquared*omegaSafe);
    const Real turbulentArgument =
        std::sqrt(kSafe)/(coefficients.betaStar*omegaSafe*ySafe);
    const Real crossDiffusionArgument =
        Real(4.0)*coefficients.alphaOmega2*kSafe/(cdPlus*ySquared);
    const Real argument = sstMinimum(
        sstMinimum(
            sstMaximum(turbulentArgument, viscousArgument),
            crossDiffusionArgument
        ),
        Real(10.0)
    );
    const Real argumentSquared = argument*argument;
    return std::tanh(argumentSquared*argumentSquared);
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
{ return sstF1<GpuReal>(k, omega, nu, wallDistance, cdKOmega, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstF2
(
    const Real k,
    const Real omega,
    const Real nu,
    const Real wallDistance,
    const SstCoefficientsT<Real>& coefficients
)
{
    const Real kSafe = sstPositive(k, Real(0.0));
    const Real omegaSafe = sstPositive(omega, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    const Real ySafe = sstPositive(wallDistance, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    const Real ySquared = ySafe*ySafe;
    const Real argument = sstMinimum(
        sstMaximum(
            Real(2.0)*std::sqrt(kSafe)
           /(coefficients.betaStar*omegaSafe*ySafe),
            Real(500.0)*nu/(ySquared*omegaSafe)
        ),
        Real(100.0)
    );
    return std::tanh(argument*argument);
}

UGKWP_SST_HD inline GpuReal sstF2
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal nu,
    const GpuReal wallDistance,
    const SstCoefficients& coefficients
)
{ return sstF2<GpuReal>(k, omega, nu, wallDistance, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstNut
(
    const Real k,
    const Real omega,
    const Real s2,
    const Real f2,
    const SstCoefficientsT<Real>& coefficients
)
{
    const Real denominator = sstMaximum(
        coefficients.a1*sstPositive(omega, (sizeof(Real)==4?Real(1e-30):Real(1e-300))),
        coefficients.b1*f2*std::sqrt(sstPositive(s2, Real(0.0)))
    );
    return coefficients.a1*sstPositive(k, Real(0.0))/denominator;
}

UGKWP_SST_HD inline GpuReal sstNut
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal s2,
    const GpuReal f2,
    const SstCoefficients& coefficients
)
{ return sstNut<GpuReal>(k, omega, s2, f2, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstKProduction
(
    const Real k,
    const Real omega,
    const Real nut,
    const Real gByNu,
    const SstCoefficientsT<Real>& coefficients
)
{
    return sstMinimum(
        nut*gByNu,
        coefficients.c1*coefficients.betaStar
       *sstPositive(k, Real(0.0))*sstPositive(omega, Real(0.0))
    );
}

UGKWP_SST_HD inline GpuReal sstKProduction
(
    const GpuReal k,
    const GpuReal omega,
    const GpuReal nut,
    const GpuReal gByNu,
    const SstCoefficients& coefficients
)
{ return sstKProduction<GpuReal>(k, omega, nut, gByNu, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstKSource
(
    const Real rho,
    const Real k,
    const Real omega,
    const Real divU,
    const Real nut,
    const Real gByNu,
    const SstCoefficientsT<Real>& coefficients
)
{
    return rho*(
        sstKProduction(k, omega, nut, gByNu, coefficients)
      - (Real(2.0)/Real(3.0))*divU*k
      - coefficients.betaStar*k*omega
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
{ return sstKSource<GpuReal>(rho, k, omega, divU, nut, gByNu, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstOmegaProductionLimit
(
    const Real omega,
    const Real s2,
    const Real f2,
    const SstCoefficientsT<Real>& coefficients
)
{
    return
        (coefficients.c1/coefficients.a1)
       *coefficients.betaStar*omega
       *sstMaximum(
            coefficients.a1*omega,
            coefficients.b1*f2*std::sqrt(sstPositive(s2, Real(0.0)))
        );
}

UGKWP_SST_HD inline GpuReal sstOmegaProductionLimit
(
    const GpuReal omega,
    const GpuReal s2,
    const GpuReal f2,
    const SstCoefficients& coefficients
)
{ return sstOmegaProductionLimit<GpuReal>(omega, s2, f2, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstOmegaSource
(
    const Real rho,
    const Real k,
    const Real omega,
    const Real divU,
    const Real gByNu,
    const Real s2,
    const Real f1,
    const Real f2,
    const Real cdKOmega,
    const SstCoefficientsT<Real>& coefficients
)
{
    (void)k;
    const Real gamma = sstGamma(f1, coefficients);
    const Real beta = sstBeta(f1, coefficients);
    const Real production = gamma*sstMinimum(
        gByNu,
        sstOmegaProductionLimit(omega, s2, f2, coefficients)
    );
    return rho*(
        production
      - (Real(2.0)/Real(3.0))*gamma*divU*omega
      - beta*omega*omega
      + (Real(1.0) - f1)*cdKOmega
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
{ return sstOmegaSource<GpuReal>(rho, k, omega, divU, gByNu, s2, f1, f2, cdKOmega, coefficients); }

template<class Real> UGKWP_SST_HD inline Real sstLowReWallOmega
(
    const Real nu,
    const Real wallDistance,
    const SstCoefficientsT<Real>& coefficients
)
{
    const Real ySafe = sstPositive(wallDistance, (sizeof(Real)==4?Real(1e-30):Real(1e-300)));
    return Real(6.0)*nu/(coefficients.beta1*ySafe*ySafe);
}

UGKWP_SST_HD inline GpuReal sstLowReWallOmega
(
    const GpuReal nu,
    const GpuReal wallDistance,
    const SstCoefficients& coefficients
)
{ return sstLowReWallOmega<GpuReal>(nu, wallDistance, coefficients); }


}                   

#undef UGKWP_SST_HD

#endif
