#include "common/gasNumerics/GpuDragAlgebra.cuh"
#include "common/gasNumerics/GpuLesAlgebra.cuh"
#include "common/gasNumerics/GpuParticlePhysicsAlgebra.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace
{

bool sameBits(const double left, const double right)
{
    std::uint64_t leftBits = 0;
    std::uint64_t rightBits = 0;
    std::memcpy(&leftBits, &left, sizeof(double));
    std::memcpy(&rightBits, &right, sizeof(double));
    return leftBits == rightBits;
}

void requireSame(const double legacy, const double common, const char* name)
{
    if (!sameBits(legacy, common))
    {
        std::cerr << name << " differs: " << legacy << " " << common << '\n';
        std::exit(1);
    }
}

double legacyRanzFromPr(const double re, const double pr)
{
    return 2.0 + 0.6*sqrt(re)*pow(pr, 1.0/3.0);
}

double legacyRanzFromPrOneThird(const double re, const double prOneThird)
{
    return 2.0 + 0.6*sqrt(re)*prOneThird;
}

double legacyRadialDistribution(const double ratio)
{
    return (2.0 - ratio)/(2.0*pow(1.0 - ratio, 3.0) + 1.0e-5);
}

double legacyCollisionalPressure
(
    const double restitution,
    const double density,
    const double eps,
    const double g0,
    const double theta
)
{
    return 2.0*(1.0 + restitution)*density*eps*eps*g0*theta;
}

double legacyMeanFreePath
(
    const double pi,
    const double diameter,
    const double eps,
    const double g0,
    const double small
)
{
    return sqrt(pi)*diameter/(12.0*eps*g0 + small);
}

double legacyCollisionTime
(
    const double meanFreePath,
    const double theta,
    const double small
)
{
    return meanFreePath/(sqrt(theta) + small);
}

double legacySmagorinsky
(
    const double coefficient,
    const double delta,
    const double ss
)
{
    return coefficient*coefficient*delta*delta*sqrt(fmax(2.0*ss, 0.0));
}

double legacyWale
(
    const double coefficient,
    const double delta,
    const double symmetricGradientSquared,
    const double sd2,
    const double small
)
{
    const double numerator = pow(fmax(sd2, 0.0), 1.5);
    const double denominator =
        pow(fmax(symmetricGradientSquared, 0.0), 2.5)
      + pow(fmax(sd2, 0.0), 1.25);
    return denominator > small
      ? coefficient*coefficient*delta*delta*numerator/denominator
      : 0.0;
}

double legacyGasSchiller
(
    const double rho,
    const double mu,
    const double rhoSolid,
    const double diameter,
    const double speed,
    const double regularization
)
{
    const double re = rho*diameter*speed/fmax(mu, 1.0e-30);
    const double reSafe = fmax(re, 1.0e-12);
    const double cd = reSafe < 1000.0
      ? 24.0/reSafe*(1.0 + 0.15*pow(reSafe, 0.687))
      : 0.44;
    return 0.75*cd*rho*speed/(rhoSolid*diameter + regularization);
}

double legacyGasGidaspow
(
    const double rho,
    const double mu,
    const double alphaInput,
    const double rhoSolid,
    const double diameterInput,
    const double speed,
    const double regularization,
    const double residualRe
)
{
    const double alpha = fmin(fmax(alphaInput, 1.0e-12), 1.0);
    const double re = fmax
    (
        rho*diameterInput*speed/fmax(mu, 1.0e-30),
        0.0
    );
    double cdRe = 0.0;
    if (alpha >= 0.8)
    {
        const double dispersedRe = alpha*re;
        const double cdsRe = dispersedRe < 1000.0
          ? 24.0*(1.0 + 0.15*pow(dispersedRe, 0.687))
          : 0.44*fmax(dispersedRe, residualRe);
        cdRe = cdsRe*pow(alpha, -2.65);
    }
    else
    {
        cdRe = (4.0/3.0)*(150.0*(1.0 - alpha)/alpha + 1.75*re);
    }
    const double diameter = fmax(diameterInput, 1.0e-30);
    return
        0.75*cdRe*fmax(mu, 1.0e-30)
       /(rhoSolid*diameter*diameter + regularization);
}

double legacyFshSchiller
(
    const double rhoInput,
    const double muInput,
    const double rhoSolidInput,
    const double diameterInput,
    const double speedInput
)
{
    const double mu = fmax(muInput, 1.0e-30);
    const double re =
        fmax(rhoInput, 0.0)*fmax(diameterInput, 1.0e-30)
       *fmax(speedInput, 0.0)/mu;
    if (re <= 1.0e-30 || speedInput <= 1.0e-30)
    {
        return 0.0;
    }
    const double cd = re < 1000.0
      ? 24.0*(1.0 + 0.15*pow(re, 0.687))/re
      : 0.44;
    return
        0.75*cd*fmax(rhoInput, 0.0)*fmax(speedInput, 0.0)
       /(fmax(rhoSolidInput, 1.0e-30)*fmax(diameterInput, 1.0e-30));
}

double legacyFshGidaspow
(
    const double rhoInput,
    const double alphaInput,
    const double muInput,
    const double rhoSolidInput,
    const double diameterInput,
    const double speedInput,
    const double residualRe
)
{
    const double alpha = fmin(fmax(alphaInput, 1.0e-12), 1.0);
    const double mu = fmax(muInput, 1.0e-30);
    const double diameter = fmax(diameterInput, 1.0e-30);
    const double re =
        fmax(rhoInput, 0.0)*diameter*fmax(speedInput, 0.0)/mu;
    const double alphaRe = alpha*re;
    const double cdReWenYu = alphaRe < 1000.0
      ? 24.0*(1.0 + 0.15*pow(fmax(alphaRe, 0.0), 0.687))
      : 0.44*fmax(alphaRe, residualRe);
    const double cdRe = alpha >= 0.8
      ? cdReWenYu*pow(alpha, -2.65)
      : (4.0/3.0)*(150.0*(1.0 - alpha)/alpha + 1.75*re);
    return
        0.75*cdRe*mu
       /(fmax(rhoSolidInput, 1.0e-30)*diameter*diameter);
}

}

int main()
{
    const double reynolds[] = {0.0, 1.0e-12, 1.0, 999.999999, 1000.0, 2.0e5};
    const double prandtl[] = {1.0e-12, 0.4, 0.71, 10.0};
    for (const double re : reynolds)
    {
        for (const double pr : prandtl)
        {
            requireSame
            (
                legacyRanzFromPr(re, pr),
                ugkwp::ranzMarshallNuFromPr(re, pr),
                "RanzMarshallPr"
            );
            const double prOneThird = pow(pr, 1.0/3.0);
            requireSame
            (
                legacyRanzFromPrOneThird(re, prOneThird),
                ugkwp::ranzMarshallNuFromPrOneThird(re, prOneThird),
                "RanzMarshallPrOneThird"
            );
        }
    }

    const double ratios[] = {0.0, 0.1, 0.5, 0.9, 0.99};
    for (const double ratio : ratios)
    {
        requireSame
        (
            legacyRadialDistribution(ratio),
            ugkwp::radialDistributionG0FromRatio(ratio),
            "radialDistribution"
        );
    }
    const double g0 = legacyRadialDistribution(0.8);
    requireSame
    (
        legacyCollisionalPressure(0.9, 2500.0, 0.4, g0, 12.0),
        ugkwp::collisionalPressure(0.9, 2500.0, 0.4, g0, 12.0),
        "collisionalPressure"
    );
    const double meanFreePath = legacyMeanFreePath
    (
        3.141592653589793238462643383279502884,
        1.2e-4,
        0.4,
        g0,
        2.22044604925031308085e-16
    );
    requireSame
    (
        meanFreePath,
        ugkwp::granularMeanFreePath
        (
            3.141592653589793238462643383279502884,
            1.2e-4,
            0.4,
            g0,
            2.22044604925031308085e-16
        ),
        "meanFreePath"
    );
    requireSame
    (
        legacyCollisionTime(meanFreePath, 12.0, 2.22044604925031308085e-16),
        ugkwp::granularCollisionTime
        (
            meanFreePath,
            12.0,
            2.22044604925031308085e-16
        ),
        "collisionTime"
    );
    requireSame
    (
        legacySmagorinsky(0.17, 2.0e-3, 15.0),
        ugkwp::smagorinskyNut(0.17, 2.0e-3, 15.0),
        "Smagorinsky"
    );
    requireSame
    (
        legacyWale(0.325, 2.0e-3, 12.0, 7.0, 2.22044604925031308085e-16),
        ugkwp::waleNut(0.325, 2.0e-3, 12.0, 7.0, 2.22044604925031308085e-16),
        "WALE"
    );

    const double alphaValues[] = {1.0e-12, 0.79, 0.8, 1.0};
    const double speeds[] = {0.0, 1.0e-12, 0.5, 15.0, 50.0};
    for (const double alpha : alphaValues)
    {
        for (const double speed : speeds)
        {
            requireSame
            (
                legacyGasSchiller(1.2, 1.8e-5, 2500.0, 3.0e-4, speed, 1.0e-300),
                ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannInverseResponseTime
                (1.2, 1.8e-5, 2500.0, 3.0e-4, speed, 1.0e-300),
                "gasSchiller"
            );
            requireSame
            (
                legacyGasGidaspow(1.2, 1.8e-5, alpha, 2500.0, 3.0e-4, speed, 1.0e-300, 1.0e-3),
                ugkwpGpuDragAlgebra::gasUgkpGidaspowInverseResponseTime
                (1.2, 1.8e-5, alpha, 2500.0, 3.0e-4, speed, 1.0e-300, 1.0e-3),
                "gasGidaspow"
            );
            requireSame
            (
                legacyFshSchiller(1.2, 1.8e-5, 2500.0, 3.0e-4, speed),
                ugkwpGpuDragAlgebra::fshChtSchillerNaumannInverseRelaxationTime
                (1.2, 1.8e-5, 2500.0, 3.0e-4, speed),
                "fshChtSchiller"
            );
            requireSame
            (
                legacyFshGidaspow(1.2, alpha, 1.8e-5, 2500.0, 3.0e-4, speed, 1.0e-3),
                ugkwpGpuDragAlgebra::fshChtGidaspowInverseRelaxationTime
                (1.2, alpha, 1.8e-5, 2500.0, 3.0e-4, speed, 1.0e-3),
                "fshChtGidaspow"
            );
        }
    }
    return 0;
}
