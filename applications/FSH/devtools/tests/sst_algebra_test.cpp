#include <cassert>
#include <cmath>

#include "../../private_backend/GpuSstAlgebra.cuh"
#include "../../private_backend/OpenFoamWallFunctions.cuh"

namespace
{
bool near(const double a, const double b, const double tolerance = 1.0e-12)
{
    return std::abs(a - b) <= tolerance*std::max(1.0, std::abs(b));
}
}

int main()
{
    const ugkwp::SstCoefficients c = ugkwp::defaultSstCoefficients();
    assert(near(c.alphaK1, 0.85));
    assert(near(c.alphaK2, 1.0));
    assert(near(c.alphaOmega1, 0.5));
    assert(near(c.alphaOmega2, 0.856));
    assert(near(c.beta1, 0.075));
    assert(near(c.beta2, 0.0828));
    assert(near(c.betaStar, 0.09));
    assert(near(c.gamma1, 5.0/9.0));
    assert(near(c.gamma2, 0.44));
    assert(near(c.a1, 0.31));
    assert(near(c.b1, 1.0));
    assert(near(c.c1, 10.0));

    const double k = 12.5;
    const double omega = 3200.0;
    const double nu = 1.7e-5;
    const double y = 2.5e-4;
    const double gradKDotGradOmega = 4.5e7;
    const double cd = 2.0*c.alphaOmega2*gradKDotGradOmega/omega;
    const double f1 = ugkwp::sstF1(k, omega, nu, y, cd, c);
    const double f2 = ugkwp::sstF2(k, omega, nu, y, c);
    assert(f1 >= 0.0 && f1 <= 1.0);
    assert(f2 >= 0.0 && f2 <= 1.0);

    const double alphaK = f1*(c.alphaK1 - c.alphaK2) + c.alphaK2;
    const double alphaOmega =
        f1*(c.alphaOmega1 - c.alphaOmega2) + c.alphaOmega2;
    assert(near(ugkwp::sstAlphaK(f1, c), alphaK));
    assert(near(ugkwp::sstAlphaOmega(f1, c), alphaOmega));

    const double s2 = 8.0e5;
    const double nutReference =
        c.a1*k/std::max(c.a1*omega, c.b1*f2*std::sqrt(s2));
    assert(near(ugkwp::sstNut(k, omega, s2, f2, c), nutReference));

    const double gByNu = 7.5e5;
    const double productionReference =
        std::min(nutReference*gByNu, c.c1*c.betaStar*k*omega);
    assert(near(
        ugkwp::sstKProduction(k, omega, nutReference, gByNu, c),
        productionReference
    ));

    const double rho = 3.2;
    const double divU = -200.0;
    const double sourceKReference = rho*(
        productionReference
      - (2.0/3.0)*divU*k
      - c.betaStar*k*omega
    );
    assert(near(
        ugkwp::sstKSource(rho, k, omega, divU, nutReference, gByNu, c),
        sourceKReference
    ));

    const double beta = f1*(c.beta1 - c.beta2) + c.beta2;
    const double gamma = f1*(c.gamma1 - c.gamma2) + c.gamma2;
    const double omegaProductionLimit =
        (c.c1/c.a1)*c.betaStar*omega
       *std::max(c.a1*omega, c.b1*f2*std::sqrt(s2));
    const double sourceOmegaReference = rho*(
        gamma*std::min(gByNu, omegaProductionLimit)
      - (2.0/3.0)*gamma*divU*omega
      - beta*omega*omega
      + (1.0 - f1)*cd
    );
    assert(near(
        ugkwp::sstOmegaSource(
            rho, k, omega, divU, gByNu, s2, f1, f2, cd, c
        ),
        sourceOmegaReference
    ));

    const double wallOmega = 6.0*nu/(c.beta1*y*y);
    assert(near(ugkwp::sstLowReWallOmega(nu, y, c), wallOmega));

    const double wallK = 10.0;
    const double wallY = 1.0e-3;
    const double wallNu = 1.5e-5;
    const double wallGradU = 1.0e5;
    const double Cmu = 0.09;
    const double kappa = 0.41;
    const double E = 9.8;
    const auto omegaWall = ugkpwall::omegaWallFunctionState
    (
        wallK, wallGradU, wallY, wallNu,
        c.beta1, Cmu, kappa, E
    );
    const double Cmu25 = std::pow(Cmu, 0.25);
    const double yPlusK = Cmu25*wallY*std::sqrt(wallK)/wallNu;
    const double uPlus = std::log(E*yPlusK)/kappa;
    const double uStar = Cmu25*std::sqrt(wallK);
    const double omegaLog =
        uStar/(std::sqrt(Cmu)*kappa*wallY);
    const double productionLog = std::pow
    (
        uStar*wallGradU*wallY/uPlus,
        2.0
    )/(wallNu*kappa*yPlusK);
    assert(near(omegaWall.yPlus, yPlusK));
    assert(near(omegaWall.omega, omegaLog));
    assert(near(omegaWall.production, productionLog));

    const auto spalding = ugkpwall::spaldingWallState
    (
        100.0, wallY, wallNu, kappa, E
    );
    assert(spalding.yPlus > 30.0);
    const auto hotWallHeat = ugkpwall::jayatillekeWallHeatFlux
    (
        3.0, 1200.0, 0.72, 0.85, kappa, E,
        spalding.uTau, spalding.yPlus, 1800.0, 750.0
    );
    const double Prat = 0.72/0.85;
    const double P =
        9.24*(std::pow(Prat, 0.75) - 1.0)
       *(1.0 + 0.28*std::exp(-0.007*Prat));
    const double expectedTPlus =
        0.85*(std::log(E*spalding.yPlus)/kappa + P);
    const double expectedHeatFlux =
        3.0*1200.0*spalding.uTau*(1800.0 - 750.0)
       /expectedTPlus;
    assert(near(hotWallHeat.temperaturePlus, expectedTPlus));
    assert(near(hotWallHeat.heatFlux, expectedHeatFlux));
    const double thermalYPlus = ugkpwall::jayatillekeThermalYPlus
    (
        Prat, kappa, E
    );
    const auto hotWallHeatPrecomputed =
        ugkpwall::jayatillekeWallHeatFluxPrecomputed
        (
            3.0, 1200.0, 0.72, 0.85, kappa, E,
            P, thermalYPlus,
            spalding.uTau, spalding.yPlus, 1800.0, 750.0
        );
    assert(near(hotWallHeatPrecomputed.temperaturePlus, expectedTPlus));
    assert(near(hotWallHeatPrecomputed.heatFlux, expectedHeatFlux));
    assert(near(hotWallHeatPrecomputed.yPlusThermal, thermalYPlus));

    const auto coldWallHeat = ugkpwall::jayatillekeWallHeatFlux
    (
        3.0, 1200.0, 0.72, 0.85, kappa, E,
        spalding.uTau, spalding.yPlus, 600.0, 750.0
    );
    assert(coldWallHeat.heatFlux < 0.0);
    assert(near(
        coldWallHeat.heatFlux/(-150.0),
        hotWallHeat.heatFlux/1050.0
    ));
    return 0;
}
