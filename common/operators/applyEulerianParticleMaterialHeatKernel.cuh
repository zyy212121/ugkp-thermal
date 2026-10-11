#pragma once
#include "gasTransport/GasEnergy.H"
__global__ void applyEulerianParticleMaterialHeatKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    using Energy = ugkwp::GasEnergyScalar<DeviceState>;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if
    (
        c >= s.nCells
     || s.solveParticleTemperature == 0
     || s.particleGasHeatTransferModelId == 0
    )
    {
        return;
    }

#if !GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL heatFactor = particleHeatFactorDevice(s);
    if (heatFactor <= GPU_OPERATOR_R(0.0)) return;
#endif
    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }
    const GPU_OPERATOR_REAL epsG = GPU_OPERATOR_R(1.0) - solidEpsFromMomentDevice(s, c);
    const GPU_OPERATOR_REAL epsGsafe = clampMin(epsG, OfSmall);

    const GPU_OPERATOR_REAL hp = clampMin(finiteOr(s.momRhoHpP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (hp <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

#if GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL specificEnthalpy = finiteOr(hp/(rhoP + OfSmall), -GPU_OPERATOR_R(1.0));
    const GPU_OPERATOR_REAL tpAvg =
        clampRange
        (
            finiteOr
            (
                particleTemperatureFromSpecificEnthalpyDevice
                (
                    specificEnthalpy
                ),
                s.TpMin
            ),
            s.TpMin,
            s.TpMax
        );
    const GPU_OPERATOR_REAL particleCp = particleSpecificHeatDevice(tpAvg);
    if (!(particleCp > GPU_OPERATOR_R(0.0)))
    {
        return;
    }

    const GPU_OPERATOR_REAL rateCapacity = s.rhoSolid*particleCp;
#else
    const GPU_OPERATOR_REAL tpAvg = clampRange
    (
        finiteOr(hp/(rhoP*heatFactor + OfSmall), s.TpMin),
        s.TpMin, s.TpMax
    );
    const GPU_OPERATOR_REAL particleCp = heatFactor;
    const GPU_OPERATOR_REAL rateCapacity = s.particleThermalRho*s.particleCp;
#endif
    const GPU_OPERATOR_REAL rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], GPU_OPERATOR_R(0.0)), s.rhoMin);
    const GPU_OPERATOR_REAL tg =
        clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            GPU_OPERATOR_R(1.0e30)
        );

#if GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL rhoDP = clampMin(finiteOr(s.momRhoPD[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
#else
    const GPU_OPERATOR_REAL rhoDP = s.momRhoPD[c];
#endif
    const GPU_OPERATOR_REAL dLocal =
        clampMin
        (
            finiteOr(rhoDP/rhoP, s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );

    const GPU_OPERATOR_REAL usx = finiteOr(s.momRhoUPx[c], GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);
    const GPU_OPERATOR_REAL usy = finiteOr(s.momRhoUPy[c], GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);
    const GPU_OPERATOR_REAL usz = finiteOr(s.momRhoUPz[c], GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);

    const GPU_OPERATOR_REAL urx = finiteOr(s.couplingUxOld[c], GPU_OPERATOR_R(0.0)) - usx;
    const GPU_OPERATOR_REAL ury = finiteOr(s.couplingUyOld[c], GPU_OPERATOR_R(0.0)) - usy;
    const GPU_OPERATOR_REAL urz = finiteOr(s.couplingUzOld[c], GPU_OPERATOR_R(0.0)) - usz;
    const GPU_OPERATOR_REAL urMag = sqrt(sqr3(urx, ury, urz));

    const GPU_OPERATOR_REAL mu = clampMin(s.gasMu, GPU_OPERATOR_R(1.0e-30));
    const GPU_OPERATOR_REAL re = rhoG*dLocal*urMag/mu;

    const GPU_OPERATOR_REAL pr = s.gasPrClamped;
    const GPU_OPERATOR_REAL gasConductivity = molecularGasConductivity(s);

    const GPU_OPERATOR_REAL nu = ugkwp::ranzMarshallNuFromPr
    (
        clampMin(re, GPU_OPERATOR_R(0.0)),
        clampMin(pr, GPU_OPERATOR_R(1.0e-12))
    );

    const GPU_OPERATOR_REAL rate =
        GPU_OPERATOR_R(6.0)*nu*gasConductivity
       /(rateCapacity*dLocal*dLocal + GPU_OPERATOR_TINY(1.0e-300));

    const GPU_OPERATOR_REAL gasCv =
        s.Rgas/clampMin(s.gammaGas - GPU_OPERATOR_R(1.0), OfSmall);
    const GPU_OPERATOR_REAL gasCapacity = epsG*rhoG*gasCv;
    const GPU_OPERATOR_REAL solidCapacity = rhoP*particleCp;

                                                                           
                                                                             
                                                                        
    const GPU_OPERATOR_REAL dHp =
        gpuFiniteCapacityHeatExchange
        (
            gasCapacity,
            solidCapacity,
            tg,
            tpAvg,
            clampMin(rate, GPU_OPERATOR_R(0.0)),
            dt
        );

    const Energy rhoEold = ugkwp::gasEnergyFiniteOr(s.rhoE[c], Energy(0));
    Energy rhoEnew = rhoEold - Energy(dHp)/Energy(epsGsafe);

    const GPU_OPERATOR_REAL rhoGCurrent =
        clampMin(finiteOr(s.rho[c], rhoG), s.rhoMin);
    const Energy mx=Energy(finiteOr(s.rhoUx[c],GPU_OPERATOR_R(0))),
        my=Energy(finiteOr(s.rhoUy[c],GPU_OPERATOR_R(0))),
        mz=Energy(finiteOr(s.rhoUz[c],GPU_OPERATOR_R(0)));
    const Energy kinetic=Energy(0.5)*(mx*mx+my*my+mz*mz)/Energy(rhoGCurrent);
    const Energy eMinGas=Energy(rhoGCurrent)*Energy(s.Rgas)*Energy(s.TgasMin)
        /ugkwp::gasEnergyClampMin(Energy(s.gammaGas)-Energy(1),Energy(OfSmall));
    rhoEnew=ugkwp::gasEnergyClampMin(ugkwp::gasEnergyFiniteOr(rhoEnew,rhoEold),kinetic+eMinGas);
    s.rhoE[c] = rhoEnew;
}
