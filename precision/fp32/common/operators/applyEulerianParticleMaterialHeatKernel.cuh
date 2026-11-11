#pragma once
#include "ParticleResidentInterval.H"
#include "gasTransport/GasEnergy.H"
#include "GpuParticleHeatMomentView.cuh"
__global__ void applyEulerianParticleMaterialHeatKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME requestedDt
)
{
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
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

    const auto heatMoments=particleGasHeatMoments(s,c,0);
#if !GPU_OPERATOR_THERMAL
    const double heatFactor = particleHeatFactorDevice(s);
    if (heatFactor <= GPU_OPERATOR_R(0.0)) return;
#endif
    const double rhoP = clampMin(finiteOr(heatMoments.mass, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }
    const double epsG = GPU_OPERATOR_R(1.0) - solidEpsFromMomentDevice(s, c);
    const double epsGsafe = clampMin(epsG, OfSmall);

    const double hp = clampMin(finiteOr(heatMoments.enthalpy, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (hp <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

#if GPU_OPERATOR_THERMAL
    const double specificEnthalpy = finiteOr(hp/(rhoP + OfSmall), -GPU_OPERATOR_R(1.0));
    const double tpAvg =
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


    const double particleCp = tpAvg > GPU_OPERATOR_R(0.0)
        && Foam::gpuThermal::finiteAluminaValue(tpAvg)
      ? particleSpecificHeatDevice(tpAvg) : GPU_OPERATOR_R(0.0);
    if (!(particleCp > GPU_OPERATOR_R(0.0)))
    {
        return;
    }

    const double rateCapacity = s.rhoSolid*particleCp;
#else
    const double tpAvg = clampRange
    (
        finiteOr(hp/(rhoP*heatFactor + OfSmall), s.TpMin),
        s.TpMin, s.TpMax
    );
    const double particleCp = heatFactor;
    const double rateCapacity = s.particleThermalRho*s.particleCp;
#endif
    const double rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], GPU_OPERATOR_R(0.0)), s.rhoMin);
    const double tg =
        clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            GPU_OPERATOR_R(1.0e30)
        );

#if GPU_OPERATOR_THERMAL
    const double rhoDP = clampMin(finiteOr(heatMoments.diameter, GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
#else
    const double rhoDP = heatMoments.diameter;
#endif
    const double dLocal =
        clampMin
        (
            finiteOr(rhoDP/rhoP, s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );

    const double usx = finiteOr(heatMoments.px, GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);
    const double usy = finiteOr(heatMoments.py, GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);
    const double usz = finiteOr(heatMoments.pz, GPU_OPERATOR_R(0.0))/(rhoP + OfSmall);

    const double urx = finiteOr(s.couplingUxOld[c], GPU_OPERATOR_R(0.0)) - usx;
    const double ury = finiteOr(s.couplingUyOld[c], GPU_OPERATOR_R(0.0)) - usy;
    const double urz = finiteOr(s.couplingUzOld[c], GPU_OPERATOR_R(0.0)) - usz;
    const double urMag = sqrt(sqr3(urx, ury, urz));

    const double mu = clampMin(ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]), GPU_OPERATOR_R(1.0e-30));
    const double re = rhoG*dLocal*urMag/mu;

    const double pr = ugkwp::gasCellMolecularPrandtl(s,c);
    const double gasConductivity = molecularGasConductivity(s,c,s.couplingTgasOld[c]);

    const double nu = ugkwp::ranzMarshallNuFromPr
    (
        clampMin(re, GPU_OPERATOR_R(0.0)),
        clampMin(pr, GPU_OPERATOR_R(1.0e-12))
    );

    const double rate =
        GPU_OPERATOR_R(6.0)*nu*gasConductivity
       /(rateCapacity*dLocal*dLocal + GPU_OPERATOR_TINY(1.0e-300));

    const double gasCv =
        s.Rgas/clampMin(s.gammaGas - GPU_OPERATOR_R(1.0), OfSmall);
    const double gasCapacity = epsG*rhoG*gasCv;
    const double solidCapacity = rhoP*particleCp;

                                                                           
                                                                             
                                                                        
    const double dHp =
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

    const double rhoGCurrent =
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
