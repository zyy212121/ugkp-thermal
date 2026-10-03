#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void recoverPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    recoverGasPrimitiveCell(s, c);

    const GPU_OPERATOR_REAL eps = clampMin(finiteOr(s.epsS[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL solidMass = eps*s.rhoSolid;
    if (eps <= s.epsSMin || solidMass <= s.epsSMin*s.rhoSolid)
    {
        clearSolidCell(s, c);
        return;
    }

    const GPU_OPERATOR_REAL usx = finiteOr(s.rhoUsx[c]/solidMass, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL usy = finiteOr(s.rhoUsy[c]/solidMass, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL usz = finiteOr(s.rhoUsz[c]/solidMass, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL solidKinetic = GPU_OPERATOR_R(0.5)*sqr3(usx, usy, usz);
    GPU_OPERATOR_REAL theta =
        (finiteOr(s.rhoEs[c], GPU_OPERATOR_R(0.0))/solidMass - solidKinetic)/GPU_OPERATOR_R(1.5);
    if (!finiteDevice(theta) || theta < GPU_OPERATOR_R(0.0))
    {
        theta = GPU_OPERATOR_R(0.0);
        s.rhoEs[c] = solidMass*(solidKinetic + GPU_OPERATOR_R(1.5)*theta);
    }

    GPU_OPERATOR_REAL dMean = finiteOr(s.rhoDs[c]/solidMass, s.particleDiameterFallback);
    if (!finiteDevice(dMean) || dMean <= GPU_OPERATOR_R(0.0))
    {
        dMean = s.particleDiameterFallback;
        s.rhoDs[c] = solidMass*dMean;
    }
    dMean = clampMin(dMean, GPU_OPERATOR_R(1.0e-12));

#if GPU_OPERATOR_THERMAL
    if (s.solveParticleTemperature != 0)
    {
        const GPU_OPERATOR_REAL specificEnthalpy =
            finiteOr(s.rhoHp[c]/solidMass, -GPU_OPERATOR_R(1.0));
        GPU_OPERATOR_REAL Tp = particleTemperatureFromSpecificEnthalpyDevice
        (
            specificEnthalpy
        );
        Tp = finiteOr(Tp, s.TpMin);
        Tp = clampRange(Tp, s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        s.rhoHp[c] = solidMass*particleSpecificEnthalpyDevice(Tp);
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = GPU_OPERATOR_R(0.0);
    }

#else
    const GPU_OPERATOR_REAL heatFactor = particleHeatFactorDevice(s);
    if (heatFactor > GPU_OPERATOR_R(0.0))
    {
        GPU_OPERATOR_REAL Tp = finiteOr(s.rhoHp[c]/(solidMass*heatFactor), s.TpMin);
        Tp = clampRange(Tp, s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        s.rhoHp[c] = solidMass*heatFactor*Tp;
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = GPU_OPERATOR_R(0.0);
    }
#endif
    s.epsS[c] = eps;
    s.Usx[c] = usx;
    s.Usy[c] = usy;
    s.Usz[c] = usz;
    s.theta[c] = theta;
    s.dMeanCell[c] = dMean;
}

__global__ void initialiseParticleMaterialEnthalpyKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_REAL eps = clampMin(s.epsS[c], GPU_OPERATOR_R(0.0));
#if !GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL cap = eps*s.particleThermalRho*s.particleCp;
    if (eps <= s.epsSMin || cap <= s.rhoMin)
#else
    if (eps <= s.epsSMin)
#endif
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = GPU_OPERATOR_R(0.0);
        return;
    }

    s.Tp[c] = clampRange(s.Tp[c], s.TpMin, s.TpMax);
#if GPU_OPERATOR_THERMAL
    s.rhoHp[c] = eps*s.rhoSolid*particleSpecificEnthalpyDevice(s.Tp[c]);
#else
    s.rhoHp[c] = cap*s.Tp[c];
#endif
}
