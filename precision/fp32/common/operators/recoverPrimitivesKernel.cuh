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
    const double solidMass = double(eps)*s.rhoSolid;
    if (eps <= s.epsSMin || solidMass <= s.epsSMin*s.rhoSolid)
    {
        clearSolidCell(s, c);
        return;
    }

    const double usx = finiteOr(s.rhoUsx[c]/solidMass, GPU_OPERATOR_R(0.0));
    const double usy = finiteOr(s.rhoUsy[c]/solidMass, GPU_OPERATOR_R(0.0));
    const double usz = finiteOr(s.rhoUsz[c]/solidMass, GPU_OPERATOR_R(0.0));
    const double solidKinetic = GPU_OPERATOR_R(0.5)*sqr3(usx, usy, usz);
    double theta =
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
        const double specificEnthalpy =
            finiteOr(s.rhoHp[c]/solidMass, -GPU_OPERATOR_R(1.0));
        const double rawTp = particleTemperatureFromSpecificEnthalpyDevice
        (
            specificEnthalpy
        );
        const double Tp = clampRange(finiteOr(rawTp, s.TpMin), s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        // Temperature is a rounded cache. Retain authoritative enthalpy unless
        // the existing invalid-state or temperature-bound repair changed it.
        if (!finiteDevice(rawTp) || rawTp != Tp)
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
        const double rawTp = s.rhoHp[c]/(solidMass*heatFactor);
        const double Tp = clampRange(finiteOr(rawTp, s.TpMin), s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        if (!finiteDevice(rawTp) || rawTp != Tp)
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
    const double cap = double(eps)*s.particleThermalRho*s.particleCp;
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
    s.rhoHp[c] = double(eps)*s.rhoSolid*particleSpecificEnthalpyDevice(s.Tp[c]);
#else
    s.rhoHp[c] = cap*s.Tp[c];
#endif
}
