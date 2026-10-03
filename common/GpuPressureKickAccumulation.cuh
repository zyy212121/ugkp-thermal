#pragma once
#include "GpuPressureUnsortedAlgebra.cuh"
// Shared cell face accumulation and positivity limiter; scratch is a field adapter.
template<class Scratch>
__device__ __forceinline__ void runPressureKickAccumulation
(DeviceState* sp,const PressureTime kickDt,const int computeScale,const int initialiseMoments)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    if (initialiseMoments != 0) Scratch::initialise(s,c);

    PressureReal deltaPx, deltaPy, deltaPz, deltaE;
    accumulatePressureFaceDelta(s,c,kickDt,deltaPx,deltaPy,deltaPz,deltaE);
    s.pressureDeltaMomX[c] = deltaPx;
    s.pressureDeltaMomY[c] = deltaPy;
    s.pressureDeltaMomZ[c] = deltaPz;
    s.pressureDeltaEnergy[c] = deltaE;

    if (computeScale == 0)
    {
        return;
    }

    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        s.pressureKickScale[c] = PressureReal(0.0);
        return;
    }

    PressureReal lambda = PressureReal(1.0);
    const PressureReal dUmag = sqrt(sqr3(deltaPx, deltaPy, deltaPz))/rhoP;
    const PressureReal maxDU =
        s.pressureKickFraction*clampMin(s.cellLength[c], PressureReal(1.0e-12))
       /clampMin(kickDt, OfSmall);
    if (dUmag > maxDU)
    {
        lambda = clampRange(maxDU/dUmag, PressureReal(0.0), PressureReal(1.0));
    }

    const PressureReal px0 = finiteOr(s.momRhoUPx[c], PressureReal(0.0));
    const PressureReal py0 = finiteOr(s.momRhoUPy[c], PressureReal(0.0));
    const PressureReal pz0 = finiteOr(s.momRhoUPz[c], PressureReal(0.0));
    const PressureReal e0 = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0));
    const PressureReal minInternal = PressureReal(1.5)*rhoP*clampMin(s.thetaMin, PressureReal(0.0));
    const PressureReal trialInternal =
        e0 + lambda*deltaE
      - PressureReal(0.5)*sqr3
        (
            px0 + lambda*deltaPx,
            py0 + lambda*deltaPy,
            pz0 + lambda*deltaPz
        )/rhoP;
    if (!finiteDevice(trialInternal) || trialInternal < minInternal)
    {
        PressureReal lo = PressureReal(0.0);
        PressureReal hi = lambda;
        for (int iter = 0; iter < 20; ++iter)
        {
            const PressureReal mid = PressureReal(0.5)*(lo + hi);
            const PressureReal internal =
                e0 + mid*deltaE
              - PressureReal(0.5)*sqr3
                (
                    px0 + mid*deltaPx,
                    py0 + mid*deltaPy,
                    pz0 + mid*deltaPz
                )/rhoP;
            if (finiteDevice(internal) && internal >= minInternal)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        lambda = lo;
    }
    s.pressureKickScale[c] = clampRange(finiteOr(lambda, PressureReal(0.0)), PressureReal(0.0), PressureReal(1.0));
}
