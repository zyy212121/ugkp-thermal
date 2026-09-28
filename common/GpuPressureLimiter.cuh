#pragma once
// Identical pressure positivity and face-limiting equations for every application.
__device__ PressureReal pressureKickInternalEnergy
(
    const PressureReal rhoP,
    const PressureReal px,
    const PressureReal py,
    const PressureReal pz,
    const PressureReal energy
)
{
    return energy - PressureReal(0.5)*sqr3(px, py, pz)/clampMin(rhoP, OfVSmall);
}

__global__ void computeCollisionalPressureKickScaleKernel
(
    DeviceState* sp,
    const PressureTime kickDt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        s.pressureKickScale[c] = PressureReal(0.0);
        return;
    }

    const PressureReal dpx = finiteOr(s.pressureDeltaMomX[c], PressureReal(0.0));
    const PressureReal dpy = finiteOr(s.pressureDeltaMomY[c], PressureReal(0.0));
    const PressureReal dpz = finiteOr(s.pressureDeltaMomZ[c], PressureReal(0.0));
    const PressureReal de = finiteOr(s.pressureDeltaEnergy[c], PressureReal(0.0));
    PressureReal lambda = PressureReal(1.0);

    const PressureReal dUmag = sqrt(sqr3(dpx, dpy, dpz))/rhoP;
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

    const PressureReal trialInternal = pressureKickInternalEnergy
    (
        rhoP,
        px0 + lambda*dpx,
        py0 + lambda*dpy,
        pz0 + lambda*dpz,
        e0 + lambda*de
    );
    if (!finiteDevice(trialInternal) || trialInternal < minInternal)
    {
        PressureReal lo = PressureReal(0.0);
        PressureReal hi = lambda;
        for (int iter = 0; iter < 20; ++iter)
        {
            const PressureReal mid = PressureReal(0.5)*(lo + hi);
            const PressureReal internal = pressureKickInternalEnergy
            (
                rhoP,
                px0 + mid*dpx,
                py0 + mid*dpy,
                pz0 + mid*dpz,
                e0 + mid*de
            );
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

__global__ void scaleCollisionalPressureFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }
    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    PressureReal scale = own >= 0 && own < s.nCells ? s.pressureKickScale[own] : PressureReal(0.0);
    if (nei >= 0 && nei < s.nCells)
    {
        scale = fmin(scale, s.pressureKickScale[nei]);
    }
    scale = clampRange(finiteOr(scale, PressureReal(0.0)), PressureReal(0.0), PressureReal(1.0));
    s.solidPressurePhiMomX[f] *= scale;
    s.solidPressurePhiMomY[f] *= scale;
    s.solidPressurePhiMomZ[f] *= scale;
    s.solidPressurePhiEnergy[f] *= scale;
}
