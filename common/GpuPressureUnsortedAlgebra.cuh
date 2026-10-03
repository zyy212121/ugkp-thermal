#pragma once

// Defined in GpuPressureLimiter.cuh; the shared recovery uses that same equation.
__device__ PressureReal pressureKickInternalEnergy
(PressureReal rhoP, PressureReal px, PressureReal py, PressureReal pz, PressureReal energy);
// Face order and double-time -> physical-real conversion are retained.
__device__ __forceinline__ void accumulatePressureFaceDelta
(
    const DeviceState& s, const int c, const PressureTime kickDt,
    PressureReal& dpx, PressureReal& dpy, PressureReal& dpz, PressureReal& de
)
{
    dpx = dpy = dpz = de = PressureReal(0.0);
    const int startFace = s.cellPlaneStart[c];
    const int faceCount = s.cellPlaneCount[c];
    for (int j = 0; j < faceCount; ++j)
    {
        const int f = s.cellFaceId[startFace + j];
        if (f < 0 || f >= s.nFaces) continue;
        const PressureReal sign = s.faceOwner[f] == c ? PressureReal(1.0) : -PressureReal(1.0);
        dpx -= sign*s.solidPressurePhiMomX[f];
        dpy -= sign*s.solidPressurePhiMomY[f];
        dpz -= sign*s.solidPressurePhiMomZ[f];
        de -= sign*s.solidPressurePhiEnergy[f];
    }
    const PressureReal factor = kickDt/clampMin(s.V[c], OfVSmall);
    dpx = finiteOr(factor*dpx, PressureReal(0.0));
    dpy = finiteOr(factor*dpy, PressureReal(0.0));
    dpz = finiteOr(factor*dpz, PressureReal(0.0));
    de = finiteOr(factor*de, PressureReal(0.0));
}

__device__ __forceinline__ void publishPressureCellState
(
    DeviceState& s, const int c, const PressureReal rhoP,
    const PressureReal px1, const PressureReal py1, const PressureReal pz1,
    const PressureReal e1, const PressureReal theta1
)
{
    s.momRhoUPx[c] = px1;
    s.momRhoUPy[c] = py1;
    s.momRhoUPz[c] = pz1;
    s.momRhoEP[c] = e1;
    s.rhoUsx[c] = px1;
    s.rhoUsy[c] = py1;
    s.rhoUsz[c] = pz1;
    s.rhoEs[c] = e1;
    s.Usx[c] = px1/rhoP;
    s.Usy[c] = py1/rhoP;
    s.Usz[c] = pz1/rhoP;
    s.theta[c] = theta1;
}

struct UnsortedPressureKinematics
{
    PressureReal ux0, uy0, uz0, ux1, uy1, uz1;
    PressureReal theta1, thermalScale, thetaScale;
    bool resolved;
};

// Inputs are the sanitized published state. Subtract the kick AFTER publication;
// using the original state changes rounding at the resolved-theta threshold.
__device__ __forceinline__ UnsortedPressureKinematics recoverUnsortedPressureKinematics
(
    const PressureReal rhoP,
    const PressureReal px1, const PressureReal py1, const PressureReal pz1,
    const PressureReal e1, const PressureReal dpx, const PressureReal dpy,
    const PressureReal dpz, const PressureReal de, const PressureReal thetaMin
)
{
    const PressureReal px0 = px1 - dpx;
    const PressureReal py0 = py1 - dpy;
    const PressureReal pz0 = pz1 - dpz;
    const PressureReal e0 = e1 - de;
    const PressureReal theta0 = clampMin
    (
        pressureKickInternalEnergy(rhoP,px0,py0,pz0,e0)
        /(PressureReal(1.5)*rhoP), PressureReal(0.0)
    );
    const PressureReal theta1 = clampMin
    (
        pressureKickInternalEnergy(rhoP,px1,py1,pz1,e1)
        /(PressureReal(1.5)*rhoP), PressureReal(0.0)
    );
    const bool resolved = theta0 > PressureReal(10.0)*thetaMin;
    const PressureReal thermalScale = resolved
      ? sqrt(clampMin(theta1/theta0,PressureReal(0.0))) : PressureReal(0.0);
    const PressureReal thetaScale = resolved ? thermalScale*thermalScale : PressureReal(0.0);
    return {px0/rhoP,py0/rhoP,pz0/rhoP,px1/rhoP,py1/rhoP,pz1/rhoP,
            theta1,thermalScale,thetaScale,resolved};
}
