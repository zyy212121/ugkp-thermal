#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void applyGasVolumeFractionSourceKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_REAL eps = solidEpsFromMomentDevice(s, c);
    const GPU_OPERATOR_REAL epsG = GPU_OPERATOR_R(1.0) - eps;
    const GPU_OPERATOR_REAL epsGsafe = clampMin(epsG, OfSmall);
    const GPU_OPERATOR_REAL epsGOld = finiteOr(s.epsGPrev[c], epsG);

    GPU_OPERATOR_REAL gradEx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gradEy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gradEz = GPU_OPERATOR_R(0.0);

    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];

    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const int f = s.cellFaceId[p];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const int own = s.faceOwner[f];
        const int neiFace = s.faceNeighbour[f];

        const GPU_OPERATOR_REAL epsOwn =
            (own >= 0 && own < s.nCells)
          ? GPU_OPERATOR_R(1.0) - solidEpsFromMomentDevice(s, own)
          : epsG;

        const GPU_OPERATOR_REAL epsNei =
            (neiFace >= 0 && neiFace < s.nCells)
          ? GPU_OPERATOR_R(1.0) - solidEpsFromMomentDevice(s, neiFace)
          : epsG;

        const GPU_OPERATOR_REAL lambda = finiteOr(s.faceWeight[f], GPU_OPERATOR_R(0.5));
        const GPU_OPERATOR_REAL epsFace = lambda*epsOwn + (GPU_OPERATOR_R(1.0) - lambda)*epsNei;
        const GPU_OPERATOR_REAL sign = (own == c) ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);

        gradEx += epsFace*sign*s.Sfx[f];
        gradEy += epsFace*sign*s.Sfy[f];
        gradEz += epsFace*sign*s.Sfz[f];
    }

    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    gradEx *= invV;
    gradEy *= invV;
    gradEz *= invV;

    const GPU_OPERATOR_REAL ugx0 = finiteOr(s.Ux[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ugy0 = finiteOr(s.Uy[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ugz0 = finiteOr(s.Uz[c], GPU_OPERATOR_R(0.0));

    const GPU_OPERATOR_REAL cepsG =
        -((epsG - epsGOld)/(dt + OfSmall)
          + ugx0*gradEx + ugy0*gradEy + ugz0*gradEz)/epsGsafe;

    const GPU_OPERATOR_REAL mgOld =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL pressureOld =
        clampMin(finiteOr(s.p[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL enerGOld = finiteOr(s.rhoE[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL massScale = GPU_OPERATOR_R(1.0) + dt*cepsG;
    const GPU_OPERATOR_REAL mgCandidate = mgOld*massScale;
    const GPU_OPERATOR_REAL momGXCandidate = s.rhoUx[c]*massScale;
    const GPU_OPERATOR_REAL momGYCandidate = s.rhoUy[c]*massScale;
    const GPU_OPERATOR_REAL momGZCandidate = s.rhoUz[c]*massScale;
    const GPU_OPERATOR_REAL enerGCandidate =
        enerGOld*massScale + dt*cepsG*pressureOld;
    /* legacy: exponential integration of the frozen-coefficient source.
       Retained for reference only; the active update is the phase-volume-compatible linear form.
    const GPU_OPERATOR_REAL sourceExponent = dt*cepsG;
    const GPU_OPERATOR_REAL massScale = exp(sourceExponent);
    const GPU_OPERATOR_REAL kineticOld =
        GPU_OPERATOR_R(0.5)
       *sqr3(s.rhoUx[c], s.rhoUy[c], s.rhoUz[c])
       /mgOld;
    const GPU_OPERATOR_REAL internalEnergyOld = enerGOld - kineticOld;
    const GPU_OPERATOR_REAL mgCandidate = mgOld*massScale;
    const GPU_OPERATOR_REAL momGXCandidate = s.rhoUx[c]*massScale;
    const GPU_OPERATOR_REAL momGYCandidate = s.rhoUy[c]*massScale;
    const GPU_OPERATOR_REAL momGZCandidate = s.rhoUz[c]*massScale;
    const GPU_OPERATOR_REAL enerGCandidate =
        massScale
       *(
            enerGOld
          + internalEnergyOld
           *expm1((s.gammaGas - GPU_OPERATOR_R(1.0))*sourceExponent)
        );
    */
    const GPU_OPERATOR_REAL kineticCandidate =
        GPU_OPERATOR_R(0.5)
       *sqr3(momGXCandidate, momGYCandidate, momGZCandidate)
       /clampMin(mgCandidate, s.rhoMin);
    const GPU_OPERATOR_REAL internalEnergyFloorCandidate =
        mgCandidate*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - GPU_OPERATOR_R(1.0), OfSmall);
    const GPU_OPERATOR_REAL rhoKCandidate = s.sstConfigured != 0
      ? s.rhoK[c]*massScale : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL rhoOmegaCandidate = s.sstConfigured != 0
      ? s.rhoOmega[c]*massScale : GPU_OPERATOR_R(0.0);
    if
    (
        !finiteDevice(rhoKCandidate)
     || !finiteDevice(rhoOmegaCandidate)
     || rhoKCandidate < GPU_OPERATOR_R(0.0)
     || rhoOmegaCandidate < GPU_OPERATOR_R(0.0)
     || !finiteDevice(cepsG)
     || !finiteDevice(massScale)
     || !finiteDevice(mgCandidate)
     || !finiteDevice(momGXCandidate)
     || !finiteDevice(momGYCandidate)
     || !finiteDevice(momGZCandidate)
     || !finiteDevice(pressureOld)
     || !finiteDevice(enerGCandidate)
     || !finiteDevice(kineticCandidate)
     || !finiteDevice(internalEnergyFloorCandidate)
     || mgCandidate < s.rhoMin
     || enerGCandidate < kineticCandidate + internalEnergyFloorCandidate
    )
    {
        asm("trap;");
        return;
    }

    if (s.sstConfigured != 0)
    {
        // rho*k and rho*omega carry the same phase-volume weight as rho.
        // This is a physical source, distinct from numerical density repair.
        s.rhoK[c] = rhoKCandidate;
        s.rhoOmega[c] = rhoOmegaCandidate;
    }
    s.rho[c] = mgCandidate;
    s.rhoUx[c] = momGXCandidate;
    s.rhoUy[c] = momGYCandidate;
    s.rhoUz[c] = momGZCandidate;
    s.rhoE[c] = enerGCandidate;
    s.epsGPrev[c] = epsG;
}
