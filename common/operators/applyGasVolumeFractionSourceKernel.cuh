#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void applyGasVolumeFractionSourceKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_REAL dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_REAL eps = solidEpsFromMomentDevice(s, c);
    const GPU_OPERATOR_REAL epsG = 1.0 - eps;
    const GPU_OPERATOR_REAL epsGsafe = clampMin(epsG, OfSmall);
    const GPU_OPERATOR_REAL epsGOld = finiteOr(s.epsGPrev[c], epsG);

    GPU_OPERATOR_REAL gradEx = 0.0;
    GPU_OPERATOR_REAL gradEy = 0.0;
    GPU_OPERATOR_REAL gradEz = 0.0;

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
          ? 1.0 - solidEpsFromMomentDevice(s, own)
          : epsG;

        const GPU_OPERATOR_REAL epsNei =
            (neiFace >= 0 && neiFace < s.nCells)
          ? 1.0 - solidEpsFromMomentDevice(s, neiFace)
          : epsG;

        const GPU_OPERATOR_REAL lambda = finiteOr(s.faceWeight[f], 0.5);
        const GPU_OPERATOR_REAL epsFace = lambda*epsOwn + (1.0 - lambda)*epsNei;
        const GPU_OPERATOR_REAL sign = (own == c) ? 1.0 : -1.0;

        gradEx += epsFace*sign*s.Sfx[f];
        gradEy += epsFace*sign*s.Sfy[f];
        gradEz += epsFace*sign*s.Sfz[f];
    }

    const GPU_OPERATOR_REAL invV = 1.0/clampMin(s.V[c], OfSmall);
    gradEx *= invV;
    gradEy *= invV;
    gradEz *= invV;

    const GPU_OPERATOR_REAL ugx0 = finiteOr(s.Ux[c], 0.0);
    const GPU_OPERATOR_REAL ugy0 = finiteOr(s.Uy[c], 0.0);
    const GPU_OPERATOR_REAL ugz0 = finiteOr(s.Uz[c], 0.0);

    const GPU_OPERATOR_REAL cepsG =
        -((epsG - epsGOld)/(dt + OfSmall)
          + ugx0*gradEx + ugy0*gradEy + ugz0*gradEz)/epsGsafe;

    const GPU_OPERATOR_REAL mgOld =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL pressureOld =
        clampMin(finiteOr(s.p[c], 0.0), 0.0);
    const GPU_OPERATOR_REAL enerGOld = finiteOr(s.rhoE[c], 0.0);
    const GPU_OPERATOR_REAL massScale = 1.0 + dt*cepsG;
    const GPU_OPERATOR_REAL mgCandidate = mgOld*massScale;
    const GPU_OPERATOR_REAL momGXCandidate = s.rhoUx[c]*massScale;
    const GPU_OPERATOR_REAL momGYCandidate = s.rhoUy[c]*massScale;
    const GPU_OPERATOR_REAL momGZCandidate = s.rhoUz[c]*massScale;
    const GPU_OPERATOR_REAL enerGCandidate =
        enerGOld*massScale + dt*cepsG*pressureOld;
    const GPU_OPERATOR_REAL kineticCandidate =
        0.5
       *sqr3(momGXCandidate, momGYCandidate, momGZCandidate)
       /clampMin(mgCandidate, s.rhoMin);
    const GPU_OPERATOR_REAL internalEnergyFloorCandidate =
        mgCandidate*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - 1.0, OfSmall);
    if
    (
        !finiteDevice(cepsG)
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

    s.rho[c] = mgCandidate;
    s.rhoUx[c] = momGXCandidate;
    s.rhoUy[c] = momGYCandidate;
    s.rhoUz[c] = momGZCandidate;
    s.rhoE[c] = enerGCandidate;
    s.epsGPrev[c] = epsG;
}
