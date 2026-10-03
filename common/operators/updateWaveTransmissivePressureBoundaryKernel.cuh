#pragma once
__global__ void updateWaveTransmissivePressureBoundaryKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces || s.gasBoundaryPWave[f] == 0)
    {
        return;
    }

    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], GPU_OPERATOR_TINY(1.0e-300));
    GPU_OPERATOR_REAL boundaryUx = s.gasBoundaryUx[f];
    GPU_OPERATOR_REAL boundaryUy = s.gasBoundaryUy[f];
    GPU_OPERATOR_REAL boundaryUz = s.gasBoundaryUz[f];
    if (s.gasBoundaryUFix[f] == 2)
    {
        const GPU_OPERATOR_REAL ownerOutwardVelocity =
            s.Ux[own]*s.Sfx[f]
          + s.Uy[own]*s.Sfy[f]
          + s.Uz[own]*s.Sfz[f];
        if (ownerOutwardVelocity >= GPU_OPERATOR_R(0.0))
        {
            boundaryUx = s.Ux[own];
            boundaryUy = s.Uy[own];
            boundaryUz = s.Uz[own];
        }
    }
    const GPU_OPERATOR_REAL phip =
        boundaryUx*s.Sfx[f]
      + boundaryUy*s.Sfy[f]
      + boundaryUz*s.Sfz[f];
    const GPU_OPERATOR_REAL psi =
        s.gasBoundaryRho[f]
       /clampMin(s.gasBoundaryP[f], OfSmall);
    GPU_OPERATOR_REAL w =
        phip/area
      + sqrt(clampMin(s.gasBoundaryPWaveGamma[f]/clampMin(psi, OfSmall), GPU_OPERATOR_R(0.0)));
    w = w > GPU_OPERATOR_R(0.0) ? w : GPU_OPERATOR_R(0.0);

    const GPU_OPERATOR_REAL alpha = w*dt*s.deltaCoeffs[f];
    const GPU_OPERATOR_REAL lInf = s.gasBoundaryPWaveLInf[f];
    const bool hasRelaxation = lInf > GPU_OPERATOR_TINY(1.0e-300);
    const GPU_OPERATOR_REAL k = hasRelaxation ? w*dt/lInf : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL oldBoundaryP = s.gasBoundaryP[f];
    const GPU_OPERATOR_REAL refValue = hasRelaxation
      ? (oldBoundaryP + k*s.gasBoundaryPWaveFieldInf[f])/(GPU_OPERATOR_R(1.0) + k)
      : oldBoundaryP;
    const GPU_OPERATOR_REAL valueFraction = hasRelaxation
      ? (GPU_OPERATOR_R(1.0) + k)/(GPU_OPERATOR_R(1.0) + alpha + k)
      : GPU_OPERATOR_R(1.0)/(GPU_OPERATOR_R(1.0) + alpha);
    const GPU_OPERATOR_REAL boundaryP =
        valueFraction*refValue
      + (GPU_OPERATOR_R(1.0) - valueFraction)*s.p[own];

    const GPU_OPERATOR_REAL updatedPressure =
        clampMin(finiteOr(boundaryP, s.p[own]), OfVSmall);
    s.gasBoundaryP[f] = updatedPressure;
    s.riemannBoundaryP[f] = updatedPressure;
}
