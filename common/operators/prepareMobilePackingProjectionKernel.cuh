#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void prepareMobilePackingProjectionKernel
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

    GPU_OPERATOR_REAL epsC = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL uxC = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL uyC = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL uzC = GPU_OPERATOR_R(0.0);
    mobilePackingPrimitive(s, c, epsC, uxC, uyC, uzC);
    const GPU_OPERATOR_REAL epsTotalC = packingTotalFraction(s, c);
    GPU_OPERATOR_REAL volumeFluxSum = GPU_OPERATOR_R(0.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const GPU_OPERATOR_REAL sign = own == c ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        if (nei >= 0 && nei < s.nCells)
        {
            GPU_OPERATOR_REAL epsOwn = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uxOwn = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uyOwn = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uzOwn = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL epsNei = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uxNei = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uyNei = GPU_OPERATOR_R(0.0);
            GPU_OPERATOR_REAL uzNei = GPU_OPERATOR_R(0.0);
            mobilePackingPrimitive
            (
                s, own, epsOwn, uxOwn, uyOwn, uzOwn
            );
            mobilePackingPrimitive
            (
                s, nei, epsNei, uxNei, uyNei, uzNei
            );
            const GPU_OPERATOR_REAL w =
                clampRange(finiteOr(s.faceWeight[f], GPU_OPERATOR_R(0.5)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
            const GPU_OPERATOR_REAL ufx = w*uxOwn + (GPU_OPERATOR_R(1.0) - w)*uxNei;
            const GPU_OPERATOR_REAL ufy = w*uyOwn + (GPU_OPERATOR_R(1.0) - w)*uyNei;
            const GPU_OPERATOR_REAL ufz = w*uzOwn + (GPU_OPERATOR_R(1.0) - w)*uzNei;
            const GPU_OPERATOR_REAL un =
                ufx*s.Sfx[f] + ufy*s.Sfy[f] + ufz*s.Sfz[f];
            const GPU_OPERATOR_REAL epsUpwind = un >= GPU_OPERATOR_R(0.0) ? epsOwn : epsNei;
            volumeFluxSum += sign*epsUpwind*un;
        }
        else if (own == c && s.gasBoundaryKind[f] == 0)
        {
                                                                            
                                                                             
            const GPU_OPERATOR_REAL un =
                uxC*s.Sfx[f] + uyC*s.Sfy[f] + uzC*s.Sfz[f];
            volumeFluxSum += epsC*fmax(un, GPU_OPERATOR_R(0.0));
        }
    }

    const GPU_OPERATOR_REAL epsPred = clampMin
    (
        epsTotalC - dt*volumeFluxSum/clampMin(s.V[c], OfVSmall),
        GPU_OPERATOR_R(0.0)
    );
                                                                            
                                                                            
                                                                            
    const GPU_OPERATOR_REAL signedSlack = epsPred - s.packingFraction;
    s.pressureDeltaEnergy[c] = clampRange
    (
        finiteOr
        (
            s.rhoSolid*s.V[c]*signedSlack/(dt*dt),
            GPU_OPERATOR_R(0.0)
        ),
        -OfGreat,
        OfGreat
    );
    if (signedSlack > GPU_OPERATOR_R(0.0))
    {
        seedMobilePackingActivity(s, c);
    }
}
