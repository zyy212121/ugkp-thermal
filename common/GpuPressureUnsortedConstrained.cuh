#pragma once
// Shared unsorted thermal projection; preserves the published-state reconstruction.
__global__ void applyCollisionalPressureProjectionCellAtomicKernel
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

    PressureReal dpx = PressureReal(0.0);
    PressureReal dpy = PressureReal(0.0);
    PressureReal dpz = PressureReal(0.0);
    PressureReal de = PressureReal(0.0);
    const int startFace = s.cellPlaneStart[c];
    const int faceCount = s.cellPlaneCount[c];
    for (int j = 0; j < faceCount; ++j)
    {
        const int f = s.cellFaceId[startFace + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
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
    s.pressureDeltaMomX[c] = dpx;
    s.pressureDeltaMomY[c] = dpy;
    s.pressureDeltaMomZ[c] = dpz;
    s.pressureDeltaEnergy[c] = de;

    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const PressureReal px1 = finiteOr(s.momRhoUPx[c], PressureReal(0.0)) + dpx;
    const PressureReal py1 = finiteOr(s.momRhoUPy[c], PressureReal(0.0)) + dpy;
    const PressureReal pz1 = finiteOr(s.momRhoUPz[c], PressureReal(0.0)) + dpz;
    const PressureReal e1 = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0)) + de;
    const PressureReal theta1 =
        clampMin
        (
            pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(PressureReal(1.5)*rhoP),
            PressureReal(0.0)
        );

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

template<bool FullMoments>
__global__ void applyCollisionalPressureProjectionParticlesAtomicKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] == 0)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        if (s.pStuck[i] != 0)
        {
            s.pux[i] = PressureReal(0.0);
            s.puy[i] = PressureReal(0.0);
            s.puz[i] = PressureReal(0.0);
            if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
            {
                s.puxOld[i] = PressureReal(0.0);
                s.puyOld[i] = PressureReal(0.0);
                s.puzOld[i] = PressureReal(0.0);
            }
            accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
            continue;
        }

        const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], PressureReal(0.0)), PressureReal(0.0));
        if (rhoP <= s.epsSMin*s.rhoSolid)
        {
            accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
            continue;
        }
        const PressureReal dpx = finiteOr(s.pressureDeltaMomX[c], PressureReal(0.0));
        const PressureReal dpy = finiteOr(s.pressureDeltaMomY[c], PressureReal(0.0));
        const PressureReal dpz = finiteOr(s.pressureDeltaMomZ[c], PressureReal(0.0));
        const PressureReal de = finiteOr(s.pressureDeltaEnergy[c], PressureReal(0.0));
        const PressureReal px1 = finiteOr(s.momRhoUPx[c], PressureReal(0.0));
        const PressureReal py1 = finiteOr(s.momRhoUPy[c], PressureReal(0.0));
        const PressureReal pz1 = finiteOr(s.momRhoUPz[c], PressureReal(0.0));
        const PressureReal e1 = clampMin(finiteOr(s.momRhoEP[c], PressureReal(0.0)), PressureReal(0.0));
        const PressureReal px0 = px1 - dpx;
        const PressureReal py0 = py1 - dpy;
        const PressureReal pz0 = pz1 - dpz;
        const PressureReal e0 = e1 - de;
        const PressureReal ux0 = px0/rhoP;
        const PressureReal uy0 = py0/rhoP;
        const PressureReal uz0 = pz0/rhoP;
        const PressureReal ux1 = px1/rhoP;
        const PressureReal uy1 = py1/rhoP;
        const PressureReal uz1 = pz1/rhoP;
        const PressureReal theta0 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)
               /(PressureReal(1.5)*rhoP),
                PressureReal(0.0)
            );
        const PressureReal theta1 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)
               /(PressureReal(1.5)*rhoP),
                PressureReal(0.0)
            );
        const bool resolved = theta0 > PressureReal(10.0)*s.thetaMin;
        const PressureReal thermalScale =
            resolved ? sqrt(clampMin(theta1/theta0, PressureReal(0.0))) : PressureReal(0.0);
        const PressureReal thetaScale = resolved ? thermalScale*thermalScale : PressureReal(0.0);
        updateMobilePressureParticle<false>
            (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
        accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
    }
}
