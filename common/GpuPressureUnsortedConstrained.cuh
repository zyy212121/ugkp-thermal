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

    PressureReal dpx, dpy, dpz, de;
    accumulatePressureFaceDelta(s,c,kickDt,dpx,dpy,dpz,de);
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

    publishPressureCellState(s,c,rhoP,px1,py1,pz1,e1,theta1);
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
        const UnsortedPressureKinematics projection = recoverUnsortedPressureKinematics
            (rhoP,px1,py1,pz1,e1,dpx,dpy,dpz,de,s.thetaMin);
        updateMobilePressureParticle<false>
            (s,i,projection.ux0,projection.uy0,projection.uz0,
             projection.ux1,projection.uy1,projection.uz1,projection.theta1,
             projection.thermalScale,projection.thetaScale,projection.resolved);
        accumulatePressureParticleMomentsAtomicDevice<FullMoments>(s, c, i);
    }
}
