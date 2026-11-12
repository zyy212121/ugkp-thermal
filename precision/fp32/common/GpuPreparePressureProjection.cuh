#pragma once
// Shared physical update; the including backend supplies its state and scalar adapters.
__device__ PressureProjectionCell preparePressureProjectionCell
(DeviceState& s, const int c, const PressureTime kickDt)
{
    PressureProjectionCell projection{};
    PressureReal scaledDelta[4];

    accumulatePressureFaceDelta(s,c,kickDt,
        scaledDelta[0],scaledDelta[1],scaledDelta[2],scaledDelta[3]);
        s.pressureDeltaMomX[c] = scaledDelta[0];
        s.pressureDeltaMomY[c] = scaledDelta[1];
        s.pressureDeltaMomZ[c] = scaledDelta[2];
        s.pressureDeltaEnergy[c] = scaledDelta[3];



    projection.active = false;
    const PressureReal rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0f), 0.0f);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return projection;
    }
    const PressureReal px0 = finiteOr(s.momRhoUPx[c], 0.0f);
    const PressureReal py0 = finiteOr(s.momRhoUPy[c], 0.0f);
    const PressureReal pz0 = finiteOr(s.momRhoUPz[c], 0.0f);
    const PressureReal e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0f), 0.0f);
    const PressureReal px1 = px0 + scaledDelta[0];
    const PressureReal py1 = py0 + scaledDelta[1];
    const PressureReal pz1 = pz0 + scaledDelta[2];
    const PressureReal e1 = e0 + scaledDelta[3];
    const PressureReal ux0 = px0/rhoP;
    const PressureReal uy0 = py0/rhoP;
    const PressureReal uz0 = pz0/rhoP;
    const PressureReal ux1 = px1/rhoP;
    const PressureReal uy1 = py1/rhoP;
    const PressureReal uz1 = pz1/rhoP;
    const PressureReal theta0 =
        clampMin(pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)/(1.5f*rhoP), 0.0f);
    const PressureReal theta1 =
        clampMin(pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(1.5f*rhoP), 0.0f);
    const bool resolved = theta0 > 10.0f*s.thetaMin;
    const PressureReal thermalScale =
        resolved ? sqrt(clampMin(theta1/theta0, 0.0f)) : 0.0f;
    const PressureReal thetaScale = resolved ? thermalScale*thermalScale : 0.0f;


    projection =
        {ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale, true, resolved};

    publishPressureCellState(s,c,rhoP,px1,py1,pz1,e1,theta1);

    // The atomic path historically publishes the cell update, then reads it
    // back and reconstructs the old state. Preserve its rounding and clamps:
    // (e0 + de) - de is not equivalent to e0 near the resolved-theta threshold.
    if (!s.csrCellLocalPathEnabled)
    {
        const PressureReal ax1 = finiteOr(px1, 0.0f);
        const PressureReal ay1 = finiteOr(py1, 0.0f);
        const PressureReal az1 = finiteOr(pz1, 0.0f);
        const PressureReal ae1 = clampMin(finiteOr(e1, 0.0f), 0.0f);
        const UnsortedPressureKinematics recovered = recoverUnsortedPressureKinematics
        (
            rhoP,ax1,ay1,az1,ae1,scaledDelta[0],scaledDelta[1],
            scaledDelta[2],scaledDelta[3],s.thetaMin
        );
        projection =
        {
            recovered.ux0,recovered.uy0,recovered.uz0,
            recovered.ux1,recovered.uy1,recovered.uz1,
            recovered.theta1,recovered.thermalScale,recovered.thetaScale,
            true,recovered.resolved
        };
    }
    return projection;
}
