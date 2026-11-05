#pragma once
// Shared physical update; the including backend supplies its state and scalar adapters.
__device__ PressureProjectionCell preparePressureProjectionCell
(DeviceState& s, const int c, const double kickDt)
{
    PressureProjectionCell projection{};
    double scaledDelta[4];

    accumulatePressureFaceDelta(s,c,kickDt,
        scaledDelta[0],scaledDelta[1],scaledDelta[2],scaledDelta[3]);
        s.pressureDeltaMomX[c] = scaledDelta[0];
        s.pressureDeltaMomY[c] = scaledDelta[1];
        s.pressureDeltaMomZ[c] = scaledDelta[2];
        s.pressureDeltaEnergy[c] = scaledDelta[3];



    projection.active = false;
    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return projection;
    }
    const double px0 = finiteOr(s.momRhoUPx[c], 0.0);
    const double py0 = finiteOr(s.momRhoUPy[c], 0.0);
    const double pz0 = finiteOr(s.momRhoUPz[c], 0.0);
    const double e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double px1 = px0 + scaledDelta[0];
    const double py1 = py0 + scaledDelta[1];
    const double pz1 = pz0 + scaledDelta[2];
    const double e1 = e0 + scaledDelta[3];
    const double ux0 = px0/rhoP;
    const double uy0 = py0/rhoP;
    const double uz0 = pz0/rhoP;
    const double ux1 = px1/rhoP;
    const double uy1 = py1/rhoP;
    const double uz1 = pz1/rhoP;
    const double theta0 =
        clampMin(pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)/(1.5*rhoP), 0.0);
    const double theta1 =
        clampMin(pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(1.5*rhoP), 0.0);
    const bool resolved = theta0 > 10.0*s.thetaMin;
    const double thermalScale =
        resolved ? sqrt(clampMin(theta1/theta0, 0.0)) : 0.0;
    const double thetaScale = resolved ? thermalScale*thermalScale : 0.0;


    projection =
        {ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale, true, resolved};

    publishPressureCellState(s,c,rhoP,px1,py1,pz1,e1,theta1);

    // The atomic path historically publishes the cell update, then reads it
    // back and reconstructs the old state. Preserve its rounding and clamps:
    // (e0 + de) - de is not equivalent to e0 near the resolved-theta threshold.
    if (!s.csrCellLocalPathEnabled)
    {
        const double ax1 = finiteOr(px1, 0.0);
        const double ay1 = finiteOr(py1, 0.0);
        const double az1 = finiteOr(pz1, 0.0);
        const double ae1 = clampMin(finiteOr(e1, 0.0), 0.0);
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
