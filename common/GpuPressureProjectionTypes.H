#pragma once
// Binary64 physical pressure-projection cache, shared by resident particle owners.
struct PressureProjectionCell
{
    double ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale;
    bool active, resolved;
};
