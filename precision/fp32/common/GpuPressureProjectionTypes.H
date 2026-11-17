#pragma once
#include "GpuPrecisionTypes.H"

struct PressureProjectionCell
{
    GpuReal ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale;
    bool active, resolved;
};
