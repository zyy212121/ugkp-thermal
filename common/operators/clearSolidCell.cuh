#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ void clearSolidCell(DeviceState& s, const int c)
{
    s.epsS[c] = GPU_OPERATOR_R(0.0);
    s.rhoUsx[c] = GPU_OPERATOR_R(0.0);
    s.rhoUsy[c] = GPU_OPERATOR_R(0.0);
    s.rhoUsz[c] = GPU_OPERATOR_R(0.0);
    s.rhoEs[c] = GPU_OPERATOR_R(0.0);
    s.rhoDs[c] = GPU_OPERATOR_R(0.0);
    s.rhoHp[c] = GPU_OPERATOR_R(0.0);
    s.Usx[c] = GPU_OPERATOR_R(0.0);
    s.Usy[c] = GPU_OPERATOR_R(0.0);
    s.Usz[c] = GPU_OPERATOR_R(0.0);
    s.theta[c] = GPU_OPERATOR_R(0.0);
    s.Tp[c] = s.TpMin;
    s.dMeanCell[c] =
        clampMin(finiteOr(s.particleDiameterFallback, GPU_OPERATOR_R(1.0e-12)), GPU_OPERATOR_R(1.0e-12));
}
