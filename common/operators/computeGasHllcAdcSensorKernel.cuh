#pragma once
#include "GpuCellNeighbour.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasHllcAdcSensorKernel(GasState* sp)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_REAL centrePressure = clampMin(s.p[c], OfSmall);
    GPU_OPERATOR_REAL omega = GPU_OPERATOR_R(1.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        GPU_OPERATOR_REAL otherPressure = centrePressure;
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = oppositeCellAcrossFace(c, own, nei);
            if (other < 0 || other >= s.nCells)
            {
                continue;
            }
            otherPressure = clampMin(s.p[other], OfSmall);
        }
        else if (s.riemannBoundaryKind[f] == 0)
        {
            const GasPrimDevice centre = makeGasPrimDevice
            (
                s.rho[c],
                s.Ux[c],
                s.Uy[c],
                s.Uz[c],
                centrePressure,
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
            otherPressure = clampMin
            (
                riemannBoundaryState(s, f, centre).p,
                OfSmall
            );
        }

        const GPU_OPERATOR_REAL ratio = clampRange
        (
            fmin
            (
                otherPressure/centrePressure,
                centrePressure/otherPressure
            ),
            GPU_OPERATOR_R(0.0),
            GPU_OPERATOR_R(1.0)
        );
        const GPU_OPERATOR_REAL faceSensor = ratio*ratio*ratio;
        omega = fmin(omega, faceSensor);
    }
    s.gasHllcAdcSensor[c] = finiteDevice(omega)
      ? clampRange(omega, GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0)) : GPU_OPERATOR_R(0.0);
}
