#pragma once
__global__ void validateDevelopmentProbeCellsKernel
(
    DeviceState* sp,
    DevelopmentProbeDeviceSummary* summary,
    const int validateSolid
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        unsigned long long mask = 0;

        if
        (
            !finiteDevice(s.rho[c])
         || !finiteDevice(s.rhoUx[c])
         || !finiteDevice(s.rhoUy[c])
         || !finiteDevice(s.rhoUz[c])
         || !finiteDevice(s.rhoE[c])
        )
        {
            mask |= ProbeBadGasConserved;
        }

        if
        (
            !finiteDevice(s.Ux[c])
         || !finiteDevice(s.Uy[c])
         || !finiteDevice(s.Uz[c])
         || !finiteDevice(s.p[c])
         || !finiteDevice(s.Tgas[c])
        )
        {
            mask |= ProbeBadGasPrimitive;
        }

        if
        (
            !(s.rho[c] > GPU_OPERATOR_R(0.0))
         || !(s.p[c] > GPU_OPERATOR_R(0.0))
         || !(s.Tgas[c] > GPU_OPERATOR_R(0.0))
        )
        {
            mask |= ProbeBadCellRange;
        }

        if (validateSolid != 0)
        {
            if
            (
                !finiteDevice(s.epsS[c])
             || !finiteDevice(s.rhoUsx[c])
             || !finiteDevice(s.rhoUsy[c])
             || !finiteDevice(s.rhoUsz[c])
             || !finiteDevice(s.rhoEs[c])
             || !finiteDevice(s.rhoDs[c])
             || !finiteDevice(s.rhoHp[c])
            )
            {
                mask |= ProbeBadSolidConserved;
            }

            if
            (
                !finiteDevice(s.Usx[c])
             || !finiteDevice(s.Usy[c])
             || !finiteDevice(s.Usz[c])
             || !finiteDevice(s.theta[c])
             || !finiteDevice(s.Tp[c])
             || !finiteDevice(s.dMeanCell[c])
            )
            {
                mask |= ProbeBadSolidPrimitive;
            }

            if
            (
                !finiteDevice(s.momRhoP[c])
             || !finiteDevice(s.momRhoUPx[c])
             || !finiteDevice(s.momRhoUPy[c])
             || !finiteDevice(s.momRhoUPz[c])
              || !finiteDevice(s.momRhoEP[c])
              || !finiteDevice(s.momRhoPD[c])
               || !finiteDevice(s.momRhoHpP[c])
            )
            {
                mask |= ProbeBadSolidMoments;
            }

            if
            (
                s.epsS[c] < GPU_OPERATOR_R(0.0)
             || s.epsS[c] > GPU_OPERATOR_R(1.0)
             || s.theta[c] < GPU_OPERATOR_R(0.0)
             || (s.solveParticleTemperature != 0 && !(s.Tp[c] > GPU_OPERATOR_R(0.0)))
             || (s.epsS[c] > s.epsSMin && !(s.dMeanCell[c] > GPU_OPERATOR_R(0.0)))
            )
            {
                mask |= ProbeBadCellRange;
            }
        }

        recordDevelopmentProbeCellFailure(summary, c, mask);
    }
}
