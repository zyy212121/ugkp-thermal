#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void computeGasEddyViscosityKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if (s.turbulenceModel == 0)
    {
        s.nut[c] = GPU_OPERATOR_R(0.0);
        return;
    }
    if (s.turbulenceModel == 3)
    {
        if (s.sstConfigured == 0)
        {
            asm("trap;");
            return;
        }
        GPU_OPERATOR_REAL divU = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL s2 = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL gByNu = GPU_OPERATOR_R(0.0);
        sstVelocityInvariants(s, c, divU, s2, gByNu);
        (void)divU;
        (void)gByNu;
        const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
        const GPU_OPERATOR_REAL nu = s.gasMu/rhoSafe;
        const GPU_OPERATOR_REAL gradDot =
            s.gradKX[c]*s.gradOmegaX[c]
          + s.gradKY[c]*s.gradOmegaY[c]
          + s.gradKZ[c]*s.gradOmegaZ[c];
        const GPU_OPERATOR_REAL cd = ugkwp::sstCrossDiffusion
        (
            s.omega[c],
            gradDot,
            s.sstCoefficients
        );
        const GPU_OPERATOR_REAL f1 = ugkwp::sstF1
        (
            s.k[c],
            s.omega[c],
            nu,
            s.sstWallDistance[c],
            cd,
            s.sstCoefficients
        );
        const GPU_OPERATOR_REAL f2 = ugkwp::sstF2
        (
            s.k[c],
            s.omega[c],
            nu,
            s.sstWallDistance[c],
            s.sstCoefficients
        );
        const GPU_OPERATOR_REAL nuT = ugkwp::sstNut
        (
            s.k[c],
            s.omega[c],
            s2,
            f2,
            s.sstCoefficients
        );
        s.sstF1[c] = clampRange(finiteOr(f1, GPU_OPERATOR_R(1.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        s.sstF2[c] = clampRange(finiteOr(f2, GPU_OPERATOR_R(1.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        s.nut[c] = finiteDevice(nuT) && nuT > GPU_OPERATOR_R(0.0) ? nuT : GPU_OPERATOR_R(0.0);
        return;
    }
    GPU_OPERATOR_REAL g[3][3] =
    {
        {s.gradUxX[c], s.gradUxY[c], s.gradUxZ[c]},
        {s.gradUyX[c], s.gradUyY[c], s.gradUyZ[c]},
        {s.gradUzX[c], s.gradUzY[c], s.gradUzZ[c]}
    };
    const GPU_OPERATOR_REAL tr = g[0][0] + g[1][1] + g[2][2];
    GPU_OPERATOR_REAL ss = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL symmetricGradientSquared = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL strain[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const GPU_OPERATOR_REAL symmetricGradient = GPU_OPERATOR_R(0.5)*(g[i][j] + g[j][i]);
            symmetricGradientSquared += symmetricGradient*symmetricGradient;
            strain[i][j] = symmetricGradient
              - (i == j ? tr/GPU_OPERATOR_R(3.0) : GPU_OPERATOR_R(0.0));
            ss += strain[i][j]*strain[i][j];
        }
    }
    const GPU_OPERATOR_REAL delta = s.lesDeltaCoeff*clampMin(s.cellLength[c], OfSmall);
    GPU_OPERATOR_REAL nuT = GPU_OPERATOR_R(0.0);
    if (s.turbulenceModel == 2)
    {
        nuT = ugkwp::smagorinskyNut(s.smagorinskyCs, delta, ss);
    }
    else
    {
        GPU_OPERATOR_REAL g2[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                g2[i][j] = GPU_OPERATOR_R(0.0);
                for (int k = 0; k < 3; ++k)
                {
                    g2[i][j] += g[i][k]*g[k][j];
                }
            }
        }
        const GPU_OPERATOR_REAL trG2 = g2[0][0] + g2[1][1] + g2[2][2];
        GPU_OPERATOR_REAL sd2 = GPU_OPERATOR_R(0.0);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const GPU_OPERATOR_REAL sd = GPU_OPERATOR_R(0.5)*(g2[i][j] + g2[j][i])
                  - (i == j ? trG2/GPU_OPERATOR_R(3.0) : GPU_OPERATOR_R(0.0));
                sd2 += sd*sd;
            }
        }
        nuT = ugkwp::waleNut
        (
            s.waleCw,
            delta,
            symmetricGradientSquared,
            sd2,
            OfSmall
        );
                                                                        
                                                                          
    }
    s.nut[c] = finiteDevice(nuT) && nuT > GPU_OPERATOR_R(0.0) ? nuT : GPU_OPERATOR_R(0.0);
}
