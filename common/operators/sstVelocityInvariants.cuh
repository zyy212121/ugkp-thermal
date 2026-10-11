#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__device__ void sstVelocityInvariants
(
    const GasState& s,
    const int c,
    GPU_OPERATOR_REAL& divU,
    GPU_OPERATOR_REAL& s2,
    GPU_OPERATOR_REAL& gByNu
)
{
    const GPU_OPERATOR_REAL g[3][3] =
    {
        {s.gradUxX[c], s.gradUxY[c], s.gradUxZ[c]},
        {s.gradUyX[c], s.gradUyY[c], s.gradUyZ[c]},
        {s.gradUzX[c], s.gradUzY[c], s.gradUzZ[c]}
    };
    divU = g[0][0] + g[1][1] + g[2][2];
    GPU_OPERATOR_REAL symmSquared = GPU_OPERATOR_R(0.0);
    gByNu = GPU_OPERATOR_R(0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const GPU_OPERATOR_REAL twoSymm = g[i][j] + g[j][i];
            const GPU_OPERATOR_REAL devTwoSymm =
                twoSymm - (i == j ? (GPU_OPERATOR_R(2.0)/GPU_OPERATOR_R(3.0))*divU : GPU_OPERATOR_R(0.0));
            const GPU_OPERATOR_REAL symm = GPU_OPERATOR_R(0.5)*twoSymm;
            symmSquared += symm*symm;
            gByNu += devTwoSymm*g[i][j];
        }
    }
    s2 = GPU_OPERATOR_R(2.0)*symmSquared;
}
