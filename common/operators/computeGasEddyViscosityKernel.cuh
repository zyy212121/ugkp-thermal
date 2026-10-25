#pragma once
#include "Diagnostics.H"
template<class GasState>
__device__ bool gasEddyFinite(const GasState& s, int c, GPU_OPERATOR_REAL value,
    unsigned site, int component=-1, unsigned quantity=ugkp::diagnostics::Quantity::LinearSystem)
{
    if (::isfinite(value)) return true;
    using namespace ugkp::diagnostics;
    recordFatal(s.diagnostics,makeRecord(Code::NonFinite,site,303,0,
        EntityKind::Cell,c,component,value,0,0,quantity));
    return false;
}
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasEddyViscosityKernel(GasState* sp)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    GasState& s = *sp;
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
    const GPU_OPERATOR_REAL gradients[] = {s.gradUxX[c],s.gradUxY[c],s.gradUxZ[c],
        s.gradUyX[c],s.gradUyY[c],s.gradUyZ[c],s.gradUzX[c],s.gradUzY[c],s.gradUzZ[c]};
    for(int i=0;i<9;++i) if(!gasEddyFinite(s,c,gradients[i],30300,i)) return;
    if (s.turbulenceModel == 3)
    {
        if (!s.k || !s.omega || !s.sstWallDistance || !s.sstF2)
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::MissingOperator,30301,303,0,
                EntityKind::Cell,c,-1,0,0,0,Quantity::LinearSystem));
            return;
        }
        const GPU_OPERATOR_REAL inputs[] = {s.rho[c],s.rhoMin,s.gasMu,s.k[c],s.omega[c],
            s.sstWallDistance[c],s.sstCoefficients.betaStar,s.sstCoefficients.a1,s.sstCoefficients.b1};
        for(int i=0;i<9;++i) if(!gasEddyFinite(s,c,inputs[i],30302,i)) return;
        GPU_OPERATOR_REAL divU = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL s2 = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL gByNu = GPU_OPERATOR_R(0.0);
        sstVelocityInvariants(s, c, divU, s2, gByNu);
        if(!gasEddyFinite(s,c,divU,30303,0) || !gasEddyFinite(s,c,s2,30319,1)
            || !gasEddyFinite(s,c,gByNu,30320,2)) return;
        const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
        if(rhoSafe==GPU_OPERATOR_R(0.0))
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,30304,303,0,
                EntityKind::Cell,c,-1,rhoSafe,0,0,Quantity::Density));
            return;
        }
        const GPU_OPERATOR_REAL nu = s.gasMu/rhoSafe;
        if(!gasEddyFinite(s,c,nu,30305)) return;
        // Inspect the actual F2 operands before its finite min/max caps can erase a fault.
        const auto omegaSafe=ugkwp::sstPositive(s.omega[c],GPU_OPERATOR_R(1e-300));
        const auto ySafe=ugkwp::sstPositive(s.sstWallDistance[c],GPU_OPERATOR_R(1e-300));
        const GPU_OPERATOR_REAL denominators[] = {s.sstCoefficients.betaStar*omegaSafe*ySafe,ySafe*ySafe*omegaSafe};
        for(int i=0;i<2;++i)
        {
            if(!gasEddyFinite(s,c,denominators[i],30306,i)) return;
            if(denominators[i]==GPU_OPERATOR_R(0.0))
            {
                using namespace ugkp::diagnostics;
                recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,30307,303,0,
                    EntityKind::Cell,c,i,denominators[i],0,0,Quantity::Denominator));
                return;
            }
        }
        if(!gasEddyFinite(s,c,GPU_OPERATOR_R(2.0)*sqrt(ugkwp::sstPositive(s.k[c],GPU_OPERATOR_R(0.0)))/denominators[0],30308,0)
            || !gasEddyFinite(s,c,GPU_OPERATOR_R(500.0)*nu/denominators[1],30321,1)) return;
        const GPU_OPERATOR_REAL f2 = ugkwp::sstF2
        (
            s.k[c],
            s.omega[c],
            nu,
            s.sstWallDistance[c],
            s.sstCoefficients
        );
        if(!gasEddyFinite(s,c,f2,30309)) return;
        const auto term1=s.sstCoefficients.a1*omegaSafe;
        const auto term2=s.sstCoefficients.b1*f2*sqrt(ugkwp::sstPositive(s2,GPU_OPERATOR_R(0.0)));
        if(!gasEddyFinite(s,c,term1,30310,0) || !gasEddyFinite(s,c,term2,30322,1)) return;
        const auto denominator=ugkwp::sstMaximum(term1,term2);
        if(denominator==GPU_OPERATOR_R(0.0))
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,30311,303,0,
                EntityKind::Cell,c,-1,denominator,0,0,Quantity::Denominator));
            return;
        }
        const GPU_OPERATOR_REAL nuT = ugkwp::sstNut
        (
            s.k[c],
            s.omega[c],
            s2,
            f2,
            s.sstCoefficients
        );
        if(!gasEddyFinite(s,c,nuT,30312)) return;
        s.sstF2[c] = clampRange(f2, GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        s.nut[c] = nuT > GPU_OPERATOR_R(0.0) ? nuT : GPU_OPERATOR_R(0.0);
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
    if(!gasEddyFinite(s,c,tr,30313,0) || !gasEddyFinite(s,c,ss,30323,1)
        || !gasEddyFinite(s,c,symmetricGradientSquared,30324,2)
        || !gasEddyFinite(s,c,s.lesDeltaCoeff,30314,0)
        || !gasEddyFinite(s,c,s.cellLength[c],30325,1)
        || !gasEddyFinite(s,c,s.turbulenceModel==2?s.smagorinskyCs:s.waleCw,30326,2)) return;
    const GPU_OPERATOR_REAL delta = s.lesDeltaCoeff*clampMin(s.cellLength[c], OfSmall);
    if(!gasEddyFinite(s,c,delta,30315)) return;
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
        if(!gasEddyFinite(s,c,trG2,30316,0) || !gasEddyFinite(s,c,sd2,30327,1)) return;
        const auto numerator=pow(fmax(sd2,GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(1.5));
        const auto denominator=pow(fmax(symmetricGradientSquared,GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(2.5))
            +pow(fmax(sd2,GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(1.25));
        if(!gasEddyFinite(s,c,numerator,30318,0) || !gasEddyFinite(s,c,denominator,30328,1)) return;
        nuT = ugkwp::waleNut
        (
            s.waleCw,
            delta,
            symmetricGradientSquared,
            sd2,
            OfSmall
        );
                                                                        
                                                                          
    }
    if(!gasEddyFinite(s,c,nuT,30317)) return;
    s.nut[c] = nuT > GPU_OPERATOR_R(0.0) ? nuT : GPU_OPERATOR_R(0.0);
}
