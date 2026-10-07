#pragma once
#include "gasTransport/MixtureThermo.H"

// Mixture extension of the common MUSCL path. The existing Green-Gauss and
// configured scalar limiters supply the slopes; the existing acoustic limiter
// acts in primitive rho/U/p variables with a frozen-composition acoustic basis.
// Formation energy is never used as a perfect-gas Roe enthalpy. Composition has
// one shared slope factor and its final component closes the mass simplex.
// A single face-local admissibility factor shrinks all reconstructed increments
// together. This modifies only disposable face states, never cell inventories.
template<class GasState>
__device__ bool reconstructGasMixtureFace
(
    const GasState& s, const int c, const int f,
    GasPrimDevice& output, GPU_OPERATOR_REAL* massFractions
)
{
    constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
    const GasPrimDevice centre=gasCellPrimitive(s,c);
    GPU_OPERATOR_REAL yc[Ns],dy[Ns];
    for(int k=0;k<Ns;++k)
    {yc[k]=s.gasSpecies.rho[k*s.nCells+c]/centre.rho;dy[k]=GPU_OPERATOR_R(0.0);}
    if(s.gasReconstruction!=1)
    {output=centre;for(int k=0;k<Ns;++k)massFractions[k]=yc[k];return true;}
    GPU_OPERATOR_REAL cx,cy,cz;
    periodicMappedCellCentre(s,f,c,cx,cy,cz);
    const GPU_OPERATOR_REAL dx=s.faceCx[f]-cx, ddy=s.faceCy[f]-cy, dz=s.faceCz[f]-cz;
    ugkpcharacteristic::Increment increment
    {
        s.gasGradientLimiterRho[c]*(s.gradRhoX[c]*dx+s.gradRhoY[c]*ddy+s.gradRhoZ[c]*dz),
        s.gasGradientLimiterUx[c]*(s.gradUxX[c]*dx+s.gradUxY[c]*ddy+s.gradUxZ[c]*dz),
        s.gasGradientLimiterUy[c]*(s.gradUyX[c]*dx+s.gradUyY[c]*ddy+s.gradUyZ[c]*dz),
        s.gasGradientLimiterUz[c]*(s.gradUzX[c]*dx+s.gradUzY[c]*ddy+s.gradUzZ[c]*dz),
        s.gasGradientLimiterP[c]*(s.gradPx[c]*dx+s.gradPy[c]*ddy+s.gradPz[c]*dz)
    };
    const int nei=coupledFaceNeighbour(s,f);
    if(nei>=0)
    {
        const int other=c==s.faceOwner[f]?nei:s.faceOwner[f];
        const auto adjacent=gasCellPrimitive(s,other);
        const GPU_OPERATOR_REAL rl=sqrt(centre.rho),rr=sqrt(adjacent.rho);
        const GPU_OPERATOR_REAL sound=(rl*s.gasSpecies.soundSpeed[c]+rr*s.gasSpecies.soundSpeed[other])/(rl+rr);
        const GPU_OPERATOR_REAL weight=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
        increment=ugkpcharacteristic::limitOneSide(increment,
            ugkpcharacteristic::Increment{adjacent.rho-centre.rho,adjacent.ux-centre.ux,
                adjacent.uy-centre.uy,adjacent.uz-centre.uz,adjacent.p-centre.p},
            s.Sfx[f]/s.magSf[f],s.Sfy[f]/s.magSf[f],s.Sfz[f]/s.magSf[f],
            rl*rr,sound,c==s.faceOwner[f]?GPU_OPERATOR_R(1.0)-weight:weight);
    }
    for(int k=0;k<Ns;++k)
    {
        const int i=k*s.nCells+c;
        dy[k]=s.gasSpecies.limiter[c]
            *(s.gasSpecies.gradX[i]*dx+s.gasSpecies.gradY[i]*ddy+s.gasSpecies.gradZ[i]*dz);
    }
    GPU_OPERATOR_REAL alpha=GPU_OPERATOR_R(1.0);
    for(int attempt=0;attempt<=64;++attempt)
    {
        GasPrimDevice candidate{centre.rho+alpha*increment.rho,centre.ux+alpha*increment.ux,
            centre.uy+alpha*increment.uy,centre.uz+alpha*increment.uz,centre.p+alpha*increment.p,centre.T};
        GPU_OPERATOR_REAL y[Ns],sum=GPU_OPERATOR_R(0.0);
        bool valid=ugkwp::gasFinite(candidate.rho)&&candidate.rho>=s.rhoMin
            && ugkwp::gasFinite(candidate.p)&&candidate.p>GPU_OPERATOR_R(0.0)
            && ugkwp::gasFinite(candidate.ux)&&ugkwp::gasFinite(candidate.uy)&&ugkwp::gasFinite(candidate.uz);
        for(int k=0;k<Ns;++k)
        {
            y[k]=k+1==Ns?GPU_OPERATOR_R(1.0)-sum:yc[k]+alpha*dy[k];
            sum+=y[k];valid=valid&&ugkwp::gasFinite(y[k])&&y[k]>=GPU_OPERATOR_R(0.0)&&y[k]<=GPU_OPERATOR_R(1.0);
        }
        if(valid)
        {
            const auto R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
            candidate.T=candidate.p/(candidate.rho*R);
            valid=ugkwp::gasFinite(candidate.T)&&candidate.T>=s.TgasMin;
            for(int k=0;k<Ns;++k)
                valid=valid&&candidate.T>=s.gasSpecies.thermo.species[k].minTemperature
                    &&candidate.T<=s.gasSpecies.thermo.species[k].maxTemperature;
            const auto cv=ugkwp::mixtureHeatCapacity(y,candidate.T,s.gasSpecies.thermo);
            valid=valid&&ugkwp::gasFinite(cv)&&cv>GPU_OPERATOR_R(0.0);
        }
        if(valid)
        {output=candidate;for(int k=0;k<Ns;++k)massFractions[k]=y[k];return true;}
        alpha=attempt==63?GPU_OPERATOR_R(0.0):GPU_OPERATOR_R(0.5)*alpha;
    }
    return false;
}
