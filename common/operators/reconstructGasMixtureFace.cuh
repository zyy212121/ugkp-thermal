#pragma once
#include "gasTransport/MixtureThermo.H"
#include "Diagnostics.H"

// Mixture extension of the common MUSCL path. The existing Green-Gauss and
// configured scalar limiters supply the slopes; the existing acoustic limiter
// acts in primitive rho/U/p variables with a frozen-composition acoustic basis.
// Formation energy is never used as a perfect-gas Roe enthalpy. Composition has
// one shared slope factor and its final component closes the mass simplex.
// Evaluate the configured reconstruction once, without candidate searches.
template<class GasState>
__device__ bool reconstructGasMixtureFace
(
    const GasState& s, const int c, const int f,
    GasPrimDevice& output, GPU_OPERATOR_REAL* massFractions, const bool nativeRefinement=false, bool* evaluated=nullptr
)
{
    if(evaluated)*evaluated=false;
    constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
    if(!gasReconstructionStorageAvailable(s,c,f) || !massFractions || !s.gasSpecies.rho
        || (s.gasReconstruction==1 && (!s.gasSpecies.thermo.species
            || !s.gasSpecies.limiter || !s.gasSpecies.gradX || !s.gasSpecies.gradY || !s.gasSpecies.gradZ)))
    {
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return false;
    }
    using namespace ugkp::diagnostics;
    const GasPrimDevice centre=gasCellPrimitive(s,c);
    if(centre.rho==GPU_OPERATOR_R(0.0))
    {
        recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,2103,21,0,EntityKind::Face,f,-1,centre.rho,0,0,Quantity::Density));
        return false;
    }
    GPU_OPERATOR_REAL yc[Ns],dy[Ns];
    for(int k=0;k<Ns;++k)
    {
        yc[k]=s.gasSpecies.rho[k*s.nCells+c]/centre.rho;
        dy[k]=GPU_OPERATOR_R(0.0);
        if(!ugkwp::gasFinite(yc[k]))
        {
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2102,21,0,EntityKind::Face,f,k,yc[k],0,0,Quantity::MassFraction));
            return false;
        }
        const auto original=yc[k];
        yc[k]=clampRange(yc[k],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
        if(yc[k]!=original)recordWarning(s.diagnostics,makeRecord(Code::FiniteYClip,2101,21,0,EntityKind::Face,f,k,original,yc[k],yc[k],Quantity::MassFraction));
    }
    (void)nativeRefinement;
    // Fractions live on one simplex. Close on a present, largest carrier:
    // 1-sum(Y_other) must never invent an absent trace species from roundoff.
    int dependent=Ns-1;
    GPU_OPERATOR_REAL total=GPU_OPERATOR_R(0.0);
    for(int k=0;k<Ns;++k){total+=yc[k];if(yc[k]>yc[dependent])dependent=k;}
    if(total==GPU_OPERATOR_R(0.0))
    {
        recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,10050,21,0,
            EntityKind::Face,f,-1,total,0,0,Quantity::Denominator));
        return false;
    }
    GPU_OPERATOR_REAL centreSum=GPU_OPERATOR_R(0.0);
    for(int k=0;k<Ns;++k)if(k!=dependent){yc[k]/=total;centreSum+=yc[k];}
    yc[dependent]=GPU_OPERATOR_R(1.0)-centreSum;
    if(s.gasReconstruction!=1)
    {output=centre;for(int k=0;k<Ns;++k)massFractions[k]=yc[k];if(evaluated)*evaluated=true;return true;}
    if(!gasCoupledFaceStorageAvailable(s,f))
    {
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return false;
    }
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
        if(nei>=s.nCells || !s.faceOwner || s.faceOwner[f]<0 || s.faceOwner[f]>=s.nCells
            || !s.gasSpecies.soundSpeed || !s.faceWeight || !s.Sfx || !s.Sfy || !s.Sfz || !s.magSf)
        {
            if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
            return false;
        }
        const int other=c==s.faceOwner[f]?nei:s.faceOwner[f];
        const auto adjacent=gasCellPrimitive(s,other);
        const GPU_OPERATOR_REAL rl=sqrt(centre.rho),rr=sqrt(adjacent.rho);
        const GPU_OPERATOR_REAL sound=(rl*s.gasSpecies.soundSpeed[c]+rr*s.gasSpecies.soundSpeed[other])/(rl+rr);
        if(s.magSf[f]==GPU_OPERATOR_R(0.0))
        {
            recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,30402,21,0,EntityKind::Face,f,-1,s.magSf[f],0,0,Quantity::Denominator));
            return false;
        }
        if(!::isfinite(s.faceWeight[f]))
        {
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,30403,21,0,EntityKind::Face,f,-1,s.faceWeight[f],0,0,Quantity::LinearSystem));
            return false;
        }
        const GPU_OPERATOR_REAL weight=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
        ugkpcharacteristic::Failure characteristicFailure;
        increment=ugkpcharacteristic::limitOneSide(increment,
            ugkpcharacteristic::Increment{adjacent.rho-centre.rho,adjacent.ux-centre.ux,
                adjacent.uy-centre.uy,adjacent.uz-centre.uz,adjacent.p-centre.p},
            s.Sfx[f]/s.magSf[f],s.Sfy[f]/s.magSf[f],s.Sfz[f]/s.magSf[f],
            rl*rr,sound,c==s.faceOwner[f]?GPU_OPERATOR_R(1.0)-weight:weight, &characteristicFailure);
        if(characteristicFailure.failed)
        {
            const auto code=characteristicFailure.zeroDenominator?Code::ZeroDenominator:
                (::isfinite(characteristicFailure.value)?Code::InvalidPhysicalState:Code::NonFinite);
            recordFatal(s.diagnostics,makeRecord(code,30400,21,0,EntityKind::Face,f,
                characteristicFailure.component,characteristicFailure.value,0,0,Quantity::LinearSystem));
            return false;
        }
    }
    for(int k=0;k<Ns;++k)
    {
        const int i=k*s.nCells+c;
        dy[k]=s.gasSpecies.limiter[c]
            *(s.gasSpecies.gradX[i]*dx+s.gasSpecies.gradY[i]*ddy+s.gasSpecies.gradZ[i]*dz);
    }
    GasPrimDevice candidate{centre.rho+increment.rho,centre.ux+increment.ux,
        centre.uy+increment.uy,centre.uz+increment.uz,centre.p+increment.p,centre.T};
    // One analytic feasible slope, not a candidate search or a physical-dt
    // retry. The dependent slope is the negative sum of the independent ones.
    GPU_OPERATOR_REAL slopeSum=GPU_OPERATOR_R(0.0);
    for(int k=0;k<Ns;++k)if(k!=dependent)slopeSum+=dy[k];
    dy[dependent]=-slopeSum;
    GPU_OPERATOR_REAL alpha=GPU_OPERATOR_R(1.0);
    for(int k=0;k<Ns;++k)
    {
        if(!ugkwp::gasFinite(dy[k]))
        {
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,10048,21,0,
                EntityKind::Face,f,k,dy[k],0,0,Quantity::MassFraction));
            return false;
        }
        if(dy[k]<GPU_OPERATOR_R(0.0))alpha=fmin(alpha,-yc[k]/dy[k]);
        else if(dy[k]>GPU_OPERATOR_R(0.0))alpha=fmin(alpha,
            (GPU_OPERATOR_R(1.0)-yc[k])/dy[k]);
    }
    GPU_OPERATOR_REAL y[Ns];
    int closure=dependent;
    for(int k=0;k<Ns;++k)
    {
        const auto original=yc[k]+dy[k];
        y[k]=clampRange(yc[k]+alpha*dy[k],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
        if(original<GPU_OPERATOR_R(0.0)||original>GPU_OPERATOR_R(1.0))
            recordWarning(s.diagnostics,makeRecord(Code::FiniteYClip,10049,21,0,
                EntityKind::Face,f,k,original,y[k],y[k],Quantity::MassFraction));
    }
    for(int k=0;k<Ns;++k)if(y[k]>y[closure])closure=k;
    // Only roundoff remains after the feasible slope. Put that closure on a
    // largest present component; exact zeros and proportional slopes survive.
    GPU_OPERATOR_REAL sum=GPU_OPERATOR_R(0.0);
    for(int k=0;k<Ns;++k)if(k!=closure)sum+=y[k];
    y[closure]=GPU_OPERATOR_R(1.0)-sum;
    const auto gasConstant=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
    const auto denominator=candidate.rho*gasConstant;
    if(denominator==GPU_OPERATOR_R(0.0))
    {
        recordFatal(s.diagnostics,makeRecord(Code::ZeroDenominator,10050,21,0,EntityKind::Face,f,-1,denominator,0,0,Quantity::Denominator));
        return false;
    }
    candidate.T=candidate.p/denominator;
    const GPU_OPERATOR_REAL primitives[]={candidate.T,candidate.rho,candidate.p,candidate.ux,candidate.uy,candidate.uz};
    for(int component=0;component<6;++component)
    {
        if(!ugkwp::gasFinite(primitives[component]))
        {
            const unsigned site=component==0?2105:(component==1?2106:(component==2?2107:2108));
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,site,21,0,EntityKind::Face,f,component,primitives[component],0,0,component==0?Quantity::Temperature:(component==1?Quantity::Density:(component==2?Quantity::Pressure:Quantity::Velocity))));
            return false;
        }
    }
    if(candidate.T<GPU_OPERATOR_R(0.0))
    {
        recordFatal(s.diagnostics,makeRecord(Code::NegativeTemperature,2104,21,0,EntityKind::Face,f,-1,candidate.T,0,0,Quantity::Temperature));
        return false;
    }
    output=candidate;for(int k=0;k<Ns;++k)massFractions[k]=y[k];
    if(evaluated)*evaluated=true;
    return true;
}
