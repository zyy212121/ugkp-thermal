#pragma once
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
template<bool IncludeTurbulence, class GasState>
__global__ void computeGasInternalFaceFluxKernel(GasState* sp, const GPU_OPERATOR_TIME dt, const bool nativeRefinement=false)
{
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    s.gasPhiRho[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUx[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUy[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUz[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoE[f] = GPU_OPERATOR_R(0.0);

    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
                s.gasSpecies.flux[k*s.nFaces+f] = GPU_OPERATOR_R(0.0);
    }
    (void)dt;
    computeRiemannGasFaceFluxDevice<IncludeTurbulence>
    (
        s,
        f,
        s.gasPhiRho[f],
        s.gasPhiRhoUx[f],
        s.gasPhiRhoUy[f],
        s.gasPhiRhoUz[f],
        s.gasPhiRhoE[f], nativeRefinement
    );
}

template<class GasState>
__global__ void enforcePeriodicGasFluxAntisymmetryKernel(GasState* sp, const bool nativeRefinement=false)
{
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if(f<s.nInternalFaces || f>=s.nFaces)return;
    if(!s.facePeriodicPair)
    {
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return;
    }
    const int pair=s.facePeriodicPair[f];
    if(pair<0)return;
    if(pair>=s.nFaces)
    {
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return;
    }
    if (f > pair)
    {
        return;
    }

    const GPU_OPERATOR_REAL originalMassF=s.gasPhiRho[f],originalMassPair=s.gasPhiRho[pair];
    const GPU_OPERATOR_REAL rho = GPU_OPERATOR_R(0.5)*(originalMassF - originalMassPair);
    const GPU_OPERATOR_REAL rhoUx = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUx[f] - s.gasPhiRhoUx[pair]);
    const GPU_OPERATOR_REAL rhoUy = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUy[f] - s.gasPhiRhoUy[pair]);
    const GPU_OPERATOR_REAL rhoUz = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUz[f] - s.gasPhiRhoUz[pair]);
    const GPU_OPERATOR_REAL rhoE = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoE[f] - s.gasPhiRhoE[pair]);
    if((!ugkwp::gasFinite(rho)||!ugkwp::gasFinite(rhoUx)||!ugkwp::gasFinite(rhoUy)
        ||!ugkwp::gasFinite(rhoUz)||!ugkwp::gasFinite(rhoE)))
    {
        ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
        ugkwp::gasRecordFaceFailure(s,pair,ugkwp::GasTransportCode::NonFiniteState);
        return;
    }
    if(nativeRefinement)
    {
        s.gasPhiRho[f]=rho;s.gasPhiRhoUx[f]=rhoUx;s.gasPhiRhoUy[f]=rhoUy;s.gasPhiRhoUz[f]=rhoUz;s.gasPhiRhoE[f]=rhoE;
        s.gasPhiRho[pair]=-rho;s.gasPhiRhoUx[pair]=-rhoUx;s.gasPhiRhoUy[pair]=-rhoUy;s.gasPhiRhoUz[pair]=-rhoUz;s.gasPhiRhoE[pair]=-rhoE;
    }
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            constexpr GPU_OPERATOR_REAL eps=GPU_OPERATOR_REAL(0x1p-52);
            GPU_OPERATOR_REAL value[Ns],sumF=0,sumPair=0,magF=0,magPair=0,sum=0,mag=0;
            bool valid=ugkwp::gasFinite(originalMassF)&&ugkwp::gasFinite(originalMassPair);
            for(int k=0;k<Ns;++k)
            {
                const auto* flux=s.gasSpecies.flux+k*s.nFaces;
                if((!ugkwp::gasFinite(flux[f])||!ugkwp::gasFinite(flux[pair])))
                {
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
                    ugkwp::gasRecordFaceFailure(s,pair,ugkwp::GasTransportCode::NonFiniteState);
                    return;
                }
                valid=valid&&ugkwp::gasFinite(flux[f])&&ugkwp::gasFinite(flux[pair]);
                sumF+=flux[f];sumPair+=flux[pair];magF+=fabs(flux[f]);magPair+=fabs(flux[pair]);
                value[k]=GPU_OPERATOR_R(.5)*(flux[f]-flux[pair]);
                if(!ugkwp::gasFinite(value[k]))
                {
                    if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState))asm("trap;");
                    ugkwp::gasRecordFaceFailure(s,pair,ugkwp::GasTransportCode::NonFiniteState);
                    return;
                }
                sum+=value[k];mag+=fabs(value[k]);
            }
            const GPU_OPERATOR_REAL boundF=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(originalMassF),magF);
            const GPU_OPERATOR_REAL boundPair=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(originalMassPair),magPair);
            const GPU_OPERATOR_REAL finalBound=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(rho),mag);
            if(nativeRefinement)valid=valid&&fabs(sumF-originalMassF)<=boundF&&fabs(sumPair-originalMassPair)<=boundPair;
            else
            valid=valid&&ugkwp::gasFinite(sumF)&&ugkwp::gasFinite(sumPair)
                &&ugkwp::gasFinite(magF)&&ugkwp::gasFinite(magPair)
                &&ugkwp::gasFinite(sum)&&ugkwp::gasFinite(mag)
                &&ugkwp::gasFinite(boundF)&&ugkwp::gasFinite(boundPair)&&ugkwp::gasFinite(finalBound)
                &&fabs(sumF-originalMassF)<=boundF&&fabs(sumPair-originalMassPair)<=boundPair;
            // Cancellation can expose rounding present in two individually closed
            // inputs. Repair only that bounded publication error, never bad input.
            if(valid&&fabs(sum-rho)>finalBound)
            {
                const GPU_OPERATOR_REAL allowance=GPU_OPERATOR_R(.5)*(boundF+boundPair)+finalBound;
                valid=(nativeRefinement || ugkwp::gasFinite(allowance))&&fabs(sum-rho)<=allowance;
                // Fix the largest original carrier before computing a correction;
                // do not probe alternate species when the correction is unsuitable.
                int carrier=Ns-1;
                for(int k=0;k<Ns;++k)if(fabs(value[k])>fabs(value[carrier]))carrier=k;
                GPU_OPERATOR_REAL rest=0;
                for(int k=0;k<Ns;++k)if(k!=carrier)rest+=value[k];
                const GPU_OPERATOR_REAL replacement=rho-rest;
                const int donor=replacement>0?s.faceOwner[f]:s.faceOwner[pair];
                const bool legal=replacement==0 || (donor>=0&&donor<s.nCells
                    &&s.gasSpecies.rho[carrier*s.nCells+donor]>0);
                valid=valid&&legal&&ugkwp::gasFinite(replacement)&&fabs(replacement-value[carrier])<=allowance;
                if(valid)value[carrier]=replacement;
            }
            // These sums are conservation audits, not published flux fields.
            // Overflow of an audit cannot discard independently finite fluxes.
            valid=valid&&ugkwp::gasFinite(sumF)&&ugkwp::gasFinite(sumPair)
                &&ugkwp::gasFinite(sum)&&ugkwp::gasFinite(magF)
                &&ugkwp::gasFinite(magPair)&&ugkwp::gasFinite(mag);
            if(!valid)
            {
                using namespace ugkp::diagnostics;
                recordWarning(s.diagnostics,makeRecord(Code::ConservationResidual,2671,31,0,
                    EntityKind::Face,f,-1,sum,rho,sum,Quantity::Conservation));
            }
            for(int k=0;k<Ns;++k)
            {
                auto* flux=s.gasSpecies.flux+k*s.nFaces;
                flux[f]=value[k];flux[pair]=-value[k];
            }
        }
    }
    if(!nativeRefinement)
    {
        s.gasPhiRho[f] = rho;
        s.gasPhiRhoUx[f] = rhoUx;
        s.gasPhiRhoUy[f] = rhoUy;
        s.gasPhiRhoUz[f] = rhoUz;
        s.gasPhiRhoE[f] = rhoE;
        s.gasPhiRho[pair] = -rho;
        s.gasPhiRhoUx[pair] = -rhoUx;
        s.gasPhiRhoUy[pair] = -rhoUy;
        s.gasPhiRhoUz[pair] = -rhoUz;
        s.gasPhiRhoE[pair] = -rhoE;
    }
}
