#pragma once
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
template<bool IncludeTurbulence, class GasState>
__global__ void computeGasInternalFaceFluxKernel(GasState* sp, const GPU_OPERATOR_TIME dt)
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
    if(ugkwp::gasMovingGeometry(s) && !ugkwp::gasGeometryIntervalValid(s,dt))
    {if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return;}
    (void)dt;
    computeRiemannGasFaceFluxDevice<IncludeTurbulence>
    (
        s,
        f,
        s.gasPhiRho[f],
        s.gasPhiRhoUx[f],
        s.gasPhiRhoUy[f],
        s.gasPhiRhoUz[f],
        s.gasPhiRhoE[f]
    );
}

template<class GasState>
__global__ void enforcePeriodicGasFluxAntisymmetryKernel(GasState* sp)
{
    GasState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (!isPeriodicFace(s, f))
    {
        return;
    }
    const int pair = s.facePeriodicPair[f];
    if(ugkwp::gasMovingGeometry(s))
    {
        bool valid=pair>=0 && pair<s.nFaces && pair!=f && s.facePeriodicPair[pair]==f;
        if constexpr(ugkwp::GasGeometryCapability<GasState>::value)
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> first,second;
            valid=valid && ugkwp::gasGeometryFaceFrame(s,f,s.gasGeometry.interval,first)
                && ugkwp::gasGeometryFaceFrame(s,pair,s.gasGeometry.interval,second);
            if(valid)
            {
                const auto&g=s.gasGeometry;
                const GPU_OPERATOR_REAL scale=fmax(fabs(g.faceSweptVolume[f]),fabs(g.faceSweptVolume[pair]));
                valid=ugkwp::gasGeometryDetail::equal(s.Sfx[f],-s.Sfx[pair])
                    &&ugkwp::gasGeometryDetail::equal(s.Sfy[f],-s.Sfy[pair])
                    &&ugkwp::gasGeometryDetail::equal(s.Sfz[f],-s.Sfz[pair])
                    &&fabs(g.faceSweptVolume[f]+g.faceSweptVolume[pair])
                        <=g.absoluteGeometryTolerance+g.relativeGeometryTolerance*scale;
            }
        }
        if(!valid)
        {
            if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");
            if(pair>=0&&pair<s.nFaces)ugkwp::gasRecordFaceFailure(s,pair,ugkwp::GasTransportCode::InvalidGeometry);
            return;
        }
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
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            constexpr GPU_OPERATOR_REAL eps=sizeof(GPU_OPERATOR_REAL)==sizeof(float)
                ? GPU_OPERATOR_REAL(0x1p-23) : GPU_OPERATOR_REAL(0x1p-52);
            GPU_OPERATOR_REAL value[Ns],sumF=0,sumPair=0,magF=0,magPair=0,sum=0,mag=0;
            bool valid=ugkwp::gasFinite(originalMassF)&&ugkwp::gasFinite(originalMassPair);
            for(int k=0;k<Ns;++k)
            {
                const auto* flux=s.gasSpecies.flux+k*s.nFaces;
                valid=valid&&ugkwp::gasFinite(flux[f])&&ugkwp::gasFinite(flux[pair]);
                sumF+=flux[f];sumPair+=flux[pair];magF+=fabs(flux[f]);magPair+=fabs(flux[pair]);
                value[k]=GPU_OPERATOR_R(.5)*(flux[f]-flux[pair]);sum+=value[k];mag+=fabs(value[k]);
            }
            const GPU_OPERATOR_REAL boundF=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(originalMassF),magF);
            const GPU_OPERATOR_REAL boundPair=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(originalMassPair),magPair);
            const GPU_OPERATOR_REAL finalBound=GPU_OPERATOR_R(8*Ns)*eps*fmax(fabs(rho),mag);
            valid=valid&&fabs(sumF-originalMassF)<=boundF&&fabs(sumPair-originalMassPair)<=boundPair;
            // Cancellation can expose rounding present in two individually closed
            // inputs. Repair only that bounded publication error, never bad input.
            if(valid&&fabs(sum-rho)>finalBound)
            {
                const GPU_OPERATOR_REAL allowance=GPU_OPERATOR_R(.5)*(boundF+boundPair)+finalBound;
                valid=fabs(sum-rho)<=allowance;
                int carrier=-1;GPU_OPERATOR_REAL replacement=0,best=-1;
                const int own=s.faceOwner[f],other=s.faceOwner[pair];
                for(int k=Ns-1;valid&&k>=0;--k)
                {
                    GPU_OPERATOR_REAL rest=0;
                    for(int j=0;j<Ns;++j)if(j!=k)rest+=value[j];
                    const GPU_OPERATOR_REAL candidate=rho-rest;
                    const int donor=candidate>0?own:other;
                    const bool legal=candidate==0 || (donor>=0&&donor<s.nCells
                        &&s.gasSpecies.rho[k*s.nCells+donor]>0);
                    if(legal&&fabs(candidate-value[k])<=allowance
                        &&(carrier<0||fabs(value[k])>best))
                    {carrier=k;replacement=candidate;best=fabs(value[k]);}
                }
                valid=valid&&carrier>=0;
                if(valid)value[carrier]=replacement;
            }
            if(!valid)
            {
                ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidComposition);
                ugkwp::gasRecordFaceFailure(s,pair,ugkwp::GasTransportCode::InvalidComposition);
                return;
            }
            for(int k=0;k<Ns;++k)
            {
                auto* flux=s.gasSpecies.flux+k*s.nFaces;
                flux[f]=value[k];flux[pair]=-value[k];
            }
        }
    }
}
