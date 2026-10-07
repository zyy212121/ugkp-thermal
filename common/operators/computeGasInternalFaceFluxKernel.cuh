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

    const GPU_OPERATOR_REAL rho = GPU_OPERATOR_R(0.5)*(s.gasPhiRho[f] - s.gasPhiRho[pair]);
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
            for (int k=0; k<ugkwp::GasStateTraits<GasState>::speciesCount; ++k)
            {
                auto* flux = s.gasSpecies.flux+k*s.nFaces;
                const GPU_OPERATOR_REAL value = GPU_OPERATOR_R(0.5)*(flux[f]-flux[pair]);
                flux[f]=value; flux[pair]=-value;
            }
    }
}
