#pragma once
#include "Diagnostics.H"
#include "CompressibleSstWallFace.cuh"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"

template<class GasState>
__device__ void gasFaceSubgridTransportProperties
(
    const GasState& s,
    const int f,
    const int own,
    const int nei,
    const int boundaryKind,
    const GPU_OPERATOR_REAL rhoFace,
    GPU_OPERATOR_REAL& muTurbulent,
    GPU_OPERATOR_REAL& kTurbulent,
    GPU_OPERATOR_REAL& directWallHeatFlux,
    int& directWallHeatFluxActive,
    const bool nativeRefinement=false, bool* evaluated=nullptr
)
{
    using namespace ugkp::diagnostics;
    (void)nativeRefinement;
    if(evaluated)*evaluated=false;
    muTurbulent=kTurbulent=directWallHeatFlux=GPU_OPERATOR_R(0.0);
    directWallHeatFluxActive=0;
    const auto failure=[&](Code code,unsigned site,GPU_OPERATOR_REAL value,
        unsigned quantity,int component=-1)
    {
        if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(code,site,26,0,
            EntityKind::Face,f,component,value,0,0,quantity));
        ugkwp::gasRecordFaceFailure(s,f,code==Code::MissingOperator
            ?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);
    };
    const auto publishTransport=[&](GPU_OPERATOR_REAL mu,GPU_OPERATOR_REAL conductivity)
    {
        const GPU_OPERATOR_REAL values[]={mu,conductivity};
        for(int component=0;component<2;++component)
            if(!ugkwp::gasFinite(values[component]))
            {failure(Code::NonFinite,2667,values[component],Quantity::Unspecified,component);return;}
        muTurbulent=mu;kTurbulent=conductivity;
        if(evaluated)*evaluated=true;
    };


    if(s.turbulenceModel==0)
    {
        muTurbulent=GPU_OPERATOR_R(0.0);kTurbulent=GPU_OPERATOR_R(0.0);
        directWallHeatFlux=GPU_OPERATOR_R(0.0);directWallHeatFluxActive=0;
        if(evaluated)*evaluated=true;
        return;
    }
    if(nei>=0)
    {
        GPU_OPERATOR_REAL cpOwner=s.gasCp,cpNeighbour=s.gasCp;
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
            if(ugkwp::mixtureGasActive(s))
            {
                cpOwner=s.gasSpecies.heatCapacity[own];
                cpNeighbour=s.gasSpecies.heatCapacity[nei];
            }
        const GPU_OPERATOR_REAL inputs[]={s.rho[own],s.rho[nei],s.nut[own],s.nut[nei],
            cpOwner,cpNeighbour,s.turbulentPrandtl,s.faceWeight[f]};
        for(int component=0;component<8;++component)
            if(!ugkwp::gasFinite(inputs[component]))
            {failure(Code::NonFinite,2668,inputs[component],Quantity::Unspecified,component);return;}
        const GPU_OPERATOR_REAL ownerWeight=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0),GPU_OPERATOR_R(1));

        const GPU_OPERATOR_REAL w=own==s.faceOwner[f]?ownerWeight:GPU_OPERATOR_R(1)-ownerWeight;



        const GPU_OPERATOR_REAL muOwner=clampMin(s.rho[own],s.rhoMin)*fmax(s.nut[own],GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL muNeighbour=clampMin(s.rho[nei],s.rhoMin)*fmax(s.nut[nei],GPU_OPERATOR_R(0.0));
        publishTransport(w*muOwner+(GPU_OPERATOR_R(1)-w)*muNeighbour,
            (w*cpOwner*muOwner+(GPU_OPERATOR_R(1)-w)*cpNeighbour*muNeighbour)
            /clampMin(s.turbulentPrandtl,OfSmall));
        return;
    }
    GPU_OPERATOR_REAL turbulentCp=s.gasCp;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            const GPU_OPERATOR_REAL w=nei>=0?clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0)):GPU_OPERATOR_R(1.0);
            turbulentCp=w*s.gasSpecies.heatCapacity[own]
                +(GPU_OPERATOR_R(1.0)-w)*s.gasSpecies.heatCapacity[nei>=0?nei:own];
        }
    directWallHeatFlux = GPU_OPERATOR_R(0.0);
    directWallHeatFluxActive = 0;
    GPU_OPERATOR_REAL nutFace = nei >= 0
      ? GPU_OPERATOR_R(0.5)*(s.nut[own] + s.nut[nei])
      : s.nut[own];

    if (nei < 0 && boundaryKind == 2)
    {
        const GPU_OPERATOR_REAL muWall=ugkwp::gasFaceMolecularViscosity(s,f,own,nei);
        if (s.turbulenceModel == 3)
        {
            if (ugkwp::gaswall::usesAlgebraicWallFunction(s.sstWallTreatment))
            {
                const auto wall = evaluateCompressibleSstWallFace(s, f, own, muWall);
                if (!wall.valid) return;
                muTurbulent = wall.dynamicViscosity;
                kTurbulent = wall.excessConductivity;
                directWallHeatFlux = wall.heatFlux;
                directWallHeatFluxActive = 1;
            }
            if (evaluated) *evaluated = true;
            return;
        }
        const GPU_OPERATOR_REAL velocities[]={s.Ux[own],s.Uy[own],s.Uz[own],
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUx[f]:GPU_OPERATOR_R(0),
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUy[f]:GPU_OPERATOR_R(0),
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUz[f]:GPU_OPERATOR_R(0)};
        for(int component=0;component<6;++component)
            if(!ugkwp::gasFinite(velocities[component]))
            {failure(Code::NonFinite,2652,velocities[component],Quantity::Velocity,component%3);return;}
        const GPU_OPERATOR_REAL transportInputs[]={muWall,s.p[own],s.deltaCoeffs[f],s.turbulentPrandtl,turbulentCp};
        for(int component=0;component<5;++component)
            if(!ugkwp::gasFinite(transportInputs[component]))
            {failure(Code::NonFinite,2664,transportInputs[component],component==1?Quantity::Pressure:Quantity::Unspecified,component);return;}
        const GPU_OPERATOR_REAL rhoSafe = clampMin
        (
            riemannFacePrimitiveForGradient(s, own, f).rho, s.rhoMin
        );
        const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL dux = s.Ux[own] - wallUx;
        const GPU_OPERATOR_REAL duy = s.Uy[own] - wallUy;
        const GPU_OPERATOR_REAL duz = s.Uz[own] - wallUz;
        const GPU_OPERATOR_REAL velocityDifference = sqrt
        (
            dux*dux + duy*duy + duz*duz
        );
        const GPU_OPERATOR_REAL wallDistance = GPU_OPERATOR_R(1.0)/clampMin(s.deltaCoeffs[f], OfVSmall);
        const ugkpwall::SpaldingWallState wallState =
            ugkpwall::spaldingWallStateFromNormalGradient
            (
                velocityDifference,
                wallDistance,
                muWall/rhoSafe,
                velocityDifference*clampMin(s.deltaCoeffs[f], GPU_OPERATOR_R(0.0)),
                GPU_OPERATOR_R(0.41),
                GPU_OPERATOR_R(9.8), s.diagnostics, f,
                s.gasScalarSolvers.spaldingMaximumIterations, s.gasScalarSolvers.spaldingTolerance
            );
        const GPU_OPERATOR_REAL mechanicalOutputs[]={wallState.uTau,wallState.yPlus,wallState.nut};
        for(int component=0;component<3;++component)
            if(!ugkwp::gasFinite(mechanicalOutputs[component]))
            {failure(Code::NonFinite,2666,mechanicalOutputs[component],Quantity::Unspecified,component);return;}
        const ugkpwall::WallSubgridTransport wallTransport =
            ugkpwall::wallSubgridTransport
            (
                rhoSafe,
                turbulentCp,
                s.turbulentPrandtl,
                wallState.nut
            );
        publishTransport(wallTransport.dynamicViscosity,wallTransport.thermalConductivity);
        return;
    }

    const GPU_OPERATOR_REAL bulkInputs[]={rhoFace,nutFace,turbulentCp,s.turbulentPrandtl};
    for(int component=0;component<4;++component)
        if(!ugkwp::gasFinite(bulkInputs[component]))
        {failure(Code::NonFinite,2668,bulkInputs[component],component==0?Quantity::Density:Quantity::Unspecified,component);return;}
    const GPU_OPERATOR_REAL muT = clampMin(rhoFace, s.rhoMin)*fmax(nutFace, GPU_OPERATOR_R(0.0));
    publishTransport(muT,turbulentCp*muT/clampMin(s.turbulentPrandtl, OfSmall));
}
