#pragma once
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
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
    int& directWallHeatFluxActive
)
{
    // Stability estimation also calls this helper for laminar walls. Never
    // evaluate a turbulent wall law (notably nu=0 Spalding) on that path.
    if(s.turbulenceModel==0)
    {
        muTurbulent=GPU_OPERATOR_R(0.0);kTurbulent=GPU_OPERATOR_R(0.0);
        directWallHeatFlux=GPU_OPERATOR_R(0.0);directWallHeatFluxActive=0;
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
        if (s.turbulenceModel == 3 && s.sstWallTreatment != 1)
        {
            muTurbulent = GPU_OPERATOR_R(0.0);
            kTurbulent = GPU_OPERATOR_R(0.0);
            return;
        }
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s) && s.turbulenceModel==3)
        {
            const GPU_OPERATOR_REAL temperature=s.riemannBoundaryTFix[f]!=0?s.riemannBoundaryT[f]:s.Tgas[own];
            if(!(temperature>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(temperature)
                || !(s.rho[own]>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(s.rho[own])
                || !(s.gasMu>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(s.gasMu))
            {
                ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);
                muTurbulent=kTurbulent=GPU_OPERATOR_R(0.0);
                return;
            }
        }
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
        const GPU_OPERATOR_REAL wallDistance = s.turbulenceModel == 3
          ? clampMin(s.sstWallDistance[own], OfVSmall)
          : GPU_OPERATOR_R(1.0)/clampMin(s.deltaCoeffs[f], OfVSmall);
        const GPU_OPERATOR_REAL rhoSafe = clampMin
        (
            riemannFacePrimitiveForGradient(s, own, f).rho, s.rhoMin
        );
        const ugkpwall::SpaldingWallState wallState =
            ugkpwall::spaldingWallStateFromNormalGradient
            (
                velocityDifference,
                wallDistance,
                s.gasMu/rhoSafe,
                velocityDifference*clampMin(s.deltaCoeffs[f], GPU_OPERATOR_R(0.0)),
                s.turbulenceModel == 3 ? s.sstWallKappa : GPU_OPERATOR_R(0.41),
                s.turbulenceModel == 3 ? s.sstWallE : GPU_OPERATOR_R(9.8)
            );
        if (s.turbulenceModel == 3)
        {
            const GPU_OPERATOR_REAL wallTemperature = s.riemannBoundaryTFix[f] != 0
              ? s.riemannBoundaryT[f] : s.Tgas[own];
            GPU_OPERATOR_REAL wallRho = rhoSafe;
            if(!ugkwp::mixtureGasActive(s))
                wallRho=clampMin(s.p[own]/(s.Rgas*clampMin(wallTemperature,s.TgasMin)),s.rhoMin);
            GPU_OPERATOR_REAL wallCp=s.gasCp, wallPr=s.gasPrClamped;
            GPU_OPERATOR_REAL wallP=s.sstJayatillekeP, thermalYPlus=s.sstThermalYPlus;
            GPU_OPERATOR_REAL molecularBaseline=molecularGasConductivity(s);
            if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
                if(ugkwp::mixtureGasActive(s))
                {
                    // The impermeable wall inherits owner composition, but its
                    // thermodynamic properties are evaluated at wall temperature.
                    // No legacy bootstrap Cp/R participates in the shared
                    // mixture closure.
                    constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
                    GPU_OPERATOR_REAL Y[Ns], sum=GPU_OPERATOR_R(0.0);
                    wallCp=GPU_OPERATOR_R(0.0);
                    for(int species=0;species<Ns;++species)
                    {
                        Y[species]=s.gasSpecies.rho[species*s.nCells+own]/s.rho[own];
                        sum+=Y[species];
                        wallCp+=Y[species]*ugkwp::speciesCp(species,wallTemperature,s.gasSpecies.thermo);
                    }
                    const GPU_OPERATOR_REAL R=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);
                    wallRho=s.p[own]/(R*wallTemperature);
                    GPU_OPERATOR_REAL molecular=s.gasMu*wallCp/wallPr;
                    if constexpr(ugkwp::GasDirectConductivity<GasState>::value)
                        if(ugkwp::gasHasDirectConductivity(s))molecular=s.gasThermalConductivity;
                    if(!(sum>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(sum)
                        || fabs(sum-GPU_OPERATOR_R(1.0))>s.gasSpecies.densityClosureTolerance
                        || !(wallCp>R) || !ugkwp::gasFinite(wallCp)
                        || !(wallRho>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(wallRho)
                        || !(molecular>GPU_OPERATOR_R(0.0)) || !ugkwp::gasFinite(molecular))
                    {
                        ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);
                        muTurbulent=kTurbulent=GPU_OPERATOR_R(0.0);
                        return;
                    }
                    if(ugkwp::gasHasDirectConductivity(s))
                    {
                        // Only direct conductivity makes Pr depend on wall Cp(T,Y).
                        // Constant-Pr transport reuses the configuration-stage cache.
                        wallPr=s.gasMu*wallCp/molecular;
                        const GPU_OPERATOR_REAL ratio=wallPr/clampMin(s.turbulentPrandtl,OfSmall);
                        wallP=ugkpwall::jayatillekeSmoothP(ratio);
                        thermalYPlus=ugkpwall::jayatillekeThermalYPlus(ratio,s.sstWallKappa,s.sstWallE);
                    }
                    // Both face flux and diffusion timestep add owner-mixture
                    // molecular conductivity. Return exactly their correction,
                    // even when Cp(Twall) differs from Cp(Towner).
                    if(!ugkwp::gasHasDirectConductivity(s))
                        molecularBaseline=s.gasMu*s.gasSpecies.heatCapacity[own]/s.gasPrClamped;
                }
            const GPU_OPERATOR_REAL gradient = (wallTemperature-s.Tgas[own])*s.deltaCoeffs[f];
            const auto thermal = ugkpwall::sstJayatillekeThermalTransport
            (
                wallRho, wallCp, s.gasMu, wallPr,
                s.turbulentPrandtl, s.sstWallCmu, s.sstWallKappa,
                s.sstWallE, wallP, thermalYPlus,
                s.k[own], wallDistance, velocityDifference,
                sqrt(wallUx*wallUx + wallUy*wallUy + wallUz*wallUz),
                gradient
            );
            if (thermal.valid == 0)
            {
                if(ugkwp::mixtureGasActive(s))
                {
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);
                    muTurbulent=kTurbulent=GPU_OPERATOR_R(0.0);
                    return;
                }
                printf("Jayatilleke thermal closure failed face=%d cell=%d rho=%g k=%g Tw=%g Tc=%g\n",
                    f, own, double(wallRho), double(s.k[own]), double(wallTemperature), double(s.Tgas[own]));
                printf("closure input mu=%g Cp=%g Pr=%g Prt=%g Cmu=%g kappa=%g E=%g P=%g yt=%g y=%g U=%g grad=%g\n",
                    double(s.gasMu),double(s.gasCp),double(s.gasPrClamped),double(s.turbulentPrandtl),double(s.sstWallCmu),double(s.sstWallKappa),double(s.sstWallE),double(s.sstJayatillekeP),double(s.sstThermalYPlus),double(wallDistance),double(velocityDifference),double(gradient));
                asm("trap;");
                return;
            }
            muTurbulent = rhoSafe*wallState.nut;
            directWallHeatFlux = thermal.heatFlux;
            directWallHeatFluxActive = s.riemannBoundaryTFix[f] != 0;
            kTurbulent = thermal.conductivity - molecularBaseline;
            return;
        }
        const ugkpwall::WallSubgridTransport wallTransport =
            ugkpwall::wallSubgridTransport
            (
                rhoSafe,
                turbulentCp,
                s.turbulentPrandtl,
                wallState.nut
            );
        muTurbulent = wallTransport.dynamicViscosity;
        kTurbulent = wallTransport.thermalConductivity;
        return;
    }

    const GPU_OPERATOR_REAL muT = clampMin(rhoFace, s.rhoMin)*fmax(nutFace, GPU_OPERATOR_R(0.0));
    muTurbulent = muT;
    kTurbulent =
        turbulentCp*muT/clampMin(s.turbulentPrandtl, OfSmall);
}
