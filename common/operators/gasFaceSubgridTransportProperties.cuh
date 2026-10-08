#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ void gasFaceSubgridTransportProperties
(
    const DeviceState& s,
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
    directWallHeatFlux = GPU_OPERATOR_R(0.0);
    directWallHeatFluxActive = 0;
    GPU_OPERATOR_REAL nutFace = nei >= 0
      ? GPU_OPERATOR_R(0.5)*(s.nut[own] + s.nut[nei])
      : s.nut[own];

    if (nei < 0 && boundaryKind == 2)
    {
        if (s.turbulenceModel == 3 && s.sstWallTreatment == 0)
        {
            muTurbulent = GPU_OPERATOR_R(0.0);
            kTurbulent = GPU_OPERATOR_R(0.0);
            return;
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
            const GPU_OPERATOR_REAL wallRho = clampMin
            (
                s.p[own]/(s.Rgas*clampMin(wallTemperature, s.TgasMin)),
                s.rhoMin
            );
            const GPU_OPERATOR_REAL gradient = (wallTemperature-s.Tgas[own])*s.deltaCoeffs[f];
            const auto thermal = ugkpwall::sstJayatillekeThermalTransport
            (
                wallRho, s.gasCp, s.gasMu, s.gasPrClamped,
                s.turbulentPrandtl, s.sstWallCmu, s.sstWallKappa,
                s.sstWallE, s.sstJayatillekeP, s.sstThermalYPlus,
                s.k[own], wallDistance, velocityDifference,
                sqrt(wallUx*wallUx + wallUy*wallUy + wallUz*wallUz),
                gradient
            );
            if (thermal.valid == 0)
            {
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
            kTurbulent = thermal.conductivity - molecularGasConductivity(s);
            return;
        }
        const ugkpwall::WallSubgridTransport wallTransport =
            ugkpwall::wallSubgridTransport
            (
                rhoSafe,
                s.gasCp,
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
        s.gasCp*muT/clampMin(s.turbulentPrandtl, OfSmall);
}
