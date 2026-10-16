#pragma once
#include "gasTransport/SpeciesDiffusion.H"
#include "gasTransport/GasGeometryValidation.H"
#ifndef GPU_GAS_WALL_EXPOSURE
#define GPU_GAS_WALL_EXPOSURE(s, f, neighbour, kind) GPU_OPERATOR_R(1.0)
#endif
template<bool IncludeTurbulence, bool MassOnly = false, class GasState>
__device__ bool computeRiemannGasFaceFluxDevice
(
    const GasState& s,
    const int f,
    GPU_OPERATOR_REAL& massFluxArea,
    GPU_OPERATOR_REAL& momFluxXArea,
    GPU_OPERATOR_REAL& momFluxYArea,
    GPU_OPERATOR_REAL& momFluxZArea,
    GPU_OPERATOR_REAL& energyFluxArea,
    const bool nativeRefinement=false
)
{
    bool numericalSuccess=true;
    massFluxArea = GPU_OPERATOR_R(0.0);
    momFluxXArea = GPU_OPERATOR_R(0.0);
    momFluxYArea = GPU_OPERATOR_R(0.0);
    momFluxZArea = GPU_OPERATOR_R(0.0);
    energyFluxArea = GPU_OPERATOR_R(0.0);

    if (f < 0 || f >= s.nFaces)
    {
        ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage,2602,26,ugkp::diagnostics::Quantity::Index);
        return false;
    }
    if (!s.faceOwner || !gasCoupledFaceStorageAvailable(s,f))
    {
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return false;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        using namespace ugkp::diagnostics;
        if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::InvalidIndex,10040,26,0,EntityKind::Face,f,-1,own,0,0,Quantity::Index));
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return false;
    }

    const int nei = coupledFaceNeighbour(s, f);
    if(nei>=s.nCells || (nei<0 && !s.riemannBoundaryKind))
    {
        using namespace ugkp::diagnostics;
        if(nei>=s.nCells && s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::InvalidIndex,10041,26,0,EntityKind::Face,f,-1,nei,0,0,Quantity::Index));
        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
        return false;
    }
    const int boundaryKind = nei >= 0 ? 0 : s.riemannBoundaryKind[f];
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
        {
            if (!s.gasSpecies.faceStatus) { asm("trap;"); return false; }
            if (!s.gasSpecies.cellStatus)
            {
                ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);
                return false;
            }
            if (s.gasSpecies.cellStatus[own]!=0 || (nei>=0 && s.gasSpecies.cellStatus[nei]!=0))
            { gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false; }

        }
    }
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasState>::value)
    {
        const int slot=ugkwp::gasBoundaryLayerFaceSlot(s,f);
        if(slot>=0)
        {
            const auto& wall=s.gasBoundaryLayer;
            if(slot>=wall.count)
            {
                if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
                return false;
            }
            if(boundaryKind==3 || boundaryKind==4)
            {
                if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
                return false;
            }
            if constexpr(MassOnly && ugkwp::GasSuppliedWallInputCapability<GasState>::value)
                if(s.gasBoundaryLayerModel.useSuppliedWallState
                    && (!wall.exchange || !wall.exchange[slot].massReady)){
                    if(!s.gasBoundaryLayerModel.input||!s.magSf){
                        if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");
                        return false;
                    }
                    GPU_OPERATOR_REAL mass=GPU_OPERATOR_R(0);
                    for(int species=0;species<ugkwp::GasStateTraits<GasState>::speciesCount;++species)
                        mass+=s.gasBoundaryLayerModel.input[slot].massFlux[species];
                    massFluxArea=-s.magSf[f]*mass;
                    const bool numerical=finiteDevice(massFluxArea);
                    if(!numerical)ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
                    return numerical;
                }
            const bool storage=wall.exchange;
            const bool ready=storage && (MassOnly?wall.exchange[slot].massReady:wall.exchange[slot].ready);
            if(!ready)
            {if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage))asm("trap;");return false;}
            if(nativeRefinement && boundaryKind!=2)
            {if(!gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");return false;}
            const auto& exchange=wall.exchange[slot];
            bool wallNumerical=true;
            if(!finiteDevice(exchange.mass))
            {ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);return false;}
            massFluxArea=exchange.mass;
            if constexpr(!MassOnly)
            {
                momFluxXArea=exchange.momentumX;momFluxYArea=exchange.momentumY;momFluxZArea=exchange.momentumZ;
                energyFluxArea=exchange.energy;
                // Keep the ordinary wall's local reconstructed pressure force.
                // Matching pressure belongs to BVP thermodynamics; imposing it
                // as a nonlocal acoustic boundary destabilizes the bulk row.
                // Correct only p*n and p*w_n: species enthalpy, blowing kinetic
                // energy, heat and traction work retain the evaluated wall state.
                bool localAvailable=false;
                const GasPrimDevice local=reconstructGasCellToFace(s,own,f,nativeRefinement,&localAvailable);
                if(!localAvailable){if(!nativeRefinement)ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                GPU_OPERATOR_REAL meshNormal=GPU_OPERATOR_R(0.0);
                if(!finiteDevice(exchange.matchingPressure)
                    || !finiteDevice(local.p)
                    || ugkwp::gasFaceFailure(s,f)!=0)
                {const bool recorded=gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);
                    if(!recorded)asm("trap;");return false;}
                if(!gasBoundaryMeshNormalSpeed(s,f,meshNormal,nativeRefinement))return false;
                const GPU_OPERATOR_REAL pressureCorrection=local.p-exchange.matchingPressure;
                momFluxXArea+=pressureCorrection*s.Sfx[f];
                momFluxYArea+=pressureCorrection*s.Sfy[f];
                momFluxZArea+=pressureCorrection*s.Sfz[f];
                energyFluxArea+=pressureCorrection*s.magSf[f]*meshNormal;
                if(!finiteDevice(momFluxXArea) || !finiteDevice(momFluxYArea)
                    || !finiteDevice(momFluxZArea) || !finiteDevice(energyFluxArea))
                {const bool recorded=ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
                    if(!recorded)asm("trap;");return false;}
                if constexpr(ugkwp::GasStateTraits<GasState>::speciesCount>0)
                    if(ugkwp::mixtureGasActive(s))
                    {
                        if(!wall.speciesFlux||!s.gasSpecies.flux)
                        {ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                        for(int species=0;species<ugkwp::GasStateTraits<GasState>::speciesCount;++species)
                            s.gasSpecies.flux[species*s.nFaces+f]=wall.speciesFlux[species*wall.count+slot];
                    }
            }
            return wallNumerical;
        }
    }
    GPU_OPERATOR_REAL mappedNeiCx = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL mappedNeiCy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL mappedNeiCz = GPU_OPERATOR_R(0.0);
    if (nei >= 0)
    {
        periodicMappedCellCentre
        (
            s, f, nei, mappedNeiCx, mappedNeiCy, mappedNeiCz
        );
    }
    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
    const GPU_OPERATOR_REAL nx = s.Sfx[f]/area;
    const GPU_OPERATOR_REAL ny = s.Sfy[f]/area;
    const GPU_OPERATOR_REAL nz = s.Sfz[f]/area;
    ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> meshFrame;
    const bool moving=ugkwp::gasMovingGeometry(s);
    if constexpr (ugkwp::GasGeometryCapability<GasState>::value)
        if(moving)
        {
            bool frameEvaluated=false;
            if(!ugkwp::gasGeometryFaceFrame(s,f,s.gasGeometry.interval,meshFrame,&frameEvaluated))
            {
                numericalSuccess=false;
                if(!gasRecordEvaluatedFaceFailure(s,f,frameEvaluated?ugkwp::GasTransportCode::InvalidGeometry:ugkwp::GasTransportCode::InvalidStorage)
                    && (nativeRefinement || !frameEvaluated))asm("trap;");
                return false;
            }
            if(ugkwp::gasFaceFailure(s,f)!=0 || ugkwp::gasCellFailure(s,own)!=0
                || (nei>=0 && ugkwp::gasCellFailure(s,nei)!=0))
            {
                numericalSuccess=false;
                if((nativeRefinement || ugkwp::gasFaceFailure(s,f)==0)
                    && !gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics)
                    && nativeRefinement)asm("trap;");
                return false;
            }

        }
    if(boundaryKind==3 || boundaryKind==4)
    {
        if(moving && meshFrame.meshVolumeRate!=GPU_OPERATOR_R(0.0))
            if(!gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");
        return false;
    }
    // Preserve ALE validation above before the mass-only impermeable shortcut.
    if constexpr (MassOnly)
        if (boundaryKind == 1 || boundaryKind == 2) return true;
    GasPrimDevice left = reconstructGasCellToFace(s,own,f,nativeRefinement);
    if(ugkp::diagnostics::hasFatal(s.diagnostics))return false;
    GasPrimDevice right = left;

    GPU_OPERATOR_REAL massFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxX = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxY = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxZ = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL energyFlux = GPU_OPERATOR_R(0.0);

    if (boundaryKind == 1)
    {
        right = riemannFacePrimitiveForGradient(s, own, f);
                                                                          
                                                                            
        momentumFluxX = left.p*nx;
        momentumFluxY = left.p*ny;
        momentumFluxZ = left.p*nz;
        if(moving)energyFlux=left.p*(meshFrame.wx*nx+meshFrame.wy*ny+meshFrame.wz*nz);
    }
    else if (boundaryKind == 2)
    {
        right = riemannFacePrimitiveForGradient(s, own, f);
        const GPU_OPERATOR_REAL wallUx = right.ux;
        const GPU_OPERATOR_REAL wallUy = right.uy;
        const GPU_OPERATOR_REAL wallUz = right.uz;
        momentumFluxX = left.p*nx;
        momentumFluxY = left.p*ny;
        momentumFluxZ = left.p*nz;
        energyFlux =
            momentumFluxX*wallUx
          + momentumFluxY*wallUy
          + momentumFluxZ*wallUz;
        if(moving)energyFlux=left.p*(meshFrame.wx*nx+meshFrame.wy*ny+meshFrame.wz*nz);
    }
    else
    {
        // The mixture MUSCL boundary is evaluated below together with its
        // reconstructed Y. An earlier cell-Y EOS check could latch a spurious
        // temperature-range failure before the consistent boundary exists.
        right = nei >= 0
          ? reconstructGasCellToFace(s,nei,f,nativeRefinement)
          : (ugkwp::mixtureGasActive(s) && s.gasReconstruction==1)
            ? left : riemannExteriorStateForFace(s, f, left,nativeRefinement);
        if(ugkp::diagnostics::hasFatal(s.diagnostics))return false;
        if (s.gasReconstruction == 1 && nei >= 0 && !ugkwp::mixtureGasActive(s))
        {
            const ugkpriemann::Primitive ownerCentre
            {
                s.rho[own], s.Ux[own], s.Uy[own], s.Uz[own], s.p[own]
            };
            const ugkpriemann::Primitive neighbourCentre
            {
                s.rho[nei], s.Ux[nei], s.Uy[nei], s.Uz[nei], s.p[nei]
            };
            const ugkpriemann::RoeAverage roe =
                ugkpriemann::makeRoeAverage
                (
                    ownerCentre,
                    neighbourCentre,
                    nx,
                    ny,
                    nz,
                    s.gammaGas
                );
            if (roe.valid)
            {
                ugkpcharacteristic::Increment leftIncrement
                {
                    left.rho - ownerCentre.rho,
                    left.ux - ownerCentre.ux,
                    left.uy - ownerCentre.uy,
                    left.uz - ownerCentre.uz,
                    left.p - ownerCentre.p
                };
                ugkpcharacteristic::Increment rightIncrement
                {
                    right.rho - neighbourCentre.rho,
                    right.ux - neighbourCentre.ux,
                    right.uy - neighbourCentre.uy,
                    right.uz - neighbourCentre.uz,
                    right.p - neighbourCentre.p
                };
                const ugkpcharacteristic::Increment centreDifference
                {
                    neighbourCentre.rho - ownerCentre.rho,
                    neighbourCentre.ux - ownerCentre.ux,
                    neighbourCentre.uy - ownerCentre.uy,
                    neighbourCentre.uz - ownerCentre.uz,
                    neighbourCentre.p - ownerCentre.p
                };
                ugkpcharacteristic::limitFacePair
                (
                    leftIncrement,
                    rightIncrement,
                    centreDifference,
                    nx,
                    ny,
                    nz,
                    roe.density,
                    roe.soundSpeed,
                    s.faceWeight[f]
                );
                left = makeGasPrimDevice
                (
                    ownerCentre.rho + leftIncrement.rho,
                    ownerCentre.ux + leftIncrement.ux,
                    ownerCentre.uy + leftIncrement.uy,
                    ownerCentre.uz + leftIncrement.uz,
                    ownerCentre.p + leftIncrement.p,
                    s.Rgas,
                    s.rhoMin,
                    s.TgasMin
                );
                right = makeGasPrimDevice
                (
                    neighbourCentre.rho + rightIncrement.rho,
                    neighbourCentre.ux + rightIncrement.ux,
                    neighbourCentre.uy + rightIncrement.uy,
                    neighbourCentre.uz + rightIncrement.uz,
                    neighbourCentre.p + rightIncrement.p,
                    s.Rgas,
                    s.rhoMin,
                    s.TgasMin
                );
            }
        }
        ugkpriemann::Primitive leftRiemann
        {
            left.rho, left.ux, left.uy, left.uz, left.p
        };
        ugkpriemann::Primitive rightRiemann
        {
            right.rho, right.ux, right.uy, right.uz, right.p
        };
        ugkpriemann::Scheme scheme;
        if (!ugkpriemann::schemeFromCreateCode(s.gasFluxScheme, scheme))
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::MissingOperator,2606,26,0,
                EntityKind::Face,f,-1,s.gasFluxScheme));
            ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::UnsupportedConfiguration);
            return false;
        }
        GPU_OPERATOR_REAL hllcAdcOmega = GPU_OPERATOR_R(1.0);
        if (scheme == ugkpriemann::Scheme::HLLC_ADC && !ugkwp::mixtureGasActive(s))
        {
            hllcAdcOmega = clampRange
            (
                finiteOr(s.gasHllcAdcSensor[own], GPU_OPERATOR_R(0.0)),
                GPU_OPERATOR_R(0.0),
                GPU_OPERATOR_R(1.0)
            );
            if (nei >= 0)
            {
                hllcAdcOmega = fmin
                (
                    hllcAdcOmega,
                    clampRange
                    (
                        finiteOr(s.gasHllcAdcSensor[nei], GPU_OPERATOR_R(0.0)),
                        GPU_OPERATOR_R(0.0),
                        GPU_OPERATOR_R(1.0)
                    )
                );
            }
        }
        ugkpriemann::FluxResult result;
        bool mixtureEvaluated = false;
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        {
            if (ugkwp::mixtureGasActive(s))
            {
                if((s.gasFluxScheme!=1 && s.gasFluxScheme!=2)
                    || (s.gasReconstruction!=0 && s.gasReconstruction!=1))
                {
                    using namespace ugkp::diagnostics;
                    recordFatal(s.diagnostics,makeRecord(Code::MissingOperator,10042,26,0,
                        EntityKind::Face,f,s.gasFluxScheme,s.gasReconstruction));
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::UnsupportedConfiguration);
                    return false;
                }
                // A composition contact carries formation energy through the
                // very same Rusanov dissipation applied to partial densities.
                // Derived Real flux input only: never feeds back into rhoE storage.
                ugkpriemann::Conservative leftConserved
                {{s.rho[own],s.rhoUx[own],s.rhoUy[own],s.rhoUz[own],GPU_OPERATOR_REAL(s.rhoE[own])}};
                constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
                using TableReal=typename std::remove_reference_t<decltype(s.gasSpecies)>::RealType;
                GPU_OPERATOR_REAL leftSpecies[Ns],rightSpecies[Ns];
                for(int k=0;k<Ns;++k)leftSpecies[k]=s.gasSpecies.rho[k*s.nCells+own];
                GPU_OPERATOR_REAL leftSound=s.gasSpecies.soundSpeed[own];
                GPU_OPERATOR_REAL rightSound=s.gasSpecies.soundSpeed[nei>=0?nei:own];
                ugkpriemann::Conservative rightConserved{};
                if(nei>=0)
                {
                    rightConserved=ugkpriemann::Conservative{{s.rho[nei],s.rhoUx[nei],s.rhoUy[nei],s.rhoUz[nei],GPU_OPERATOR_REAL(s.rhoE[nei])}};
                    for(int k=0;k<Ns;++k)rightSpecies[k]=s.gasSpecies.rho[k*s.nCells+nei];
                }
                else
                {
                    if(s.gasSpecies.faceStatus[f]!=0){return false;}
                    GPU_OPERATOR_REAL boundaryY[Ns];
                    if(s.gasReconstruction==1)
                    {
                        bool reconstructionEvaluated=false;
                        if(!reconstructGasMixtureFace(s,own,f,left,boundaryY,nativeRefinement,&reconstructionEvaluated))
                        {ugkwp::gasRecordFaceFailure(s,f,!nativeRefinement&&!reconstructionEvaluated?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                        right=riemannBoundaryState(s,f,left,boundaryY,nativeRefinement);
                        if(ugkp::diagnostics::hasFatal(s.diagnostics))return false;
                        rightRiemann={right.rho,right.ux,right.uy,right.uz,right.p};
                    }
                    else for(int k=0;k<Ns;++k)boundaryY[k]=s.gasSpecies.rho[k*s.nCells+own]/s.rho[own];
                    if(s.gasSpecies.faceStatus[f]!=0){return false;}
                    for(int k=0;k<Ns;++k)rightSpecies[k]=right.rho*
                        (s.gasSpecies.compositionBoundaryFixed[f]!=0?s.gasSpecies.boundaryMassFraction[k*s.nFaces+f]
                            :boundaryY[k]);
                    const GPU_OPERATOR_REAL R=ugkwp::mixtureGasConstant(rightSpecies,s.gasSpecies.thermo);
                    GPU_OPERATOR_REAL mixtureEnergy,capacity;bool thermoEvaluated=false;
                    if(nativeRefinement)
                    {mixtureEnergy=ugkwp::mixtureEnergy(rightSpecies,right.T,s.gasSpecies.thermo);capacity=ugkwp::mixtureHeatCapacity(rightSpecies,right.T,s.gasSpecies.thermo);thermoEvaluated=true;}
                    else
                    {
                        const auto caloric=ugkwp::evaluateMixtureEnergy<false,GPU_OPERATOR_REAL,Ns,TableReal,true,true>(rightSpecies,right.T,s.gasSpecies.thermo,mixtureEnergy,capacity,&thermoEvaluated);
                        if(caloric!=ugkwp::ThermoStatus::Success){gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                    }
                    if(!thermoEvaluated){ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                    const GPU_OPERATOR_REAL cv=capacity/right.rho;
                    rightSound=sqrt((cv+R)/cv*R*right.T);
                    rightConserved=ugkpriemann::Conservative{{right.rho,right.rho*right.ux,right.rho*right.uy,right.rho*right.uz,
                        mixtureEnergy
                        +GPU_OPERATOR_R(0.5)*right.rho*(right.ux*right.ux+right.uy*right.uy+right.uz*right.uz)}};
                }
                if(s.gasReconstruction==1)
                {
                    GPU_OPERATOR_REAL y[Ns];
                    bool reconstructionEvaluated=false;
                    if(!reconstructGasMixtureFace(s,own,f,left,y,nativeRefinement,&reconstructionEvaluated))
                    {ugkwp::gasRecordFaceFailure(s,f,!nativeRefinement&&!reconstructionEvaluated?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                    for(int k=0;k<Ns;++k)leftSpecies[k]=left.rho*y[k];
                    GPU_OPERATOR_REAL leftEnergy;bool leftThermoEvaluated=false;
                    if(nativeRefinement)
                    {leftEnergy=ugkwp::mixtureEnergy(leftSpecies,left.T,s.gasSpecies.thermo);leftThermoEvaluated=true;}
                    else
                    {
                        const auto caloric=ugkwp::evaluateMixtureInternalEnergy(leftSpecies,left.T,s.gasSpecies.thermo,leftEnergy,&leftThermoEvaluated);
                        if(caloric!=ugkwp::ThermoStatus::Success){gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                    }
                    if(!leftThermoEvaluated){ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                    leftConserved=ugkpriemann::Conservative{{left.rho,left.rho*left.ux,left.rho*left.uy,left.rho*left.uz,
                        leftEnergy
                        +GPU_OPERATOR_R(0.5)*left.rho*(left.ux*left.ux+left.uy*left.uy+left.uz*left.uz)}};
                    const auto R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
                    GPU_OPERATOR_REAL cv;
                    if(nativeRefinement)cv=ugkwp::mixtureHeatCapacity(y,left.T,s.gasSpecies.thermo);
                    else
                    {
                        bool evaluated=false;
                        const auto caloric=ugkwp::evaluateMixtureHeatCapacity(y,left.T,s.gasSpecies.thermo,cv,&evaluated);
                        if(caloric!=ugkwp::ThermoStatus::Success){gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                        if(!evaluated){ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                    }
                    leftSound=sqrt((cv+R)/cv*R*left.T);
                    if(nei>=0)
                    {
                        bool reconstructionEvaluated=false;
                        if(!reconstructGasMixtureFace(s,nei,f,right,y,nativeRefinement,&reconstructionEvaluated))
                        {ugkwp::gasRecordFaceFailure(s,f,!nativeRefinement&&!reconstructionEvaluated?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                        for(int k=0;k<Ns;++k)rightSpecies[k]=right.rho*y[k];
                        GPU_OPERATOR_REAL rightEnergy;bool rightThermoEvaluated=false;
                        if(nativeRefinement)
                        {rightEnergy=ugkwp::mixtureEnergy(rightSpecies,right.T,s.gasSpecies.thermo);rightThermoEvaluated=true;}
                        else
                        {
                            const auto caloric=ugkwp::evaluateMixtureInternalEnergy(rightSpecies,right.T,s.gasSpecies.thermo,rightEnergy,&rightThermoEvaluated);
                            if(caloric!=ugkwp::ThermoStatus::Success){gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                        }
                        if(!rightThermoEvaluated){ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                        rightConserved=ugkpriemann::Conservative{{right.rho,right.rho*right.ux,right.rho*right.uy,right.rho*right.uz,
                            rightEnergy
                            +GPU_OPERATOR_R(0.5)*right.rho*(right.ux*right.ux+right.uy*right.uy+right.uz*right.uz)}};
                        const auto Rright=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
                        GPU_OPERATOR_REAL cvright;
                        if(nativeRefinement)cvright=ugkwp::mixtureHeatCapacity(y,right.T,s.gasSpecies.thermo);
                        else
                        {
                            bool evaluated=false;
                            const auto caloric=ugkwp::evaluateMixtureHeatCapacity(y,right.T,s.gasSpecies.thermo,cvright,&evaluated);
                            if(caloric!=ugkwp::ThermoStatus::Success){gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
                            if(!evaluated){ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidStorage);return false;}
                        }
                        rightSound=sqrt((cvright+Rright)/cvright*Rright*right.T);
                    }
                    leftRiemann={left.rho,left.ux,left.uy,left.uz,left.p};
                    rightRiemann={right.rho,right.ux,right.uy,right.uz,right.p};
                }
                if(moving)
                {
                    const bool leftShift=ugkwp::shiftGasStateToFaceFrame(leftRiemann,leftConserved,meshFrame,leftRiemann,leftConserved);
                    const bool rightShift=(!nativeRefinement || leftShift)
                        &&ugkwp::shiftGasStateToFaceFrame(rightRiemann,rightConserved,meshFrame,rightRiemann,rightConserved);
                    if(!leftShift || !rightShift)
                    {gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry);return false;}
                }
                if(s.gasFluxScheme==2)
                    result=ugkpriemann::hllKurganovFluxUnitNormal
                    (leftRiemann,leftConserved,leftSound,rightRiemann,rightConserved,rightSound,nx,ny,nz,false);
                else
                    result=ugkpriemann::rusanovTadmorFluxUnitNormal
                    (leftRiemann,leftConserved,leftSound,rightRiemann,rightConserved,rightSound,nx,ny,nz,false);
                if(!result.valid)
                {
                    using namespace ugkp::diagnostics;
                    int component=-1;double value=result.maxSignalSpeed;
                    for(int k=0;k<5;++k)if(!finiteDevice(result.flux[k])){component=k;value=result.flux[k];break;}
                    if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2601,26,0,EntityKind::Face,f,component,value));
                    ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
                    return false;
                }
                const GPU_OPERATOR_REAL unLeft=leftRiemann.ux*nx+leftRiemann.uy*ny+leftRiemann.uz*nz;
                const GPU_OPERATOR_REAL unRight=rightRiemann.ux*nx+rightRiemann.uy*ny+rightRiemann.uz*nz;
                for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
                {
                    const GPU_OPERATOR_REAL l=leftSpecies[k];
                    const GPU_OPERATOR_REAL r=rightSpecies[k];
                    s.gasSpecies.flux[k*s.nFaces+f]=area*(result.evaluatedScheme==ugkpriemann::Scheme::HllKurganov
                        ?ugkpriemann::hllConservativeComponentFlux(unLeft*l,unRight*r,l,r,unLeft,unRight,leftSound,rightSound)
                        :ugkpriemann::rusanovConservativeComponentFlux(unLeft*l,unRight*r,l,r,result.maxSignalSpeed));
                }
                if(moving && !ugkwp::restoreGasFluxFromFaceFrame(result,meshFrame,result))
                {gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry);return false;}
                int massCarrier=Ns-1;
                // Redirect closure only when the last species is absent on
                // both sides. One-sided inventory can carry physical inflow.
                if(leftSpecies[massCarrier]==GPU_OPERATOR_R(0.0) && rightSpecies[massCarrier]==GPU_OPERATOR_R(0.0))
                    for(int k=0;k<Ns;++k)
                        if(leftSpecies[k]+rightSpecies[k]>leftSpecies[massCarrier]+rightSpecies[massCarrier])massCarrier=k;
                GPU_OPERATOR_REAL speciesSum=GPU_OPERATOR_R(0.0);
                for(int k=0;k<Ns;++k)if(k!=massCarrier)speciesSum+=s.gasSpecies.flux[k*s.nFaces+f];
                s.gasSpecies.flux[massCarrier*s.nFaces+f]=area*result.flux[0]-speciesSum;
                mixtureEvaluated=true;
            }
        }
        if(moving && !mixtureEvaluated)
        {
            auto leftConserved=ugkpriemann::conservative(leftRiemann,s.gammaGas);
            auto rightConserved=ugkpriemann::conservative(rightRiemann,s.gammaGas);
            const bool leftShift=ugkwp::shiftGasStateToFaceFrame(leftRiemann,leftConserved,meshFrame,leftRiemann,leftConserved);
            const bool rightShift=(!nativeRefinement || leftShift)
                &&ugkwp::shiftGasStateToFaceFrame(rightRiemann,rightConserved,meshFrame,rightRiemann,rightConserved);
            if(!leftShift || !rightShift)
            {
                numericalSuccess=false;
                if(!gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry) && nativeRefinement)asm("trap;");
                return false;
            }
        }
        if (mixtureEvaluated) {}
        else if (scheme == ugkpriemann::Scheme::SLAU2_2)
        {
            const int gradientNeighbour = nei >= 0 ? nei : own;
            result = ugkpriemann::slau22FluxUnitNormal
            (
                leftRiemann,
                rightRiemann,
                ugkpriemann::DensityGradient
                {
                    s.gradRhoX[own],
                    s.gradRhoY[own],
                    s.gradRhoZ[own]
                },
                ugkpriemann::DensityGradient
                {
                    s.gradRhoX[gradientNeighbour],
                    s.gradRhoY[gradientNeighbour],
                    s.gradRhoZ[gradientNeighbour]
                },
                nx,
                ny,
                nz,
                s.gammaGas,
                s.rhoMin,
                s.rhoMin*s.Rgas*s.TgasMin
            );
        }
        else
        {
            result = ugkpriemann::fluxUnitArea
            (
                leftRiemann,
                rightRiemann,
                nx,
                ny,
                nz,
                s.gammaGas,
                scheme,
                s.rhoMin,
                s.rhoMin*s.Rgas*s.TgasMin,
                hllcAdcOmega,nativeRefinement
            );
        }
        if (!result.valid)
        {
            using namespace ugkp::diagnostics;
            int component=-1;double value=result.maxSignalSpeed;
            for(int k=0;k<5;++k)if(!finiteDevice(result.flux[k])){component=k;value=result.flux[k];break;}
            if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,
                10043,26,0,EntityKind::Face,f,component,value));
            ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
            return false;
        }
        if(moving && !mixtureEvaluated && !ugkwp::restoreGasFluxFromFaceFrame(result,meshFrame,result))
        {
            numericalSuccess=false;
            if(!gasRecordEvaluatedFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry) && nativeRefinement)asm("trap;");
            return false;
        }
        massFlux = result.flux[0];
        if constexpr (MassOnly)
        {
            massFluxArea = massFlux*area;
            return numericalSuccess;
        }
        momentumFluxX = result.flux[1];
        momentumFluxY = result.flux[2];
        momentumFluxZ = result.flux[3];
        energyFlux = result.flux[4];
        if (s.gasReconstruction == 2 && nei >= 0 && !ugkwp::mixtureGasActive(s))
        {
                                                                      
                                                                           
                                                                         
                                                                        
                                                    
                                                                           
                                                                       
                                                                          
                                                                           
                                                                     
            const GPU_OPERATOR_REAL ownerVelocitySquared =
                s.Ux[own]*s.Ux[own]
              + s.Uy[own]*s.Uy[own]
              + s.Uz[own]*s.Uz[own];
            const GPU_OPERATOR_REAL neighbourVelocitySquared =
                s.Ux[nei]*s.Ux[nei]
              + s.Uy[nei]*s.Uy[nei]
              + s.Uz[nei]*s.Uz[nei];
            const GPU_OPERATOR_REAL ownerThermalEnthalpy = s.gasCp*s.Tgas[own];
            const GPU_OPERATOR_REAL neighbourThermalEnthalpy = s.gasCp*s.Tgas[nei];
            const GPU_OPERATOR_REAL ownerKineticEnergy = GPU_OPERATOR_R(0.5)*ownerVelocitySquared;
            const GPU_OPERATOR_REAL neighbourKineticEnergy =
                GPU_OPERATOR_R(0.5)*neighbourVelocitySquared;
            const ugkpinterpolation::Vector3
                ownerThermalEnthalpyGradient
            {
                s.gasCp*s.gradTX[own],
                s.gasCp*s.gradTY[own],
                s.gasCp*s.gradTZ[own]
            };
            const ugkpinterpolation::Vector3
                neighbourThermalEnthalpyGradient
            {
                s.gasCp*s.gradTX[nei],
                s.gasCp*s.gradTY[nei],
                s.gasCp*s.gradTZ[nei]
            };
            const ugkpinterpolation::Vector3 ownerKineticEnergyGradient
            {
                s.Ux[own]*s.gradUxX[own]
              + s.Uy[own]*s.gradUyX[own]
              + s.Uz[own]*s.gradUzX[own],
                s.Ux[own]*s.gradUxY[own]
              + s.Uy[own]*s.gradUyY[own]
              + s.Uz[own]*s.gradUzY[own],
                s.Ux[own]*s.gradUxZ[own]
              + s.Uy[own]*s.gradUyZ[own]
              + s.Uz[own]*s.gradUzZ[own]
            };
            const ugkpinterpolation::Vector3
                neighbourKineticEnergyGradient
            {
                s.Ux[nei]*s.gradUxX[nei]
              + s.Uy[nei]*s.gradUyX[nei]
              + s.Uz[nei]*s.gradUzX[nei],
                s.Ux[nei]*s.gradUxY[nei]
              + s.Uy[nei]*s.gradUyY[nei]
              + s.Uz[nei]*s.gradUzY[nei],
                s.Ux[nei]*s.gradUxZ[nei]
              + s.Uy[nei]*s.gradUyZ[nei]
              + s.Uz[nei]*s.gradUzZ[nei]
            };
            const ugkpinterpolation::Vector3 centreToCentre
            {
                mappedNeiCx - s.Cx[own],
                mappedNeiCy - s.Cy[own],
                mappedNeiCz - s.Cz[own]
            };
            const GPU_OPERATOR_REAL ownerWeight =
                clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
            const GPU_OPERATOR_REAL faceThermalEnthalpy =
                ugkpinterpolation::limitedLinearFaceValue
                (
                    ownerThermalEnthalpy,
                    neighbourThermalEnthalpy,
                    ownerThermalEnthalpyGradient,
                    neighbourThermalEnthalpyGradient,
                    centreToCentre,
                    ownerWeight,
                    massFlux,
                    GPU_OPERATOR_R(1.0)
                );
            const GPU_OPERATOR_REAL faceKineticEnergy =
                ugkpinterpolation::limitedLinearFaceValue
                (
                    ownerKineticEnergy,
                    neighbourKineticEnergy,
                    ownerKineticEnergyGradient,
                    neighbourKineticEnergyGradient,
                    centreToCentre,
                    ownerWeight,
                    massFlux,
                    GPU_OPERATOR_R(1.0)
                );
            const bool ownerIsUpwind = massFlux >= GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL upwindSpecificEnergy =
                (ownerIsUpwind
                  ? ownerThermalEnthalpy
                  : neighbourThermalEnthalpy)
              + (ownerIsUpwind
                  ? ownerKineticEnergy
                  : neighbourKineticEnergy);
            const GPU_OPERATOR_REAL limitedSpecificEnergy =
                faceThermalEnthalpy + faceKineticEnergy;
            energyFlux =
                ugkpinterpolation::limitedLinearRiemannEnergyFlux
                (
                    energyFlux,
                    massFlux,
                    upwindSpecificEnergy,
                    limitedSpecificEnergy
                );
        }
    }

                                                                       
    // A fixed-temperature slip wall conducts heat while remaining stress-free.
    if (boundaryKind != 1 || s.riemannBoundaryTFix[f] != 0)
    {
                                                                
                                                                          
                                                                     
                                                    
        GPU_OPERATOR_REAL gradUxX = s.gradUxX[own];
        GPU_OPERATOR_REAL gradUxY = s.gradUxY[own];
        GPU_OPERATOR_REAL gradUxZ = s.gradUxZ[own];
        GPU_OPERATOR_REAL gradUyX = s.gradUyX[own];
        GPU_OPERATOR_REAL gradUyY = s.gradUyY[own];
        GPU_OPERATOR_REAL gradUyZ = s.gradUyZ[own];
        GPU_OPERATOR_REAL gradUzX = s.gradUzX[own];
        GPU_OPERATOR_REAL gradUzY = s.gradUzY[own];
        GPU_OPERATOR_REAL gradUzZ = s.gradUzZ[own];
        GPU_OPERATOR_REAL gradTX = s.gradTX[own];
        GPU_OPERATOR_REAL gradTY = s.gradTY[own];
        GPU_OPERATOR_REAL gradTZ = s.gradTZ[own];
        const ugkptransport::Vector3 unitNormal{nx, ny, nz};
        ugkptransport::Vector3 compactSnGradU
        {
            gradUxX*nx + gradUxY*ny + gradUxZ*nz,
            gradUyX*nx + gradUyY*ny + gradUyZ*nz,
            gradUzX*nx + gradUzY*ny + gradUzZ*nz
        };
        GPU_OPERATOR_REAL normalTemperatureGradient =
            gradTX*nx + gradTY*ny + gradTZ*nz;

        if (nei >= 0)
        {
            const GPU_OPERATOR_REAL ownerWeight =
                clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
            const ugkptransport::SnGradGeometry snGradGeometry =
                ugkptransport::makeInternalSnGradGeometry
                (
                    ugkptransport::Vector3
                    {
                        s.Cx[own], s.Cy[own], s.Cz[own]
                    },
                    ugkptransport::Vector3
                    {
                        mappedNeiCx, mappedNeiCy, mappedNeiCz
                    },
                    unitNormal
                );
            const ugkptransport::Vector3 gradUx =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUxX[nei],
                        s.gradUxY[nei],
                        s.gradUxZ[nei]
                    },
                    ownerWeight
                );
            const ugkptransport::Vector3 gradUy =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUyX[nei],
                        s.gradUyY[nei],
                        s.gradUyZ[nei]
                    },
                    ownerWeight
                );
            const ugkptransport::Vector3 gradUz =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUzX[nei],
                        s.gradUzY[nei],
                        s.gradUzZ[nei]
                    },
                    ownerWeight
                );
            compactSnGradU.x =
                ugkptransport::correctedSnGrad
                (
                    s.Ux[own],
                    s.Ux[nei],
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUxX[nei],
                        s.gradUxY[nei],
                        s.gradUxZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            compactSnGradU.y =
                ugkptransport::correctedSnGrad
                (
                    s.Uy[own],
                    s.Uy[nei],
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUyX[nei],
                        s.gradUyY[nei],
                        s.gradUyZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            compactSnGradU.z =
                ugkptransport::correctedSnGrad
                (
                    s.Uz[own],
                    s.Uz[nei],
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUzX[nei],
                        s.gradUzY[nei],
                        s.gradUzZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            normalTemperatureGradient =
                ugkptransport::correctedSnGrad
                (
                    s.Tgas[own],
                    s.Tgas[nei],
                    ugkptransport::Vector3
                    {
                        gradTX, gradTY, gradTZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradTX[nei],
                        s.gradTY[nei],
                        s.gradTZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            gradUxX = gradUx.x;
            gradUxY = gradUx.y;
            gradUxZ = gradUx.z;
            gradUyX = gradUy.x;
            gradUyY = gradUy.y;
            gradUyZ = gradUy.z;
            gradUzX = gradUz.x;
            gradUzY = gradUz.y;
            gradUzZ = gradUz.z;
        }
        else
        {
            const bool velocityFixed = boundaryKind == 2
              || useRiemannBoundaryVelocity(s, f, left);
            const GPU_OPERATOR_REAL boundaryUx = boundaryKind == 2
              ? right.ux : s.riemannBoundaryUx[f];
            const GPU_OPERATOR_REAL boundaryUy = boundaryKind == 2
              ? right.uy : s.riemannBoundaryUy[f];
            const GPU_OPERATOR_REAL boundaryUz = boundaryKind == 2
              ? right.uz : s.riemannBoundaryUz[f];
            const GPU_OPERATOR_REAL currentNormalGradUx =
                gradUxX*nx + gradUxY*ny + gradUxZ*nz;
            const GPU_OPERATOR_REAL currentNormalGradUy =
                gradUyX*nx + gradUyY*ny + gradUyZ*nz;
            const GPU_OPERATOR_REAL currentNormalGradUz =
                gradUzX*nx + gradUzY*ny + gradUzZ*nz;
            const GPU_OPERATOR_REAL targetNormalGradUx = velocityFixed
              ? (boundaryUx - s.Ux[own])*s.deltaCoeffs[f] : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL targetNormalGradUy = velocityFixed
              ? (boundaryUy - s.Uy[own])*s.deltaCoeffs[f] : GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL targetNormalGradUz = velocityFixed
              ? (boundaryUz - s.Uz[own])*s.deltaCoeffs[f] : GPU_OPERATOR_R(0.0);
            // Component rows store grad(U_i). Replace only their normal
            // derivative, as gaussGrad::correctBoundaryConditions does, so
            // transpose and deviatoric traction use the same boundary snGrad.
            gradUxX += nx*(targetNormalGradUx - currentNormalGradUx);
            gradUxY += ny*(targetNormalGradUx - currentNormalGradUx);
            gradUxZ += nz*(targetNormalGradUx - currentNormalGradUx);
            gradUyX += nx*(targetNormalGradUy - currentNormalGradUy);
            gradUyY += ny*(targetNormalGradUy - currentNormalGradUy);
            gradUyZ += nz*(targetNormalGradUy - currentNormalGradUy);
            gradUzX += nx*(targetNormalGradUz - currentNormalGradUz);
            gradUzY += ny*(targetNormalGradUz - currentNormalGradUz);
            gradUzZ += nz*(targetNormalGradUz - currentNormalGradUz);
            compactSnGradU =
                ugkptransport::Vector3
                {
                    targetNormalGradUx,
                    targetNormalGradUy,
                    targetNormalGradUz
                };
            const GPU_OPERATOR_REAL targetNormalGradT =
                s.riemannBoundaryTFix[f] != 0
              ? (s.riemannBoundaryT[f] - s.Tgas[own])
               *s.deltaCoeffs[f]
              : GPU_OPERATOR_R(0.0);
            normalTemperatureGradient = targetNormalGradT;
        }

        const GPU_OPERATOR_REAL rhoFace =
            GPU_OPERATOR_R(0.5)*(left.rho + right.rho);
        GPU_OPERATOR_REAL muTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL kTurbulent = GPU_OPERATOR_R(0.0);
        GPU_OPERATOR_REAL directWallHeatFlux = GPU_OPERATOR_R(0.0);
        int directWallHeatFluxActive = 0;
        if constexpr (IncludeTurbulence)
        {
            bool subgridEvaluated=false;
            gasFaceSubgridTransportProperties
            (
                s,
                f,
                own,
                nei,
                boundaryKind,
                rhoFace,
                muTurbulent,
                kTurbulent,
                directWallHeatFlux,
                directWallHeatFluxActive,nativeRefinement,&subgridEvaluated
            );
            if(!subgridEvaluated)return false;
            if(ugkwp::mixtureGasActive(s) && ugkwp::gasFaceFailure(s,f)!=0)
            {return false;}
        }
        const GPU_OPERATOR_REAL muEffective = boundaryKind == 1
          ? GPU_OPERATOR_R(0.0) : s.gasMu + muTurbulent;
        GPU_OPERATOR_REAL kMolecular=molecularGasConductivity(s);
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
            if(ugkwp::mixtureGasActive(s) && !ugkwp::gasHasDirectConductivity(s))
            {
                const GPU_OPERATOR_REAL w=nei>=0?clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0)):GPU_OPERATOR_R(1.0);
                const GPU_OPERATOR_REAL cp=w*s.gasSpecies.heatCapacity[own]
                    +(GPU_OPERATOR_R(1.0)-w)*s.gasSpecies.heatCapacity[nei>=0?nei:own];
                kMolecular=s.gasMu*cp/s.gasPrClamped;
            }
        const GPU_OPERATOR_REAL kEffective = kMolecular + kTurbulent;
        const GPU_OPERATOR_REAL wallThermalAreaFraction =
            GPU_GAS_WALL_EXPOSURE(s, f, nei, boundaryKind);
        if (muEffective > GPU_OPERATOR_R(0.0) || kEffective > GPU_OPERATOR_R(0.0))
        {
            const ugkptransport::Vector3 traction =
                ugkptransport::openFoamNewtonianTraction
                (
                    muEffective,
                    unitNormal,
                    compactSnGradU,
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    }
                );

            GPU_OPERATOR_REAL faceUx = GPU_OPERATOR_R(0.5)*(left.ux + right.ux);
            GPU_OPERATOR_REAL faceUy = GPU_OPERATOR_R(0.5)*(left.uy + right.uy);
            GPU_OPERATOR_REAL faceUz = GPU_OPERATOR_R(0.5)*(left.uz + right.uz);
            if (nei >= 0)
            {
                const GPU_OPERATOR_REAL ownerWeight =
                    clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
                faceUx =
                    ownerWeight*left.ux + (GPU_OPERATOR_R(1.0) - ownerWeight)*right.ux;
                faceUy =
                    ownerWeight*left.uy + (GPU_OPERATOR_R(1.0) - ownerWeight)*right.uy;
                faceUz =
                    ownerWeight*left.uz + (GPU_OPERATOR_R(1.0) - ownerWeight)*right.uz;
            }
            if
            (
                boundaryKind == 2
             || (nei < 0 && useRiemannBoundaryVelocity(s, f, left))
            )
            {
                faceUx = right.ux;
                faceUy = right.uy;
                faceUz = right.uz;
            }
            momentumFluxX -= traction.x;
            momentumFluxY -= traction.y;
            momentumFluxZ -= traction.z;
            energyFlux -=
                traction.x*faceUx
              + traction.y*faceUy
              + traction.z*faceUz;
            if (directWallHeatFluxActive != 0)
            {
                energyFlux +=
                    wallThermalAreaFraction*directWallHeatFlux;
            }
            else
            {
                energyFlux -=
                    wallThermalAreaFraction
                   *kEffective*normalTemperatureGradient;
            }
        }
    }

    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s) && (s.gasSpecies.diffusivity || (IncludeTurbulence && s.turbulenceModel!=0))
            && (nei>=0 || (boundaryKind==0 && s.gasSpecies.compositionBoundaryFixed[f]!=0)))
        {
            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            GPU_OPERATOR_REAL Y[Ns], gradArea[Ns], diffusion[Ns], enthalpyFlux=GPU_OPERATOR_R(0.0);
            const GPU_OPERATOR_REAL w=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
            for(int k=0;k<Ns;++k)
            {
                const int o=k*s.nCells+own;
                const GPU_OPERATOR_REAL yo=s.gasSpecies.rho[o]/s.rho[own];
                if(nei>=0)
                {
                    const int n=k*s.nCells+nei;
                    const GPU_OPERATOR_REAL yn=s.gasSpecies.rho[n]/s.rho[nei];
                    Y[k]=w*yo+(GPU_OPERATOR_R(1.0)-w)*yn;
                    const auto geometry=ugkptransport::makeInternalSnGradGeometry
                    (
                        ugkptransport::Vector3{s.Cx[own],s.Cy[own],s.Cz[own]},
                        ugkptransport::Vector3{mappedNeiCx,mappedNeiCy,mappedNeiCz},
                        ugkptransport::Vector3{nx,ny,nz}
                    );
                    gradArea[k]=area*ugkptransport::correctedSnGrad
                    (
                        yo,yn,ugkptransport::Vector3{s.gasSpecies.gradX[o],s.gasSpecies.gradY[o],s.gasSpecies.gradZ[o]},
                        ugkptransport::Vector3{s.gasSpecies.gradX[n],s.gasSpecies.gradY[n],s.gasSpecies.gradZ[n]},w,geometry
                    );
                }
                else
                {
                    Y[k]=s.gasSpecies.boundaryMassFraction[k*s.nFaces+f];
                    gradArea[k]=area*(Y[k]-yo)*s.deltaCoeffs[f];
                }
            }
            bool diffusionEvaluated=false;
            if(!ugkwp::correctedSpeciesDiffusion
            (
                Y,gradArea,nei>=0?w*left.rho+(GPU_OPERATOR_R(1.0)-w)*right.rho:right.rho,
                s.gasSpecies.diffusivity,IncludeTurbulence && s.turbulenceModel!=0
                    ?(nei>=0?w*s.nut[own]+(GPU_OPERATOR_R(1.0)-w)*s.nut[nei]:s.nut[own]):GPU_OPERATOR_R(0.0),
                s.gasSpecies.turbulentSchmidt,
                nei>=0?w*left.T+(GPU_OPERATOR_R(1.0)-w)*right.T:right.T,s.gasSpecies.thermo,diffusion,enthalpyFlux,nativeRefinement,&diffusionEvaluated
            ))
            {ugkwp::gasRecordFaceFailure(s,f,!nativeRefinement&&!diffusionEvaluated?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);return false;}
            for(int k=0;k<Ns;++k)s.gasSpecies.flux[k*s.nFaces+f]+=diffusion[k];
            energyFlux+=enthalpyFlux/area;
        }
    }
    const GPU_OPERATOR_REAL completeFlux[5]={massFlux*area,momentumFluxX*area,
        momentumFluxY*area,momentumFluxZ*area,energyFlux*area};
    for(int k=0;k<5;++k)if(!finiteDevice(completeFlux[k]))
    {
        using namespace ugkp::diagnostics;
        if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::NonFinite,10044,26,0,
            EntityKind::Face,f,k,completeFlux[k]));
        ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::NonFiniteState);
        return false;
    }
    massFluxArea = massFlux*area;
    momFluxXArea = momentumFluxX*area;
    momFluxYArea = momentumFluxY*area;
    momFluxZArea = momentumFluxZ*area;
    energyFluxArea = energyFlux*area;
    return numericalSuccess;
}
