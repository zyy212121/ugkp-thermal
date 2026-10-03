#pragma once
#ifndef GPU_GAS_WALL_EXPOSURE
#define GPU_GAS_WALL_EXPOSURE(s, f, neighbour, kind) GPU_OPERATOR_R(1.0)
#endif
template<bool IncludeTurbulence>
__device__ bool computeRiemannGasFaceFluxDevice
(
    const DeviceState& s,
    const int f,
    GPU_OPERATOR_REAL& massFluxArea,
    GPU_OPERATOR_REAL& momFluxXArea,
    GPU_OPERATOR_REAL& momFluxYArea,
    GPU_OPERATOR_REAL& momFluxZArea,
    GPU_OPERATOR_REAL& energyFluxArea
)
{
    massFluxArea = GPU_OPERATOR_R(0.0);
    momFluxXArea = GPU_OPERATOR_R(0.0);
    momFluxYArea = GPU_OPERATOR_R(0.0);
    momFluxZArea = GPU_OPERATOR_R(0.0);
    energyFluxArea = GPU_OPERATOR_R(0.0);

    if (f < 0 || f >= s.nFaces)
    {
        return false;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        return false;
    }

    const int nei = coupledFaceNeighbour(s, f);
    const int boundaryKind = nei >= 0 ? 0 : s.riemannBoundaryKind[f];
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
    if (boundaryKind == 3 || boundaryKind == 4)
    {
        return false;
    }

    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
    const GPU_OPERATOR_REAL nx = s.Sfx[f]/area;
    const GPU_OPERATOR_REAL ny = s.Sfy[f]/area;
    const GPU_OPERATOR_REAL nz = s.Sfz[f]/area;
    GasPrimDevice left = reconstructGasCellToFace(s, own, f);
    GasPrimDevice right = left;

    GPU_OPERATOR_REAL massFlux = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxX = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxY = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momentumFluxZ = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL energyFlux = GPU_OPERATOR_R(0.0);

    if (boundaryKind == 1)
    {
                                                                          
                                                                            
        momentumFluxX = left.p*nx;
        momentumFluxY = left.p*ny;
        momentumFluxZ = left.p*nz;
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
    }
    else
    {
        right = nei >= 0
          ? reconstructGasCellToFace(s, nei, f)
          : riemannExteriorStateForFace(s, f, left);
        if (s.gasReconstruction == 1 && nei >= 0)
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
        const ugkpriemann::Primitive leftRiemann
        {
            left.rho, left.ux, left.uy, left.uz, left.p
        };
        const ugkpriemann::Primitive rightRiemann
        {
            right.rho, right.ux, right.uy, right.uz, right.p
        };
        ugkpriemann::Scheme scheme;
        if (!ugkpriemann::schemeFromCreateCode(s.gasFluxScheme, scheme))
        {
            asm("trap;");
            return false;
        }
        GPU_OPERATOR_REAL hllcAdcOmega = GPU_OPERATOR_R(1.0);
        if (scheme == ugkpriemann::Scheme::HLLC_ADC)
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
        if (scheme == ugkpriemann::Scheme::SLAU2_2)
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
                hllcAdcOmega
            );
        }
        if (!result.valid)
        {
            asm("trap;");
            return false;
        }
        massFlux = result.flux[0];
        momentumFluxX = result.flux[1];
        momentumFluxY = result.flux[2];
        momentumFluxZ = result.flux[3];
        energyFlux = result.flux[4];
        if (s.gasReconstruction == 2 && nei >= 0)
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

                                                                       
    if (boundaryKind != 1)
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
            (void)currentNormalGradUx;
            (void)currentNormalGradUy;
            (void)currentNormalGradUz;
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
                directWallHeatFluxActive
            );
        }
        const GPU_OPERATOR_REAL muEffective = s.gasMu + muTurbulent;
        const GPU_OPERATOR_REAL kEffective =
            molecularGasConductivity(s) + kTurbulent;
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

    massFluxArea = massFlux*area;
    momFluxXArea = momentumFluxX*area;
    momFluxYArea = momentumFluxY*area;
    momFluxZArea = momentumFluxZ*area;
    energyFluxArea = energyFlux*area;
    return true;
}
