// Shared input/material/contact preparation; included inside each precision wrapper.
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            asm("trap;");
        }
        const GpuReal dPart = clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_R(1.0e-12)
        );
        const GpuReal tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const Foam::gpuThermal::AluminaLiquidProperties material =
            Foam::gpuThermal::liquidAluminaProperties(tpOld);
        const GpuReal physicalVolume =
            (Foam::gpuThermal::finiteContactPi/GPU_R(6.0))*dPart*dPart*dPart;
        const GpuReal physicalMass = material.densityKgM3*physicalVolume;
        const GpuReal parcelMultiplicity = s.pm[i]/physicalMass;
        GpuReal gasConductanceWK = GPU_R(0.0);
        const GpuReal gasTemperatureK = clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            GPU_R(1.0e30)
        );
        if
        (
            lane == 0
         && s.solveParticleTemperature != 0
         && s.particleGasHeatTransferModelId != 0
        )
        {
            const GpuReal rhoG = clampMin
            (
                finiteOr(s.couplingRhoOld[c], s.rhoMin),
                s.rhoMin
            );
            const GpuReal ugx = finiteOr(s.couplingUxOld[c], GPU_R(0.0));
            const GpuReal ugy = finiteOr(s.couplingUyOld[c], GPU_R(0.0));
            const GpuReal ugz = finiteOr(s.couplingUzOld[c], GPU_R(0.0));
            const GpuReal relMag = sqrt(ugx*ugx + ugy*ugy + ugz*ugz);
            const GpuReal re = rhoG*dPart*relMag/clampMin(s.gasMu, GPU_R(1.0e-30));
            const GpuReal nu = GPU_R(2.0)
              + GPU_R(0.6)*sqrt(clampMin(re, GPU_R(0.0)))*s.gasPrOneThird;
            const GpuReal particleCp = particleSpecificHeatDevice(tpOld);
            const GpuReal rate = GPU_R(6.0)*nu*molecularGasConductivity(s)
              /(s.rhoSolid*particleCp*dPart*dPart + GPU_TINY(1.0e-300));
            gasConductanceWK = rate*physicalMass*particleCp;
        }
        gasConductanceWK = __shfl_sync(mask, gasConductanceWK, 0, 8);
        const unsigned char wallState = s.pStuck[i];
        const bool finiteContact =
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit;
        GpuReal maximumArea = static_cast<GpuReal>(s.pContactMaximumArea[i]);
        GpuReal intrinsicArea = GPU_R(0.0);
        GpuTime duration = static_cast<GpuTime>(s.pContactDuration[i]);
        GpuReal peakTimeFraction =
            static_cast<GpuReal>(s.pContactPeakFraction[i]);
        GpuTime activeDt = dt;
        if (finiteContact)
        {
            const GpuReal damageArea =
                static_cast<GpuReal>(s.pDepositionArea[i]);
            // Validate mechanical contact metadata before either thermal path
            // can publish. The later finalizer must not be the first rejection
            // after a gas-only update has already changed the particle.
            if
            (
                !(duration > 0.0) || duration > DBL_MAX
             || !(peakTimeFraction > GPU_R(0.0))
             || !(peakTimeFraction < GPU_R(1.0))
             || !Foam::gpuThermal::finiteColdWallValue(damageArea)
             || damageArea < GPU_R(0.0)
            )
            {
                asm("trap;");
            }
            const GpuTime age0 = clampRange
            (
                finiteOr(s.pContactAge[i], GPU_R(0.0)),
                GPU_R(0.0),
                duration
            );
            activeDt = clampRange(dt, GPU_R(0.0), duration - age0);
            const GpuTime ageMid = age0 + GPU_R(0.5)*activeDt;
            const GpuReal kinematicAreaMid = maximumArea
              *Foam::gpuThermal::normalizedKinematicArea
                (
                    ageMid/duration,
                    peakTimeFraction
                );
            if (!Foam::gpuThermal::finiteColdWallValue(kinematicAreaMid))
            {
                asm("trap;");
            }
            intrinsicArea = clampMin
            (
                fmax
                (
                    kinematicAreaMid,
                    static_cast<GpuReal>(s.pColdFrozenArea[i])
                ) - damageArea,
                GPU_R(0.0)
            );
        }
        else
        {
            intrinsicArea = static_cast<GpuReal>(s.pDepositionArea[i]);
        }
        if (!Foam::gpuThermal::finiteColdWallValue(intrinsicArea))
        {
            asm("trap;");
        }
        if (!(activeDt > GPU_R(0.0)) || !(intrinsicArea > GPU_R(0.0)))
        {
            // A finite contact may retract below its accumulated damage, or
            // expire inside this step. Gas exchange still spans the full dt;
            // the later generic finalizer alone performs the state transition.
            if
            (
                (!finiteContact && !(intrinsicArea > GPU_R(0.0)))
             || !(parcelMultiplicity > GPU_R(0.0))
             || !Foam::gpuThermal::finiteColdWallValue(parcelMultiplicity)
             || !(maximumArea > GPU_R(0.0))
             || !Foam::gpuThermal::finiteColdWallValue(maximumArea)
             || static_cast<GpuReal>(s.pColdFrozenArea[i]) > maximumArea
            )
            {
                asm("trap;");
            }
            GpuReal gasOnlyMeanTemperature = GPU_R(0.0);
            const bool gasOnlyValid = advanceColdWall1DGasOnlyGroup
            (
                s, i, lane, mask, physicalMass, gasTemperatureK, gasConductanceWK,
                dt, activeDt, gasOnlyMeanTemperature
            );
            if (!gasOnlyValid)
            {
                asm("trap;");
            }
            if (lane == 0)
            {
                s.pT[i] = gasOnlyMeanTemperature;
            }
            continue;
        }
        GpuReal meanTemperature = GPU_R(0.0);
        GpuReal frozenArea = GPU_R(0.0);
        GpuReal wallEnergy = GPU_R(0.0);
        const GpuReal efficiency = finiteContact
          ? s.particleWallReflectionHeatTransferEfficiency
          : s.particleWallDepositionHeatTransferEfficiency;
        const GpuReal wallEffusivity =
            s.particleWallEffusivityByFace != nullptr
          ? s.particleWallEffusivityByFace[faceI]
          : sqrt
            (
                s.particleWallDensityKgM3
               *s.particleWallSpecificHeatJkgK
               *s.particleWallConductivityWmK
            );
