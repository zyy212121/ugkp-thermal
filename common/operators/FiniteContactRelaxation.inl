// Finite-contact heat, reaction and state transition; native contact-age storage adapter.
// Inputs: s, i, dt, finiteContact, coldWallContact and contact-age adapter macros.
// Outputs: particle heat/contact state, wall energy, saved velocity and age.
// Caller supplies the finiteContact branch and owns the following deposited-state else.
        const int faceI = s.pStuckFaceId[i];
        const GPU_OPERATOR_TIME duration = static_cast<GPU_OPERATOR_TIME>(s.pContactDuration[i]);
        const GPU_OPERATOR_REAL maximumArea = static_cast<GPU_OPERATOR_REAL>(s.pContactMaximumArea[i]);
        const GPU_OPERATOR_REAL peakTimeFraction =
            static_cast<GPU_OPERATOR_REAL>(s.pContactPeakFraction[i]);
        const GPU_OPERATOR_REAL damageArea = static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
        const GPU_OPERATOR_TIME age0 = clampRange(finiteOr(GPU_CONTACT_AGE(s, i), GPU_CONTACT_TIME_ZERO), GPU_CONTACT_TIME_ZERO, duration);
        if
        (
            faceI < 0 || faceI >= s.nFaces
         || s.particleStuckCandidateMask[faceI] == 0
         || !(duration > GPU_OPERATOR_R(0.0)) || !(maximumArea > GPU_OPERATOR_R(0.0))
         || !(peakTimeFraction > GPU_OPERATOR_R(0.0)) || !(peakTimeFraction < GPU_OPERATOR_R(1.0))
         || damageArea < GPU_OPERATOR_R(0.0)
        )
        {
            asm("trap;");
        }
        const GPU_OPERATOR_TIME activeDt = clampRange(dt, GPU_OPERATOR_R(0.0), duration - age0);
        const GPU_OPERATOR_TIME age1 = age0 + activeDt;
        if
        (
            activeDt > GPU_OPERATOR_R(0.0)
         && s.particleWallHeatTransferEnabled != 0
         && !coldWallContact
        )
        {
            const GPU_OPERATOR_REAL thetaMid = (age0 + GPU_OPERATOR_R(0.5)*activeDt)/duration;
            const GPU_OPERATOR_REAL contactAreaMid = fmax
            (
                maximumArea
               *Foam::gpuThermal::normalizedKinematicArea
                (
                    thetaMid, peakTimeFraction
                )
              - damageArea,
                GPU_OPERATOR_R(0.0)
            );
            const GPU_OPERATOR_REAL wallEffusivity =
                s.particleWallEffusivityByFace != nullptr
              ? s.particleWallEffusivityByFace[faceI]
              : sqrt
                (
                    s.particleWallDensityKgM3
                   *s.particleWallSpecificHeatJkgK
                   *s.particleWallConductivityWmK
                );
            const Foam::gpuThermal::AluminaLiquidProperties material =
                Foam::gpuThermal::liquidAluminaProperties(s.pT[i]);
            const GPU_OPERATOR_REAL physicalMass =
                (Foam::gpuThermal::finiteContactPi/GPU_OPERATOR_R(6.0))
               *material.densityKgM3*s.pd[i]*s.pd[i]*s.pd[i];
            const GPU_OPERATOR_REAL conductanceTimeIntegral =
                Foam::gpuThermal::finiteContactWallConductanceTimeIntegral
                (
                    maximumArea,
                    contactAreaMid,
                    duration,
                    peakTimeFraction,
                    age0,
                    activeDt,
                    s.particleWallReflectionHeatTransferEfficiency
                   *s.particleWallContactAreaScale[faceI],
                    s.coldWallSolidificationParameters
                     .interfaceResistanceM2KW,
                    GPU_OPERATOR_R(0.0),
                    wallEffusivity,
                    s.coldWallSolidificationParameters
                     .wallTransientResistance != 0
                );
            const Foam::gpuThermal::ParticleWallContactResult result =
                Foam::gpuThermal::lumpedParticleWallInterfaceContact
                (
                    s.pT[i],
                    s.gasBoundaryT[faceI],
                    physicalMass,
                    s.pm[i],
                    conductanceTimeIntegral
                );
            if
            (
                contactAreaMid < GPU_OPERATOR_R(0.0) || !(wallEffusivity > GPU_OPERATOR_R(0.0))
             || !material.valid || !(physicalMass > GPU_OPERATOR_R(0.0))
             || !(conductanceTimeIntegral >= GPU_OPERATOR_R(0.0)) || !result.valid
            )
            {
                asm("trap;");
            }
            s.pT[i] = result.particleTemperatureK;
            atomicAddParticleWallEnergyByFace
            (
                s,
                s.particleWallReflectedEnergy,
                faceI,
                result.wallEnergyJ
            );
        }
        GPU_CONTACT_AGE(s, i) = age1;

        bool detach = false;
        bool enterLongDeposit = false;
        GPU_OPERATOR_REAL longDepositArea = GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL theta1 = age1/duration;
        const GPU_OPERATOR_REAL kinematicArea =
            maximumArea
           *Foam::gpuThermal::normalizedKinematicArea
            (
                theta1, peakTimeFraction
            );
        const GPU_OPERATOR_REAL frozenArea = coldWallContact
          ? static_cast<GPU_OPERATOR_REAL>(s.pColdFrozenArea[i])
          : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL effectiveContactArea =
            fmax(kinematicArea, frozenArea) - damageArea;
        if
        (
            wallStateAtStepStart
         == Foam::gpuThermal::particleWallTransientRebound
        )
        {
            detach = !(effectiveContactArea > GPU_OPERATOR_R(0.0));
            enterLongDeposit =
                coldWallContact && !detach && !(kinematicArea > GPU_OPERATOR_R(0.0));
            longDepositArea = effectiveContactArea;
        }
        else
        {
            const Foam::gpuThermal::CapillaryDetachmentState capillary =
                Foam::gpuThermal::evaluateCapillaryDetachmentState
                (
                    s.pT[i], s.pd[i], s.particleWallAdhesionEnergyScale,
                    s.particleWallContactAngleCosine
                );
            const GPU_OPERATOR_REAL equilibriumArea =
                capillary.equilibriumContactAreaM2;
            const GPU_OPERATOR_REAL targetContactArea = fmin(equilibriumArea, maximumArea);
            if (!coldWallContact)
            {
                asm("trap;");
            }
            longDepositArea =
                fmax(targetContactArea, frozenArea) - damageArea;
            detach = !(effectiveContactArea > GPU_OPERATOR_R(0.0));
            enterLongDeposit =
                !detach
             && theta1 >= peakTimeFraction
             &&
                (
                    kinematicArea <= targetContactArea
                 || !(kinematicArea > GPU_OPERATOR_R(0.0))
                );
            if (!capillary.valid)
            {
                asm("trap;");
            }
        }

        if (detach)
        {
            s.pux[i] = s.puxOld[i];
            s.puy[i] = s.puyOld[i];
            s.puz[i] = s.puzOld[i];
            s.pStuck[i] = Foam::gpuThermal::particleWallMobile;
            s.pStuckFaceId[i] = -1;
            s.pTheta[i] = GPU_OPERATOR_R(0.0);
            GPU_RESET_CONTACT_AGE(s, i)
            s.pDepositionArea[i] = 0.0f;
            s.pContactDuration[i] = 0.0f;
            s.pContactMaximumArea[i] = 0.0f;
            s.pContactPeakFraction[i] = 0.0f;
            clearColdWallParticleState(s, i);
            clearColdWall2DParticleState(s, i);
        }
        else if (enterLongDeposit)
        {
            s.pStuck[i] = Foam::gpuThermal::particleWallDeposited;
            s.pTheta[i] = GPU_OPERATOR_R(0.0);
            GPU_RESET_CONTACT_AGE(s, i)
            s.pDepositionArea[i] = static_cast<float>(longDepositArea);
            s.puxOld[i] = GPU_OPERATOR_R(0.0);
            s.puyOld[i] = GPU_OPERATOR_R(0.0);
            s.puzOld[i] = GPU_OPERATOR_R(0.0);
        }
