#pragma once
// One face-walk algorithm; wall state and coordinate precision are adapters.
__device__ void trackOneParticleLocalFaceWalk(DeviceState& s, const int i, const GPU_OPERATOR_TIME dt)
{
    if (i >= s.particleCapacity || s.pStatus[i] == 0)
    {
        return;
    }

    int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

#if GPU_OPERATOR_THERMAL
    if (s.pStuck[i] != 0)
    {
        s.pux[i] = GPU_OPERATOR_R(0.0);
        s.puy[i] = GPU_OPERATOR_R(0.0);
        s.puz[i] = GPU_OPERATOR_R(0.0);
        if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
        {
            s.puxOld[i] = GPU_OPERATOR_R(0.0);
            s.puyOld[i] = GPU_OPERATOR_R(0.0);
            s.puzOld[i] = GPU_OPERATOR_R(0.0);
        }
        return;
    }

#endif
    GPU_OPERATOR_REAL x = s.px[i];
    GPU_OPERATOR_REAL y = s.py[i];
    GPU_OPERATOR_REAL z = s.pz[i];
    GPU_OPERATOR_REAL vxStep = GPU_OPERATOR_R(0.5)*(s.puxOld[i] + s.pux[i]);
    GPU_OPERATOR_REAL vyStep = GPU_OPERATOR_R(0.5)*(s.puyOld[i] + s.puy[i]);
    GPU_OPERATOR_REAL vzStep = GPU_OPERATOR_R(0.5)*(s.puzOld[i] + s.puz[i]);
    GPU_OPERATOR_TIME remainingDt = dt;
    bool inside = pointInsideCell(s, c, x, y, z);
    for (int hop = 0; hop < s.maxFaceWalkHops && remainingDt > GPU_OPERATOR_R(0.0); ++hop)
    {
        const GPU_OPERATOR_REAL x1 = x + vxStep*remainingDt;
        const GPU_OPERATOR_REAL y1 = y + vyStep*remainingDt;
        const GPU_OPERATOR_REAL z1 = z + vzStep*remainingDt;
        inside = pointInsideCell(s, c, x1, y1, z1);
        if (inside)
        {
            x = x1;
            y = y1;
            z = z1;
            remainingDt = GPU_OPERATOR_R(0.0);
            break;
        }

        GPU_OPERATOR_REAL hitT = GPU_OPERATOR_R(1.0);
        const int plane = firstSegmentIntersection(s, c, x, y, z, x1, y1, z1, hitT);
        if (plane < 0)
        {
            s.pStatus[i] = 0;
            return;
        }

        const int kind = s.cellFaceKind[plane];
        const int next = s.cellFaceNeighbor[plane];
        const GPU_OPERATOR_REAL xHit = x + hitT*(x1 - x);
        const GPU_OPERATOR_REAL yHit = y + hitT*(y1 - y);
        const GPU_OPERATOR_REAL zHit = z + hitT*(z1 - z);
        remainingDt *= clampRange(GPU_OPERATOR_R(1.0) - hitT, GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        if (kind == 0 && next >= 0 && next < s.nCells)
        {
            c = next;
            x = xHit;
            y = yHit;
            z = zHit;
        }
        else if (kind == GPU_PERIODIC_FACE_KIND && next >= 0 && next < s.nCells)
        {
            const int faceI = s.cellFaceId[plane];
            if (!isPeriodicFace(s, faceI))
            {
                s.pStatus[i] = 0;
                return;
            }
            const GPU_OPERATOR_REAL eps =
                GPU_OPERATOR_R(1.0e-9)*clampMin(s.cellLength[next], GPU_OPERATOR_R(1.0e-12));
            c = next;
            x = GPU_PERIODIC_COORDINATE(xHit, s.facePeriodicDx[faceI], eps, s.planeNx[plane]);
            y = GPU_PERIODIC_COORDINATE(yHit, s.facePeriodicDy[faceI], eps, s.planeNy[plane]);
            z = GPU_PERIODIC_COORDINATE(zHit, s.facePeriodicDz[faceI], eps, s.planeNz[plane]);
        }
        else if (kind == 1 || kind == 2)
        {
            const GPU_OPERATOR_REAL nx = s.planeNx[plane];
            const GPU_OPERATOR_REAL ny = s.planeNy[plane];
            const GPU_OPERATOR_REAL nz = s.planeNz[plane];
            const GPU_OPERATOR_REAL un = s.pux[i]*nx + s.puy[i]*ny + s.puz[i]*nz;
            const GPU_OPERATOR_REAL unStep = vxStep*nx + vyStep*ny + vzStep*nz;
            const GPU_OPERATOR_REAL restitution = s.cellFaceRestitution[plane];
            s.pux[i] -= (GPU_OPERATOR_R(1.0) + restitution)*un*nx;
            s.puy[i] -= (GPU_OPERATOR_R(1.0) + restitution)*un*ny;
            s.puz[i] -= (GPU_OPERATOR_R(1.0) + restitution)*un*nz;
            vxStep -= (GPU_OPERATOR_R(1.0) + restitution)*unStep*nx;
            vyStep -= (GPU_OPERATOR_R(1.0) + restitution)*unStep*ny;
            vzStep -= (GPU_OPERATOR_R(1.0) + restitution)*unStep*nz;

            const GPU_OPERATOR_REAL eps = GPU_OPERATOR_R(1.0e-9)*clampMin(s.cellLength[c], GPU_OPERATOR_R(1.0e-12));
            x = GPU_WALL_COORDINATE(xHit, eps, nx);
            y = GPU_WALL_COORDINATE(yHit, eps, ny);
            z = GPU_WALL_COORDINATE(zHit, eps, nz);
        }
#if GPU_OPERATOR_THERMAL
        else if (kind == 5)
        {
            const GPU_OPERATOR_REAL nx = s.planeNx[plane];
            const GPU_OPERATOR_REAL ny = s.planeNy[plane];
            const GPU_OPERATOR_REAL nz = s.planeNz[plane];
            const int globalFaceId = s.cellFaceId[plane];
            if
            (
                s.particleStuckModelConfigured == 0
             || globalFaceId < 0
             || globalFaceId >= s.nFaces
             || s.particleStuckCandidateMask[globalFaceId] == 0
            )
            {
                asm("trap;");
            }
            const unsigned char wallInteractionType =
                s.particleStuckCandidateMask[globalFaceId];
            const GPU_OPERATOR_REAL un =
                s.pux[i]*nx + s.puy[i]*ny + s.puz[i]*nz;
            const GPU_OPERATOR_REAL normalSpeed = fabs
            (
                (s.pux[i] - s.gasBoundaryUx[globalFaceId])*nx
              + (s.puy[i] - s.gasBoundaryUy[globalFaceId])*ny
              + (s.puz[i] - s.gasBoundaryUz[globalFaceId])*nz
            );
            const Foam::gpuThermal::SommerfeldImpact impact =
                Foam::gpuThermal::evaluateSommerfeldImpact
                (
                    s.pT[i],
                    normalSpeed,
                    s.pd[i],
                    s.sommerfeldThreshold
                );
            if (!impact.valid)
            {
                asm("trap;");
            }
            const Foam::gpuThermal::FiniteWallContactImpact finiteImpact =
                Foam::gpuThermal::evaluateFiniteWallContactImpact
                (
                    s.pT[i],
                    s.pd[i],
                    normalSpeed,
                    s.particleWallContactAngleCosine
                );
            if
            (
                !finiteImpact.valid
             || finiteImpact.contactDurationS > static_cast<GPU_OPERATOR_REAL>(FLT_MAX)
             || finiteImpact.maximumAreaM2 > static_cast<GPU_OPERATOR_REAL>(FLT_MAX)
             || finiteImpact.peakTimeFraction > static_cast<GPU_OPERATOR_REAL>(FLT_MAX)
            )
            {
                asm("trap;");
            }
            const GPU_OPERATOR_REAL eps = GPU_OPERATOR_R(1.0e-9)*clampMin(s.cellLength[c], GPU_OPERATOR_R(1.0e-12));
            x = GPU_WALL_COORDINATE(xHit, eps, nx);
            y = GPU_WALL_COORDINATE(yHit, eps, ny);
            z = GPU_WALL_COORDINATE(zHit, eps, nz);
            const GPU_OPERATOR_REAL restitution = s.cellFaceRestitution[plane];
            s.puxOld[i] = s.pux[i] - (GPU_OPERATOR_R(1.0) + restitution)*un*nx;
            s.puyOld[i] = s.puy[i] - (GPU_OPERATOR_R(1.0) + restitution)*un*ny;
            s.puzOld[i] = s.puz[i] - (GPU_OPERATOR_R(1.0) + restitution)*un*nz;
            s.pux[i] = GPU_OPERATOR_R(0.0);
            s.puy[i] = GPU_OPERATOR_R(0.0);
            s.puz[i] = GPU_OPERATOR_R(0.0);
            s.pStuck[i] =
                wallInteractionType
             == Foam::gpuThermal::particleWallReboundContact
              ? Foam::gpuThermal::particleWallTransientRebound
              :
                (
                    impact.deposit
                  ? Foam::gpuThermal::particleWallTransientDeposit
                  : Foam::gpuThermal::particleWallTransientRebound
                );
            s.pStuckFaceId[i] = globalFaceId;
            s.pTheta[i] = GPU_OPERATOR_R(0.0);
            GPU_RESET_CONTACT_AGE(s, i)
            s.pDepositionArea[i] = 0.0f;
            s.pContactDuration[i] =
                GPU_CONTACT_DURATION(finiteImpact.contactDurationS);
            s.pContactMaximumArea[i] =
                static_cast<float>(finiteImpact.maximumAreaM2);
            s.pContactPeakFraction[i] =
                GPU_CONTACT_PEAK(finiteImpact.peakTimeFraction);
            if
            (
                wallInteractionType
             == Foam::gpuThermal::particleWallSolidifyingDeposition
            )
            {
                initialiseColdWallParticleState(s, i, s.pT[i]);
                clearColdWall2DParticleState(s, i);
            }
            else if
            (
                wallInteractionType
             == Foam::gpuThermal::particleWallColdWall2D
            )
            {
                clearColdWallParticleState(s, i);
                initialiseColdWall2DParticleState(s, i, s.pT[i]);
            }
            else
            {
                clearColdWallParticleState(s, i);
                clearColdWall2DParticleState(s, i);
            }
            remainingDt = GPU_OPERATOR_R(0.0);
            break;
        }
#endif
        else
        {
            const GPU_OPERATOR_REAL exitMass =
                clampMin(finiteOr(s.pm[i], GPU_PARTICLE_EXIT_MASS_FALLBACK), GPU_OPERATOR_R(0.0));
            s.pStatus[i] = 0;
            return;
        }

        if (c < 0 || c >= s.nCells)
        {
            s.pStatus[i] = 0;
            return;
        }
    }

    inside = pointInsideCell(s, c, x, y, z);
    if (!inside || remainingDt > GPU_OPERATOR_R(0.0))
    {
        s.pStatus[i] = 0;
        return;
    }
    s.px[i] = x;
    s.py[i] = y;
    s.pz[i] = z;
    s.pCellId[i] = c;
}
