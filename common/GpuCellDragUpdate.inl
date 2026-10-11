
    DeviceState& s = *sp;
    using Energy = ugkwp::GasEnergyScalar<DeviceState>;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    s.couplingRhoOld[c] =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    s.couplingUxOld[c] = finiteOr(s.Ux[c], GPU_OPERATOR_R(0.0));
    s.couplingUyOld[c] = finiteOr(s.Uy[c], GPU_OPERATOR_R(0.0));
    s.couplingUzOld[c] = finiteOr(s.Uz[c], GPU_OPERATOR_R(0.0));
    s.couplingTgasOld[c] =
        clampMin(finiteOr(s.Tgas[c], s.TgasMin), s.TgasMin);

    s.thetaDragAlpha[c] = GPU_OPERATOR_R(1.0);
    // A zero-time source may refresh coupling caches, never conserved state.
    if (dt == GPU_OPERATOR_TIME(0)) return;
    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));

    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const GPU_OPERATOR_REAL eps = solidEpsFromMomentDevice(s, c);
    const GPU_OPERATOR_REAL epsG = GPU_OPERATOR_R(1.0) - eps;
    const GPU_OPERATOR_REAL epsGsafe = clampMin(epsG, OfSmall);
    const GPU_OPERATOR_REAL rhoG = s.couplingRhoOld[c];
    const GPU_OPERATOR_REAL mg = epsG*rhoG;
    const GPU_OPERATOR_REAL ugx0 = s.couplingUxOld[c];
    const GPU_OPERATOR_REAL ugy0 = s.couplingUyOld[c];
    const GPU_OPERATOR_REAL ugz0 = s.couplingUzOld[c];

    const GPU_OPERATOR_REAL momSX = finiteOr(s.momRhoUPx[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL momSY = finiteOr(s.momRhoUPy[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL momSZ = finiteOr(s.momRhoUPz[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL enerS0 = clampMin(finiteOr(s.momRhoEP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));

    const GPU_OPERATOR_REAL ms = rhoP;
    if (!finiteDevice(ms) || ms < s.rhoMin)
    {
        return;
    }

    const GPU_OPERATOR_REAL usx0 = momSX/ms;
    const GPU_OPERATOR_REAL usy0 = momSY/ms;
    const GPU_OPERATOR_REAL usz0 = momSZ/ms;

    const GPU_OPERATOR_REAL dLocal =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/ms, s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );

    const GPU_OPERATOR_REAL urx = ugx0 - usx0;
    const GPU_OPERATOR_REAL ury = ugy0 - usy0;
    const GPU_OPERATOR_REAL urz = ugz0 - usz0;

    const GPU_OPERATOR_REAL urMag = sqrt(sqr3(urx, ury, urz));
    const GPU_OPERATOR_REAL beta = cellDragInverseTime<DragModel>
    (
        s,
        rhoG,
        epsG,
        dLocal,
        urMag,
        dragModel
    );

    if (!finiteDevice(beta) || beta <= OfSmall)
    {
        return;
    }

    const GPU_OPERATOR_REAL tauDragCell = clampMin(GPU_OPERATOR_R(1.0)/beta, OfSmall);

    const GPU_OPERATOR_REAL momGX0 = epsG*s.rhoUx[c];
    const GPU_OPERATOR_REAL momGY0 = epsG*s.rhoUy[c];
    const GPU_OPERATOR_REAL momGZ0 = epsG*s.rhoUz[c];
    const Energy enerG0 = Energy(epsG)*s.rhoE[c];

    GPU_OPERATOR_REAL momGX = momGX0;
    GPU_OPERATOR_REAL momGY = momGY0;
    GPU_OPERATOR_REAL momGZ = momGZ0;

                                                                              
                                                                                
                                                                              
                                                                            
    Energy enerG = enerG0;
    GPU_OPERATOR_REAL enerS = enerS0;

    if (!finiteDevice(mg) || mg < s.rhoMin)
    {
        return;
    }

    const GPU_OPERATOR_REAL ugx = momGX/mg;
    const GPU_OPERATOR_REAL ugy = momGY/mg;
    const GPU_OPERATOR_REAL ugz = momGZ/mg;

    const GPU_OPERATOR_REAL usx = momSX/ms;
    const GPU_OPERATOR_REAL usy = momSY/ms;
    const GPU_OPERATOR_REAL usz = momSZ/ms;

    const GPU_OPERATOR_REAL gm1 = clampMin(s.gammaGas - GPU_OPERATOR_R(1.0), OfSmall);
    const GPU_OPERATOR_REAL eMinGas = s.Rgas*s.TgasMin/gm1;

    const Energy trueGasKineticDensity = Energy(0.5)
        *(Energy(s.rhoUx[c])*Energy(s.rhoUx[c])+Energy(s.rhoUy[c])*Energy(s.rhoUy[c])+Energy(s.rhoUz[c])*Energy(s.rhoUz[c]))/Energy(rhoG);
    const Energy kgOld = std::is_same<Energy,double>::value
        ? Energy(epsG)*trueGasKineticDensity
        : Energy(0.5)*Energy(mg)*(Energy(ugx)*Energy(ugx)+Energy(ugy)*Energy(ugy)+Energy(ugz)*Energy(ugz));
    const GPU_OPERATOR_REAL ksOld = GPU_OPERATOR_R(0.5)*ms*sqr3(usx, usy, usz);

    Energy igOld = std::is_same<Energy,double>::value
        ? Energy(epsG)*(s.rhoE[c]-trueGasKineticDensity) : enerG-kgOld;
    GPU_OPERATOR_REAL isOld = enerS - ksOld;

    const GPU_OPERATOR_REAL igMin = mg*eMinGas;
    const GPU_OPERATOR_REAL isMin = GPU_OPERATOR_R(0.0);

    const bool repairGasInternal=!ugkwp::gasEnergyFinite(igOld) || igOld<igMin;
    if (repairGasInternal)
    {
        igOld = igMin;
    }

    if (!finiteDevice(isOld) || isOld < isMin)
    {
        isOld = isMin;
    }

    const GPU_OPERATOR_REAL invTauDragCell = GPU_OPERATOR_R(1.0)/clampMin(tauDragCell, OfSmall);

    const GPU_OPERATOR_REAL alphaI =
        clampRange
        (
            finiteOr(exp(-GPU_OPERATOR_R(2.0)*dt*invTauDragCell), GPU_OPERATOR_R(1.0)),
            GPU_OPERATOR_R(0.0),
            GPU_OPERATOR_R(1.0)
        );

                                                                   
                                                                    
                                                      
    s.thetaDragAlpha[c] = alphaI;

    GPU_OPERATOR_REAL isAfter = isOld*alphaI;
    if (isAfter < isMin)
    {
        isAfter = isMin;
    }

    const GPU_OPERATOR_REAL dI = isOld - isAfter;
    const Energy igAfter = igOld + dI;

    const GPU_OPERATOR_REAL wx = ugx - usx;
    const GPU_OPERATOR_REAL wy = ugy - usy;
    const GPU_OPERATOR_REAL wz = ugz - usz;

    // Strictly zero resolved exchange must not reconstruct rounded momenta or
    // phase-weighted energy. Preserve the existing repair for invalid inputs.
    if (!repairGasInternal && wx==GPU_OPERATOR_R(0) && wy==GPU_OPERATOR_R(0)
        && wz==GPU_OPERATOR_R(0) && dI==GPU_OPERATOR_R(0)) return;

    const GPU_OPERATOR_REAL kW = dt*invTauDragCell*(GPU_OPERATOR_R(1.0) + ms/(mg + OfSmall));
    const GPU_OPERATOR_REAL alphaW = exp(-kW);

    const GPU_OPERATOR_REAL wxNew = wx*alphaW;
    const GPU_OPERATOR_REAL wyNew = wy*alphaW;
    const GPU_OPERATOR_REAL wzNew = wz*alphaW;

    const GPU_OPERATOR_REAL pX = momGX + momSX;
    const GPU_OPERATOR_REAL pY = momGY + momSY;
    const GPU_OPERATOR_REAL pZ = momGZ + momSZ;

    const GPU_OPERATOR_REAL mTot = mg + ms;

    const GPU_OPERATOR_REAL usNewX = (pX - mg*wxNew)/mTot;
    const GPU_OPERATOR_REAL usNewY = (pY - mg*wyNew)/mTot;
    const GPU_OPERATOR_REAL usNewZ = (pZ - mg*wzNew)/mTot;

    const GPU_OPERATOR_REAL ugNewX = usNewX + wxNew;
    const GPU_OPERATOR_REAL ugNewY = usNewY + wyNew;
    const GPU_OPERATOR_REAL ugNewZ = usNewZ + wzNew;

    const Energy kgNew = Energy(0.5)*Energy(mg)*(Energy(ugNewX)*Energy(ugNewX)+Energy(ugNewY)*Energy(ugNewY)+Energy(ugNewZ)*Energy(ugNewZ));
    const GPU_OPERATOR_REAL ksNew = GPU_OPERATOR_R(0.5)*ms*sqr3(usNewX, usNewY, usNewZ);

    Energy diss = (kgOld + ksOld) - (kgNew + ksNew);
    if (!ugkwp::gasEnergyFinite(diss) || diss < GPU_OPERATOR_R(0.0))
    {
        diss = GPU_OPERATOR_R(0.0);
    }

    s.rho[c] = rhoG;
    s.rhoUx[c] = rhoG*ugNewX;
    s.rhoUy[c] = rhoG*ugNewY;
    s.rhoUz[c] = rhoG*ugNewZ;
    s.rhoE[c] = (kgNew + igAfter + diss)/Energy(epsGsafe);
