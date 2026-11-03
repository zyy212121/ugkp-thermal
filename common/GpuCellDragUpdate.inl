#include "GpuMomentDragRelaxation.H"

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
        s, c,
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

    const GPU_OPERATOR_REAL betaDt=dt*beta;
    const GPU_OPERATOR_REAL particleMomentum[3]={momSX,momSY,momSZ};
    const GPU_OPERATOR_REAL gasVelocity[3]={s.rhoUx[c]/rhoG,s.rhoUy[c]/rhoG,s.rhoUz[c]/rhoG};
    const GPU_OPERATOR_REAL eMinGas=s.Rgas*s.TgasMin/clampMin(s.gammaGas-GPU_OPERATOR_REAL(1.0),OfSmall);
    const auto relaxed=ugkwp::relaxGasFromParticleMoments
    (
        mg,ms,epsG,epsGsafe,GPU_OPERATOR_REAL(s.rhoE[c]),
        mg*eMinGas,gasVelocity,particleMomentum,enerS0,betaDt
    );
    s.thetaDragAlpha[c]=relaxed.alphaInternal;
    s.rho[c]=rhoG;
    s.rhoUx[c]=rhoG*relaxed.velocity[0];
    s.rhoUy[c]=rhoG*relaxed.velocity[1];
    s.rhoUz[c]=rhoG*relaxed.velocity[2];
    s.rhoE[c]=Energy(relaxed.intrinsicGasEnergy);
