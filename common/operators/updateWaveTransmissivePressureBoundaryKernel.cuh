#pragma once
#include "gasTransport/GasPseudoTime.H"
#include "gasTransport/GasBoundaryThermodynamics.H"
template<class GasState>
__global__ void updateWaveTransmissivePressureBoundaryKernel
(
    GasState* sp,
    const GPU_OPERATOR_TIME stepDt
)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    GasState& s = *sp;
    const int f = s.nInternalFaces + blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces || s.gasBoundaryPWave[f] == 0)
    {
        return;
    }

    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        return;
    }

    const GPU_OPERATOR_TIME dt = ugkwp::gasCellTimeStep(s, own, stepDt);

    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], GPU_OPERATOR_TINY(1.0e-300));
    GPU_OPERATOR_REAL boundaryUx = s.gasBoundaryUx[f];
    GPU_OPERATOR_REAL boundaryUy = s.gasBoundaryUy[f];
    GPU_OPERATOR_REAL boundaryUz = s.gasBoundaryUz[f];
    if (s.gasBoundaryUFix[f] == 2)
    {
        const GPU_OPERATOR_REAL ownerOutwardVelocity =
            s.Ux[own]*s.Sfx[f]
          + s.Uy[own]*s.Sfy[f]
          + s.Uz[own]*s.Sfz[f];
        if (ownerOutwardVelocity >= GPU_OPERATOR_R(0.0))
        {
            boundaryUx = s.Ux[own];
            boundaryUy = s.Uy[own];
            boundaryUz = s.Uz[own];
        }
    }
    const GPU_OPERATOR_REAL phip =
        boundaryUx*s.Sfx[f]
      + boundaryUy*s.Sfy[f]
      + boundaryUz*s.Sfz[f];
    GPU_OPERATOR_REAL psi =
        s.gasBoundaryRho[f]
       /clampMin(s.gasBoundaryP[f], OfSmall);
    // Frozen mixtures use the boundary EOS, not owner rho divided by a
    // separately evolved boundary pressure. Gamma remains the user-specified
    // waveTransmissive parameter, as in the single-gas boundary law.
    if constexpr(ugkwp::GasStateTraits<GasState>::speciesCount>0)if(ugkwp::mixtureGasActive(s)){
        constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
        GPU_OPERATOR_REAL y[Ns];
        for(int k=0;k<Ns;++k)y[k]=s.gasSpecies.compositionBoundaryFixed[f]
            ?s.gasSpecies.boundaryMassFraction[k*s.nFaces+f]:s.gasSpecies.rho[k*s.nCells+own]/s.rho[own];
        const auto R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
        const auto thermo=ugkwp::mixtureBoundaryThermodynamics(
            ugkwp::BoundaryThermodynamics<GPU_OPERATOR_REAL>{s.rho[own],s.p[own],s.gasBoundaryT[f]},
            ugkwp::BoundaryThermodynamics<GPU_OPERATOR_REAL>{s.gasBoundaryRho[f],s.gasBoundaryP[f],s.gasBoundaryT[f]},
            s.riemannBoundaryRhoFix[f]!=0,true,s.riemannBoundaryTFix[f]!=0,R);
        const GPU_OPERATOR_REAL rt=R*thermo.temperature;
        if(!ugkwp::gasFinite(rt)||!(rt>0)){
            ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidThermodynamics);return;
        }
        psi=GPU_OPERATOR_R(1)/rt;
    }
    GPU_OPERATOR_REAL w =
        phip/area
      + sqrt(clampMin(s.gasBoundaryPWaveGamma[f]/clampMin(psi, OfSmall), GPU_OPERATOR_R(0.0)));
    w = w > GPU_OPERATOR_R(0.0) ? w : GPU_OPERATOR_R(0.0);

    const GPU_OPERATOR_REAL alpha = w*dt*s.deltaCoeffs[f];
    const GPU_OPERATOR_REAL lInf = s.gasBoundaryPWaveLInf[f];
    const bool hasRelaxation = lInf > GPU_OPERATOR_TINY(1.0e-300);
    const GPU_OPERATOR_REAL k = hasRelaxation ? w*dt/lInf : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL oldBoundaryP = s.gasBoundaryP[f];
    const GPU_OPERATOR_REAL refValue = hasRelaxation
      ? (oldBoundaryP + k*s.gasBoundaryPWaveFieldInf[f])/(GPU_OPERATOR_R(1.0) + k)
      : oldBoundaryP;
    const GPU_OPERATOR_REAL valueFraction = hasRelaxation
      ? (GPU_OPERATOR_R(1.0) + k)/(GPU_OPERATOR_R(1.0) + alpha + k)
      : GPU_OPERATOR_R(1.0)/(GPU_OPERATOR_R(1.0) + alpha);
    const GPU_OPERATOR_REAL boundaryP =
        valueFraction*refValue
      + (GPU_OPERATOR_R(1.0) - valueFraction)*s.p[own];

    const GPU_OPERATOR_REAL updatedPressure =
        clampMin(finiteOr(boundaryP, s.p[own]), OfVSmall);
    s.gasBoundaryP[f] = updatedPressure;
    s.riemannBoundaryP[f] = updatedPressure;
}
