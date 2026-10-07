#pragma once
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GasPrimDevice makeGasPrimDevice
(
    const GPU_OPERATOR_REAL rho,
    const GPU_OPERATOR_REAL ux,
    const GPU_OPERATOR_REAL uy,
    const GPU_OPERATOR_REAL uz,
    const GPU_OPERATOR_REAL p,
    const GPU_OPERATOR_REAL Rgas,
    const GPU_OPERATOR_REAL rhoMin,
    const GPU_OPERATOR_REAL Tmin
)
{
    GasPrimDevice g;
    g.rho = clampMin(finiteOr(rho, rhoMin), rhoMin);
    g.ux = finiteOr(ux, GPU_OPERATOR_R(0.0));
    g.uy = finiteOr(uy, GPU_OPERATOR_R(0.0));
    g.uz = finiteOr(uz, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL pMinimum =
        g.rho*clampMin(Rgas, OfSmall)*clampMin(Tmin, OfSmall);
    g.p = clampMin(finiteOr(p, pMinimum), pMinimum);
    g.T = g.p/clampMin(g.rho*Rgas, OfSmall);
    return g;
}

template<class GasState>
__device__ GasPrimDevice gasCellPrimitive(const GasState& s,const int c)
{
    if(ugkwp::mixtureGasActive(s))
        return GasPrimDevice{s.rho[c],s.Ux[c],s.Uy[c],s.Uz[c],s.p[c],s.Tgas[c]};
    return makeGasPrimDevice(s.rho[c],s.Ux[c],s.Uy[c],s.Uz[c],s.p[c],s.Rgas,s.rhoMin,s.TgasMin);
}

template<class GasState>
__device__ bool gasBoundaryMeshNormalSpeed(const GasState& s,const int f,GPU_OPERATOR_REAL& normalSpeed)
{
    normalSpeed=GPU_OPERATOR_R(0.0);
    if constexpr (ugkwp::GasGeometryCapability<GasState>::value)
        if(ugkwp::gasMovingGeometry(s))
        {
            ugkwp::GasFaceFrame<GPU_OPERATOR_REAL> frame;
            if(!ugkwp::gasGeometryFaceFrame(s,f,s.gasGeometry.interval,frame))
            {
                if(!ugkwp::gasRecordFaceFailure(s,f,ugkwp::GasTransportCode::InvalidGeometry))asm("trap;");
                return false;
            }
            normalSpeed=frame.meshVolumeRate/frame.area;
        }
    return true;
}

template<class GasState>
__device__ bool useRiemannBoundaryVelocity
(
    const GasState& s,
    const int f,
    const GasPrimDevice& ownerState
)
{
    const int mode = s.riemannBoundaryUFix[f];
    if (mode == 1)
    {
        return true;
    }
    if (mode != 2)
    {
        return false;
    }

                                                                           
    const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
    GPU_OPERATOR_REAL outwardVelocity =
        (ownerState.ux*s.Sfx[f]
       + ownerState.uy*s.Sfy[f]
       + ownerState.uz*s.Sfz[f])/area;
    if(ugkwp::gasMovingGeometry(s))
    {
        GPU_OPERATOR_REAL meshNormal;
        if(!gasBoundaryMeshNormalSpeed(s,f,meshNormal))return false;
        outwardVelocity-=meshNormal;
    }
    return outwardVelocity < GPU_OPERATOR_R(0.0);
}

template<class GasState>
__device__ GasPrimDevice riemannBoundaryState
(
    const GasState& s,
    const int f,
    const GasPrimDevice& ownerState,
    const GPU_OPERATOR_REAL* extrapolatedMassFraction = nullptr
)
{
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if(ugkwp::mixtureGasActive(s))
        {
            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            GPU_OPERATOR_REAL Y[Ns];
            const int own=s.faceOwner[f];
            const bool fixedY=s.gasSpecies.compositionBoundaryFixed[f]!=0;
            for(int k=0;k<Ns;++k)
                Y[k]=fixedY?s.gasSpecies.boundaryMassFraction[k*s.nFaces+f]
                    :(extrapolatedMassFraction?extrapolatedMassFraction[k]
                        :s.gasSpecies.rho[k*s.nCells+own]/s.rho[own]);
            const GPU_OPERATOR_REAL R=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);
            if(s.riemannBoundaryUFix[f]==3 || s.riemannBoundaryPWave[f]!=0 || s.nScheduledInletFaces>0)
            {s.gasSpecies.faceStatus[f]=int(ugkwp::GasTransportCode::UnsupportedConfiguration);return ownerState;}
            GasPrimDevice out=ownerState;
            const bool rf=s.riemannBoundaryRhoFix[f]!=0,pf=s.riemannBoundaryPFix[f]!=0,tf=s.riemannBoundaryTFix[f]!=0;
            if(rf)out.rho=s.riemannBoundaryRho[f];
            if(pf)out.p=s.riemannBoundaryP[f];
            if(tf)out.T=s.riemannBoundaryT[f];
            if(rf&&pf&&!tf)out.T=out.p/(out.rho*R);
            else if(rf&&!pf)out.p=out.rho*R*out.T;
            GPU_OPERATOR_REAL density=out.rho;
            const auto status=ugkwp::mixtureDensityFromPressureTemperature
                (out.p,out.T,Y,s.gasSpecies.thermo,density,s.gasSpecies.densityClosureTolerance);
            if(status!=ugkwp::ThermoStatus::Success || (rf&&pf&&tf
                && fabs(density-out.rho)>s.gasSpecies.densityClosureTolerance*density))
            {s.gasSpecies.faceStatus[f]=int(ugkwp::GasTransportCode::InvalidThermodynamics);return ownerState;}
            out.rho=density;
            if(useRiemannBoundaryVelocity(s,f,ownerState))
            {out.ux=s.riemannBoundaryUx[f];out.uy=s.riemannBoundaryUy[f];out.uz=s.riemannBoundaryUz[f];}
            if(!ugkwp::gasFinite(out.ux)||!ugkwp::gasFinite(out.uy)||!ugkwp::gasFinite(out.uz))
            {s.gasSpecies.faceStatus[f]=int(ugkwp::GasTransportCode::NonFiniteState);return ownerState;}
            return out;
        }
    }
    if (s.riemannBoundaryUFix[f] == 3)
    {
        const GPU_OPERATOR_REAL totalTemperature = clampMin
        (
            finiteOr(s.riemannBoundaryT[f], ownerState.T),
            s.TgasMin
        );
        const GPU_OPERATOR_REAL totalPressure = clampMin
        (
            finiteOr(s.riemannBoundaryP[f], ownerState.p),
            s.rhoMin*s.Rgas*totalTemperature
        );
        const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
        const ugkpboundary::Primitive boundary =
            ugkpboundary::totalConditionInletState
            (
                ugkpboundary::Primitive
                {
                    ownerState.rho,
                    ownerState.ux,
                    ownerState.uy,
                    ownerState.uz,
                    ownerState.p,
                    ownerState.T
                },
                s.Sfx[f]/area,
                s.Sfy[f]/area,
                s.Sfz[f]/area,
                totalPressure,
                totalTemperature,
                s.gammaGas,
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
        return makeGasPrimDevice
        (
            boundary.rho,
            boundary.ux,
            boundary.uy,
            boundary.uz,
            boundary.p,
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }

    const bool velocityFixed =
        useRiemannBoundaryVelocity(s, f, ownerState);
    const bool rhoFixed = s.riemannBoundaryRhoFix[f] != 0;
    const bool pFixed =
        s.riemannBoundaryPFix[f] != 0
     || s.riemannBoundaryPWave[f] != 0;
    const bool TFixed = s.riemannBoundaryTFix[f] != 0;

    GPU_OPERATOR_REAL rho = ownerState.rho;
    GPU_OPERATOR_REAL p = ownerState.p;
    GPU_OPERATOR_REAL T = ownerState.T;
    if (rhoFixed)
    {
        rho = clampMin
        (
            finiteOr(s.riemannBoundaryRho[f], ownerState.rho),
            s.rhoMin
        );
    }
    if (pFixed)
    {
        p = clampMin
        (
            finiteOr(s.riemannBoundaryP[f], ownerState.p),
            rho*s.Rgas*s.TgasMin
        );
    }
    if (TFixed)
    {
        T = clampMin
        (
            finiteOr(s.riemannBoundaryT[f], ownerState.T),
            s.TgasMin
        );
    }

                                                                             
                                                                               
                                                         
    if (pFixed && TFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }
    else if (rhoFixed && TFixed)
    {
        p = rho*s.Rgas*T;
    }
    else if (rhoFixed && pFixed)
    {
        T = p/clampMin(rho*s.Rgas, OfSmall);
    }
    else if (pFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }
    else if (rhoFixed)
    {
        p = rho*s.Rgas*T;
    }
    else if (TFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }

    const GPU_OPERATOR_REAL ux = velocityFixed
      ? s.riemannBoundaryUx[f] : ownerState.ux;
    const GPU_OPERATOR_REAL uy = velocityFixed
      ? s.riemannBoundaryUy[f] : ownerState.uy;
    const GPU_OPERATOR_REAL uz = velocityFixed
      ? s.riemannBoundaryUz[f] : ownerState.uz;
    return makeGasPrimDevice
    (
        rho, ux, uy, uz, p, s.Rgas, s.rhoMin, s.TgasMin
    );
}

template<class GasState>
__device__ bool isPeriodicFace(const GasState& s, const int f)
{
    return
        f >= s.nInternalFaces
     && f < s.nFaces
     && s.facePeriodicPair[f] >= s.nInternalFaces
     && s.facePeriodicPair[f] < s.nFaces;
}

template<class GasState>
__device__ int coupledFaceNeighbour(const GasState& s, const int f)
{
    return (f < s.nInternalFaces || isPeriodicFace(s, f))
      ? s.faceNeighbour[f] : -1;
}

template<class GasState>
__device__ void periodicMappedCellCentre
(
    const GasState& s,
    const int f,
    const int c,
    GPU_OPERATOR_REAL& x,
    GPU_OPERATOR_REAL& y,
    GPU_OPERATOR_REAL& z
)
{
    x = s.Cx[c];
    y = s.Cy[c];
    z = s.Cz[c];
    if (isPeriodicFace(s, f) && c == s.faceNeighbour[f])
    {
        x += s.facePeriodicDx[f];
        y += s.facePeriodicDy[f];
        z += s.facePeriodicDz[f];
    }
}

template<class GasState>
__device__ GasPrimDevice riemannExteriorStateForFace
(
    const GasState& s,
    const int f,
    const GasPrimDevice& ownerFaceState
)
{
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        return makeGasPrimDevice
        (
            s.rho[nei],
            s.Ux[nei],
            s.Uy[nei],
            s.Uz[nei],
            s.p[nei],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }
    return riemannBoundaryState(s, f, ownerFaceState);
}
