// Exports gas primitive-gradient kernels and the SST wall-distance/viscosity helper used by them.
#pragma once
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__global__ void computeGasPrimitiveGradientsKernel
(
    GasState* sp,
    const bool refreshSstCourantSensor = false
)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL grx = GPU_OPERATOR_R(0.0), gry = GPU_OPERATOR_R(0.0), grz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guxx = GPU_OPERATOR_R(0.0), guxy = GPU_OPERATOR_R(0.0), guxz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guyx = GPU_OPERATOR_R(0.0), guyy = GPU_OPERATOR_R(0.0), guyz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL guzx = GPU_OPERATOR_R(0.0), guzy = GPU_OPERATOR_R(0.0), guzz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gpx = GPU_OPERATOR_R(0.0), gpy = GPU_OPERATOR_R(0.0), gpz = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL gtx = GPU_OPERATOR_R(0.0), gty = GPU_OPERATOR_R(0.0), gtz = GPU_OPERATOR_R(0.0);
    const bool refreshSensor = refreshSstCourantSensor
      && s.sstConfigured != 0 && s.gasFluxScheme == 7;
    const GPU_OPERATOR_REAL centrePressure = refreshSensor
      ? clampMin(s.p[c], OfSmall) : GPU_OPERATOR_R(1.0);
    GPU_OPERATOR_REAL sensor = GPU_OPERATOR_R(1.0);
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        if (refreshSensor)
        {
            // Same pressure-ratio stencil as computeGasHllcAdcSensorKernel,
            // folded into this existing traversal only during Courant checks.
            GPU_OPERATOR_REAL otherPressure = centrePressure;
            if (f < s.nInternalFaces || isPeriodicFace(s, f))
            {
                const int other = oppositeCellAcrossFace
                (
                    c, s.faceOwner[f], s.faceNeighbour[f]
                );
                if (other >= 0 && other < s.nCells)
                {
                    otherPressure = clampMin(s.p[other], OfSmall);
                }
            }
            else if (s.riemannBoundaryKind[f] == 0)
            {
                const GasPrimDevice centre = makeGasPrimDevice
                (
                    s.rho[c], s.Ux[c], s.Uy[c], s.Uz[c], centrePressure,
                    s.Rgas, s.rhoMin, s.TgasMin
                );
                otherPressure = clampMin
                (
                    riemannBoundaryState(s, f, centre).p, OfSmall
                );
            }
            const GPU_OPERATOR_REAL ratio = clampRange
            (
                fmin(otherPressure/centrePressure, centrePressure/otherPressure),
                GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0)
            );
            sensor = fmin(sensor, ratio*ratio*ratio);
        }
        const GPU_OPERATOR_REAL sign = s.faceOwner[f] == c ? GPU_OPERATOR_R(1.0) : -GPU_OPERATOR_R(1.0);
        const GPU_OPERATOR_REAL sx = sign*s.Sfx[f];
        const GPU_OPERATOR_REAL sy = sign*s.Sfy[f];
        const GPU_OPERATOR_REAL sz = sign*s.Sfz[f];
        const GasPrimDevice qf =
            riemannFacePrimitiveForGradient(s, c, f);
        grx += qf.rho*sx; gry += qf.rho*sy; grz += qf.rho*sz;
        guxx += qf.ux*sx; guxy += qf.ux*sy; guxz += qf.ux*sz;
        guyx += qf.uy*sx; guyy += qf.uy*sy; guyz += qf.uy*sz;
        guzx += qf.uz*sx; guzy += qf.uz*sy; guzz += qf.uz*sz;
        gpx += qf.p*sx; gpy += qf.p*sy; gpz += qf.p*sz;
        gtx += qf.T*sx; gty += qf.T*sy; gtz += qf.T*sz;
    }
    if (refreshSensor)
    {
        s.gasHllcAdcSensor[c] = finiteDevice(sensor)
          ? clampRange(sensor, GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0))
          : GPU_OPERATOR_R(0.0);
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfSmall);
    s.gradRhoX[c] = grx*invV; s.gradRhoY[c] = gry*invV; s.gradRhoZ[c] = grz*invV;
    s.gradUxX[c] = guxx*invV; s.gradUxY[c] = guxy*invV; s.gradUxZ[c] = guxz*invV;
    s.gradUyX[c] = guyx*invV; s.gradUyY[c] = guyy*invV; s.gradUyZ[c] = guyz*invV;
    s.gradUzX[c] = guzx*invV; s.gradUzY[c] = guzy*invV; s.gradUzZ[c] = guzz*invV;
    s.gradPx[c] = gpx*invV; s.gradPy[c] = gpy*invV; s.gradPz[c] = gpz*invV;
    s.gradTX[c] = gtx*invV; s.gradTY[c] = gty*invV; s.gradTZ[c] = gtz*invV;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
    {
        if (ugkwp::mixtureGasActive(s))
            for(int k=0;k<ugkwp::GasStateTraits<GasState>::speciesCount;++k)
            {
                GPU_OPERATOR_REAL gx=GPU_OPERATOR_R(0.0),gy=GPU_OPERATOR_R(0.0),gz=GPU_OPERATOR_R(0.0);
                const GPU_OPERATOR_REAL centre=s.gasSpecies.rho[k*s.nCells+c]/s.rho[c];
                for(int i=0;i<count;++i)
                {
                    const int f=s.cellFaceId[start+i];
                    if(f<0||f>=s.nFaces)continue;
                    const int own=s.faceOwner[f],nei=coupledFaceNeighbour(s,f);
                    const GPU_OPERATOR_REAL sign=own==c?GPU_OPERATOR_R(1.0):-GPU_OPERATOR_R(1.0);
                    GPU_OPERATOR_REAL value=centre;
                    if(nei>=0)
                    {
                        const GPU_OPERATOR_REAL w=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0));
                        value=w*s.gasSpecies.rho[k*s.nCells+own]/s.rho[own]
                            +(GPU_OPERATOR_R(1.0)-w)*s.gasSpecies.rho[k*s.nCells+nei]/s.rho[nei];
                    }
                    else if(s.riemannBoundaryKind[f]==0 && s.gasSpecies.compositionBoundaryFixed[f]!=0)
                        value=s.gasSpecies.boundaryMassFraction[k*s.nFaces+f];
                    gx+=sign*s.Sfx[f]*value;gy+=sign*s.Sfy[f]*value;gz+=sign*s.Sfz[f]*value;
                }
                s.gasSpecies.gradX[k*s.nCells+c]=gx*invV;
                s.gasSpecies.gradY[k*s.nCells+c]=gy*invV;
                s.gasSpecies.gradZ[k*s.nCells+c]=gz*invV;
                if(k==ugkwp::GasStateTraits<GasState>::speciesCount-1)
                {
                    GPU_OPERATOR_REAL sumX=GPU_OPERATOR_R(0.0),sumY=GPU_OPERATOR_R(0.0),sumZ=GPU_OPERATOR_R(0.0);
                    for(int j=0;j<k;++j)
                    {sumX+=s.gasSpecies.gradX[j*s.nCells+c];sumY+=s.gasSpecies.gradY[j*s.nCells+c];sumZ+=s.gasSpecies.gradZ[j*s.nCells+c];}
                    s.gasSpecies.gradX[k*s.nCells+c]=-sumX;
                    s.gasSpecies.gradY[k*s.nCells+c]=-sumY;
                    s.gasSpecies.gradZ[k*s.nCells+c]=-sumZ;
                }
            }
    }
}

template<class GasState>
__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue
(
    const GasState& s,
    const int f,
    const int owner
)
{
    const GPU_OPERATOR_REAL rhoSafe = clampMin(riemannFacePrimitiveForGradient(s, owner, f).rho, s.rhoMin);
    const GPU_OPERATOR_REAL nu = s.gasMu/rhoSafe;
    if (s.sstWallTreatment == 0)
    {
        // OF10 viscous omegaWallFunction branch: wall nu, cell wall distance.
        // This constrains the adjacent cell, not a wall Dirichlet value.
        return ugkwp::sstLowReWallOmega
        (
            nu, s.sstWallDistance[owner], s.sstCoefficients
        );
    }
    const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
    const GPU_OPERATOR_REAL dux = s.Ux[owner] - wallUx;
    const GPU_OPERATOR_REAL duy = s.Uy[owner] - wallUy;
    const GPU_OPERATOR_REAL duz = s.Uz[owner] - wallUz;
    const GPU_OPERATOR_REAL y = clampMin(s.sstWallDistance[owner], OfVSmall);
    const GPU_OPERATOR_REAL magGradU = sqrt(dux*dux + duy*duy + duz*duz)*s.deltaCoeffs[f];
    return ugkpwall::omegaWallFunctionState
    (
        s.k[owner],
        magGradU,
        y,
        nu,
        s.sstCoefficients.beta1,
        s.sstWallCmu,
        s.sstWallKappa,
        s.sstWallE
    ).omega;
}

template<class GasState>
__device__ GPU_OPERATOR_REAL sstBoundaryValue
(
    const GasState& s,
    const int f,
    const int owner,
    const bool omegaField
)
{
    const GPU_OPERATOR_REAL centre = omegaField ? s.omega[owner] : s.k[owner];
    const int boundaryKind = s.riemannBoundaryKind[f];
    if (boundaryKind == 2)
    {
        if (!omegaField)
        {
            return s.sstWallTreatment == 0 ? GPU_OPERATOR_R(0.0) : centre;
        }
        // Copy constrained cell omega to the wall: zero wall diffusion flux.
        return centre;
    }
    if (boundaryKind == 1 || boundaryKind == 3 || boundaryKind == 4)
    {
        return centre;
    }

    const int mode = omegaField
      ? s.sstBoundaryOmegaMode[f] : s.sstBoundaryKMode[f];
    const GPU_OPERATOR_REAL prescribed = omegaField
      ? s.sstBoundaryOmega[f] : s.sstBoundaryK[f];
    if (mode == 1)
    {
        return prescribed;
    }
    if (mode == 2)
    {
        // inletOutlet uses the same-stage face flux, including pressure-driven
        // inflow opposing the owner velocity. Zero flux is the outlet branch.
        return s.gasPhiRho[f] >= GPU_OPERATOR_R(0.0) ? centre : prescribed;
    }
    return centre;
}

template<class GasState>
__device__ GPU_OPERATOR_REAL sstFaceDensity(const GasState& s, const int f)
{
    const int own = s.faceOwner[f];
    const int nei = coupledFaceNeighbour(s, f);
    if (nei >= 0)
    {
        const GPU_OPERATOR_REAL weight = clampRange
        (
            s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0)
        );
        return clampMin
        (
            weight*s.rho[own] + (GPU_OPERATOR_R(1.0) - weight)*s.rho[nei],
            s.rhoMin
        );
    }
    return clampMin(riemannFacePrimitiveForGradient(s, own, f).rho, s.rhoMin);
}

// Courant owns unscaled Riemann mass flux. Apply the same periodic averaging
// algebra on read; the advance path already stores finalized antisymmetric phi.
template<class GasState>
__device__ GPU_OPERATOR_REAL sstPredictorMassFlux(const GasState& s, const int f)
{
    return isPeriodicFace(s, f)
      ? GPU_OPERATOR_R(0.5)*(s.gasPhiRho[f] - s.gasPhiRho[s.facePeriodicPair[f]])
      : s.gasPhiRho[f];
}

template<class GasState>
__device__ void applySstWallFunctionStateCell(GasState& s, const int c)
{
    if
    (
        c >= s.nCells
     || s.sstConfigured == 0
    )
    {
        return;
    }

    const GPU_OPERATOR_REAL beforeK=s.rhoK[c],beforeOmega=s.rhoOmega[c];
    GPU_OPERATOR_REAL omegaSum = GPU_OPERATOR_R(0.0);
    int wallCount = 0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f >= s.nInternalFaces
         && f < s.nFaces
         && s.riemannBoundaryKind[f] == 2
        )
        {
            omegaSum += sstDynamicOmegaWallValue(s, f, c);
            ++wallCount;
        }
    }
    if (wallCount > 0)
    {
        const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
        const GPU_OPERATOR_REAL omegaTarget = clampMin
        (
            finiteOr(omegaSum/GPU_OPERATOR_REAL(wallCount), s.sstOmegaMin),
            s.sstOmegaMin
        );
        s.omega[c] = omegaTarget;
        s.rhoOmega[c] = rhoSafe*omegaTarget;
    }
    ugkwp::gasSstAuditConstraint(s,c,beforeK,beforeOmega);
}

template<class GasState>
__global__ void applySstWallFunctionStateKernel(GasState* sp)
{
    applySstWallFunctionStateCell
    (
        *sp, blockIdx.x*blockDim.x + threadIdx.x
    );
}

template<class GasState>
__global__ void initialiseSstConservativeStateKernel(GasState* sp)
{
    GasState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c == 0 && s.sstConfigured != 0)
    {
        const GPU_OPERATOR_REAL PrRatio =
            s.gasPrClamped/clampMin(s.turbulentPrandtl, OfSmall);
        s.sstJayatillekeP = ugkpwall::jayatillekeSmoothP(PrRatio);
        s.sstThermalYPlus = ugkpwall::jayatillekeThermalYPlus
        (
            PrRatio,
            s.sstWallKappa,
            s.sstWallE
        );
    }
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL kSafe = clampMin(finiteOr(s.k[c], s.sstKMin), s.sstKMin);
    const GPU_OPERATOR_REAL omegaSafe =
        clampMin(finiteOr(s.omega[c], s.sstOmegaMin), s.sstOmegaMin);
    s.k[c] = kSafe;
    s.omega[c] = omegaSafe;
    s.rhoK[c] = rhoSafe*kSafe;
    s.rhoOmega[c] = rhoSafe*omegaSafe;
    s.rhoKInitial[c] = s.rhoK[c];
    s.rhoOmegaInitial[c] = s.rhoOmega[c];
    s.nut[c] = GPU_OPERATOR_R(0.0);
}

template<class GasState>
__device__ void recoverSstPrimitiveCell(GasState& s, const int c)
{
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const GPU_OPERATOR_REAL beforeK=s.rhoK[c],beforeOmega=s.rhoOmega[c];
    const GPU_OPERATOR_REAL rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const GPU_OPERATOR_REAL rhoKFloor = rhoSafe*s.sstKMin;
    const GPU_OPERATOR_REAL rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.rhoK[c] = clampMin(finiteOr(s.rhoK[c], rhoKFloor), rhoKFloor);
    s.rhoOmega[c] =
        clampMin(finiteOr(s.rhoOmega[c], rhoOmegaFloor), rhoOmegaFloor);
    s.k[c] = s.rhoK[c]/rhoSafe;
    s.omega[c] = s.rhoOmega[c]/rhoSafe;
    ugkwp::gasSstAuditConstraint(s,c,beforeK,beforeOmega);
    if (s.sstWallTreatment == 0 || s.sstWallTreatment == 1)
    {
        // Project the wall-adjacent omega equation after Euler updates and RK
        // blends. Refresh the target from recovered k and the current gas state
        // (including wall viscosity); do not freeze k or the gas equations.
        // This is an explicit stage constraint, not an implicit OF10 solve.
        applySstWallFunctionStateCell(s, c);
    }
}

template<class GasState>
__global__ void recoverSstPrimitivesKernel(GasState* sp)
{
    recoverSstPrimitiveCell(*sp, blockIdx.x*blockDim.x + threadIdx.x);
}
