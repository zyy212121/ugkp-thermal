#pragma once
#include "gasStencilDiagnostics.cuh"
#include "GpuCellNeighbour.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__device__ GasPrimDevice riemannFacePrimitiveForGradient
(
    const GasState& s,
    const int c,
    const int f,
    const bool nativeRefinement=false
)
{
    const GasPrimDevice invalid={GPU_OPERATOR_REAL(NAN),GPU_OPERATOR_REAL(NAN),GPU_OPERATOR_REAL(NAN),
        GPU_OPERATOR_REAL(NAN),GPU_OPERATOR_REAL(NAN),GPU_OPERATOR_REAL(NAN)};
    if(c<0 || c>=s.nCells){gasStencilIndexFailure(s,c,c,30040);return invalid;}
    if(!gasStencilFaceValid(s,c,f))return invalid;
    if(!gasProducerScalarValid(s,c,s.Tgas[c],30041,true))return invalid;
    if(f>=s.nInternalFaces && !isPeriodicFace(s,f))
    {
        const int kind=s.riemannBoundaryKind[f];
        if((kind==1 || kind==2) && s.riemannBoundaryTFix[f]
            && !gasProducerScalarValid(s,c,s.riemannBoundaryT[f],30042,true))return invalid;
        if(kind==2 && s.riemannBoundaryUFix[f]
            && (!gasProducerScalarValid(s,c,s.riemannBoundaryUx[f],30043)
                ||!gasProducerScalarValid(s,c,s.riemannBoundaryUy[f],30112)
                ||!gasProducerScalarValid(s,c,s.riemannBoundaryUz[f],30113)))return invalid;
    }
    GasPrimDevice centre = gasCellPrimitive(s,c);
    centre.T = clampMin(finiteOr(s.Tgas[c], centre.T), s.TgasMin);
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const int other = oppositeCellAcrossFace(c, own, nei);
        if (other < 0 || other >= s.nCells)
        {
            gasStencilIndexFailure(s,c,other,30044);return invalid;
        }
        if(!gasProducerScalarValid(s,c,s.Tgas[other],30045,true))return invalid;
        GasPrimDevice adjacent = gasCellPrimitive(s,other);
        adjacent.T = clampMin
        (
            finiteOr(s.Tgas[other], adjacent.T),
            s.TgasMin
        );
        const GPU_OPERATOR_REAL ownerWeight = clampRange(s.faceWeight[f], GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        const GPU_OPERATOR_REAL wc = c == own ? ownerWeight : GPU_OPERATOR_R(1.0) - ownerWeight;
        if(ugkwp::mixtureGasActive(s))
            return GasPrimDevice
            {
                wc*centre.rho+(GPU_OPERATOR_R(1.0)-wc)*adjacent.rho,
                wc*centre.ux+(GPU_OPERATOR_R(1.0)-wc)*adjacent.ux,
                wc*centre.uy+(GPU_OPERATOR_R(1.0)-wc)*adjacent.uy,
                wc*centre.uz+(GPU_OPERATOR_R(1.0)-wc)*adjacent.uz,
                wc*centre.p+(GPU_OPERATOR_R(1.0)-wc)*adjacent.p,
                wc*centre.T+(GPU_OPERATOR_R(1.0)-wc)*adjacent.T
            };
        GasPrimDevice face = makeGasPrimDevice
        (
            wc*centre.rho + (GPU_OPERATOR_R(1.0) - wc)*adjacent.rho,
            wc*centre.ux + (GPU_OPERATOR_R(1.0) - wc)*adjacent.ux,
            wc*centre.uy + (GPU_OPERATOR_R(1.0) - wc)*adjacent.uy,
            wc*centre.uz + (GPU_OPERATOR_R(1.0) - wc)*adjacent.uz,
            wc*centre.p + (GPU_OPERATOR_R(1.0) - wc)*adjacent.p,
            s.Rgas, s.rhoMin, s.TgasMin
        );
        face.T = wc*centre.T + (GPU_OPERATOR_R(1.0) - wc)*adjacent.T;
        return face;
    }

    const int kind = s.riemannBoundaryKind[f];
    if (kind == 4 || kind == 3)
    {
        return centre;
    }
    if (kind == 1)
    {
        const GPU_OPERATOR_REAL area = clampMin(s.magSf[f], OfSmall);
        const GPU_OPERATOR_REAL nx = s.Sfx[f]/area;
        const GPU_OPERATOR_REAL ny = s.Sfy[f]/area;
        const GPU_OPERATOR_REAL nz = s.Sfz[f]/area;
        GPU_OPERATOR_REAL un = centre.ux*nx + centre.uy*ny + centre.uz*nz;
        if(ugkwp::gasMovingGeometry(s))
        {
            GPU_OPERATOR_REAL meshNormal;
            if(!gasBoundaryMeshNormalSpeed(s,f,meshNormal,nativeRefinement))return centre;
            un-=meshNormal;
        }
        GasPrimDevice face = centre;
        face.ux -= un*nx;
        face.uy -= un*ny;
        face.uz -= un*nz;
        // Slip constrains velocity, not the independent thermal boundary.
        if (s.riemannBoundaryTFix[f] != 0)
        {
            face.T = clampMin(finiteOr(s.riemannBoundaryT[f], centre.T), s.TgasMin);
            GPU_OPERATOR_REAL R = s.Rgas;
            if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
                if (ugkwp::mixtureGasActive(s)) R = s.gasSpecies.gasConstant[c];
            face.rho = face.p/clampMin(R*face.T, OfSmall);
        }
        return face;
    }

    if (kind == 2)
    {
        GasPrimDevice wall = centre;
        wall.ux = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        wall.uy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        wall.uz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        if(ugkwp::gasMovingGeometry(s))
        {
            GPU_OPERATOR_REAL meshNormal;
            if(!gasBoundaryMeshNormalSpeed(s,f,meshNormal,nativeRefinement))return centre;
            const GPU_OPERATOR_REAL area=s.magSf[f],nx=s.Sfx[f]/area,ny=s.Sfy[f]/area,nz=s.Sfz[f]/area;
            const GPU_OPERATOR_REAL correction=meshNormal-(wall.ux*nx+wall.uy*ny+wall.uz*nz);
            wall.ux+=correction*nx;wall.uy+=correction*ny;wall.uz+=correction*nz;
        }
        if (s.riemannBoundaryTFix[f] != 0)
        {
            wall.T = clampMin
            (
                finiteOr(s.riemannBoundaryT[f], centre.T),
                s.TgasMin
            );
            GPU_OPERATOR_REAL R=s.Rgas;
            if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
                if(ugkwp::mixtureGasActive(s))R=s.gasSpecies.gasConstant[c];
            wall.rho = wall.p/clampMin(R*wall.T, OfSmall);
        }
        return wall;
    }

    return riemannBoundaryState(s, f, centre,nullptr,nativeRefinement);
}
