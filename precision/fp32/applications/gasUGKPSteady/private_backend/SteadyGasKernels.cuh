#pragma once
#include "SteadyChemistryResidual.cuh"
#include <cmath>
#include <limits>

// Steady driving deliberately calls the same face, reconstruction, molecular
// transport, turbulence and wall stages as the transient resident solver.
// The common operators scale only their cell updates by gasPseudoDt/referenceDt.
__global__ void prepareSteadyGasTimeStepKernel(DeviceState* sp,GpuReal courant,GpuTime maximumDt)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    const bool mixture=ugkwp::mixtureGasActive(s);
    const GpuReal cp=mixture?s.gasSpecies.heatCapacity[c]:s.gasCp;
    const GpuReal R=mixture?s.gasSpecies.gasConstant[c]:s.Rgas;
    const GpuReal cv=cp-R;
    GpuReal diffusionRate=0;
    const int begin=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
    for(int j=0;j<count;++j)
    {
        const int f=s.cellFaceId[begin+j];
        const int other=coupledFaceNeighbour(s,f);
        const int own=s.faceOwner[f];
        const GpuReal mu=ugkwp::gasFaceMolecularViscosity(s,f,own,other);
        const int adjacent=own==c?other:own;
        const GpuReal nut=std::fmax(s.nut[c],adjacent>=0?s.nut[adjacent]:s.nut[c]);
        const GpuReal faceCp=ugkwp::gasFaceMolecularHeatCapacity(s,f,own,other);
        const GpuReal densityRatio=adjacent>=0?std::fmax(GpuReal(1),s.rho[adjacent]/s.rho[c]):GpuReal(1);
        const GpuReal nu=mu/s.rho[c];
        const GpuReal thermal=s.gasThermalConductivity>=0
            ?s.gasThermalConductivity/(s.rho[c]*cv)
            :nu*faceCp/(cv*(ugkwp::gasVariablePrandtl(s)?ugkwp::gasFaceMolecularPrandtl(s,f,own,other):s.gasPr));
        GpuReal diffusivity=std::fmax(GpuReal(4)/GpuReal(3)*(nu+densityRatio*nut),
            std::fmax(thermal+densityRatio*nut*faceCp/(cv*s.turbulentPrandtl),nu+densityRatio*nut));
        if(mixture)
            for(int k=0;k<ugkwp::compiledGasSpecies;++k)
            {
                const GpuReal w=ugkwp::gasTransportFaceWeight(s,f,own,other);
                const GpuReal T=ugkwp::gasTransportFaceTemperature(s,f,own,other);
                const GpuReal pressure=other>=0?w*s.p[own]+(1-w)*s.p[other]
                    :(s.riemannBoundaryPFix[f]?s.riemannBoundaryP[f]:s.p[own]);
                const GpuReal D=ugkwp::gasLocalMolecularDiffusivity(s,k,T,pressure);
                // Mixture correction couples species: sum of raw diagonal
                // diffusion plus correction has row norm bounded by 2 max D.
                diffusivity=std::fmax(diffusivity,2*densityRatio*(D+nut/s.gasSpecies.turbulentSchmidt));
            }
        diffusionRate+=2*diffusivity*s.magSf[f]*s.deltaCoeffs[f]/s.V[c];
    }
    // Courant scratch contains Co(dt=1). Add explicit diffusive and SST
    // destruction rates; recomputed every pseudo iteration, never host fields.
    const GpuReal sourceRate=s.sstConfigured?s.omega[c]:0;
    const GpuReal rate=s.gasFluxPositivityScale[c]+diffusionRate+sourceRate;
    s.gasPseudoDt[c]=std::fmin(maximumDt,GpuTime(courant
        /std::fmax(rate,std::numeric_limits<GpuReal>::min())));
}

// Raw FV equation residual, not a timestep-weighted update norm. Components:
// rho, rhoUx/y/z, rhoE, each species, rhoK, unconstrained rhoOmega.
__global__ void steadyGasSpatialResidualKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    constexpr int Ns=ugkwp::compiledGasSpecies;
    GpuReal r[7+Ns]={};
    GpuReal volumeFlux=0;
    bool constrainedOmega=ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
    for(int j=0;j<s.cellPlaneCount[c];++j)
    {
        const int f=s.cellFaceId[s.cellPlaneStart[c]+j];
        const GpuReal sign=s.faceOwner[f]==c?-1:1;
        r[0]+=sign*s.gasPhiRho[f];
        r[1]+=sign*s.gasPhiRhoUx[f];
        r[2]+=sign*s.gasPhiRhoUy[f];
        r[3]+=sign*s.gasPhiRhoUz[f];
        r[4]+=sign*s.gasPhiRhoE[f];
        if(ugkwp::mixtureGasActive(s))
            for(int k=0;k<Ns;++k)r[5+k]+=sign*s.gasSpecies.flux[k*s.nFaces+f];
        if(s.sstConfigured)
        {
            r[5+Ns]+=sign*s.sstPhiRhoK[f];
            r[6+Ns]+=sign*s.sstPhiRhoOmega[f];
            volumeFlux-=sign*s.gasPhiRho[f]/sstFaceDensity(s,f);
            constrainedOmega=constrainedOmega||
                ((s.sstWallTreatment==0||s.sstWallTreatment==1||s.sstWallTreatment==2)
                 &&f>=s.nInternalFaces&&s.riemannBoundaryKind[f]==2);
        }
    }
    for(int k=0;k<7+Ns;++k)r[k]/=s.V[c];
    if(!steadyAddChemistryResidual(s,c,r+5))
    {
        // An unavailable chemical source must never appear as zero/converged.
        for(int k=0;k<7+Ns;++k)s.gasSteadyResidual[k*s.nCells+c]=NAN;
        return;
    }
    r[1]+=s.rho[c]*s.gravityX;
    r[2]+=s.rho[c]*s.gravityY;
    r[3]+=s.rho[c]*s.gravityZ;
    r[4]+=s.rhoUx[c]*s.gravityX+s.rhoUy[c]*s.gravityY+s.rhoUz[c]*s.gravityZ;
    if(s.sstConfigured)
    {
        GpuReal sourceK=0,sourceOmega=0;
        if(!sstSourcesForCell(s,c,GpuReal(volumeFlux/s.V[c]),sourceK,sourceOmega))return;
        r[5+Ns]+=sourceK;
        r[6+Ns]=constrainedOmega?0:r[6+Ns]+sourceOmega;
    }
    for(int k=0;k<7+Ns;++k)s.gasSteadyResidual[k*s.nCells+c]=std::fabs(r[k]);
}
