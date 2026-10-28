#pragma once
#include "SteadyChemistryResidual.cuh"

// Steady driving deliberately calls the same face, reconstruction, molecular
// transport, turbulence and wall stages as the transient resident solver.
// The common operators scale only their cell updates by gasPseudoDt/referenceDt.
__global__ void prepareSteadyGasTimeStepKernel(DeviceState* sp,double courant,double maximumDt)
{
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    const bool mixture=ugkwp::mixtureGasActive(s);
    const double cp=mixture?s.gasSpecies.heatCapacity[c]:s.gasCp;
    const double R=mixture?s.gasSpecies.gasConstant[c]:s.Rgas;
    const double cv=cp-R;
    double diffusionRate=0;
    const int begin=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
    for(int j=0;j<count;++j)
    {
        const int f=s.cellFaceId[begin+j];
        const int other=coupledFaceNeighbour(s,f);
        const int own=s.faceOwner[f];
        const double mu=ugkwp::gasFaceMolecularViscosity(s,f,own,other);
        const int adjacent=own==c?other:own;
        const double nut=fmax(s.nut[c],adjacent>=0?s.nut[adjacent]:s.nut[c]);
        const double faceCp=ugkwp::gasFaceMolecularHeatCapacity(s,f,own,other);
        const double densityRatio=adjacent>=0?fmax(1.0,s.rho[adjacent]/s.rho[c]):1.0;
        const double nu=mu/s.rho[c];
        const double thermal=s.gasThermalConductivity>=0
            ?s.gasThermalConductivity/(s.rho[c]*cv)
            :nu*faceCp/(cv*s.gasPr);
        double diffusivity=fmax(4.0/3.0*(nu+densityRatio*nut),
            fmax(thermal+densityRatio*nut*faceCp/(cv*s.turbulentPrandtl),nu+densityRatio*nut));
        if(mixture)
            for(int k=0;k<ugkwp::compiledGasSpecies;++k)
            {
                const double D=s.gasSpecies.diffusivity?s.gasSpecies.diffusivity[k]:0;
                // Mixture correction couples species: sum of raw diagonal
                // diffusion plus correction has row norm bounded by 2 max D.
                diffusivity=fmax(diffusivity,2*densityRatio*(D+nut/s.gasSpecies.turbulentSchmidt));
            }
        diffusionRate+=2*diffusivity*s.magSf[f]*s.deltaCoeffs[f]/s.V[c];
    }
    // Courant scratch contains Co(dt=1). Add explicit diffusive and SST
    // destruction rates; recomputed every pseudo iteration, never host fields.
    const double sourceRate=s.sstConfigured?s.omega[c]:0;
    const double rate=s.gasFluxPositivityScale[c]+diffusionRate+sourceRate;
    s.gasPseudoDt[c]=fmin(maximumDt,courant/fmax(rate,1e-300));
}

// Raw FV equation residual, not a timestep-weighted update norm. Components:
// rho, rhoUx/y/z, rhoE, each species, rhoK, unconstrained rhoOmega.
__global__ void steadyGasSpatialResidualKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    constexpr int Ns=ugkwp::compiledGasSpecies;
    double r[7+Ns]={};
    double volumeFlux=0;
    bool constrainedOmega=ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
    for(int j=0;j<s.cellPlaneCount[c];++j)
    {
        const int f=s.cellFaceId[s.cellPlaneStart[c]+j];
        const double sign=s.faceOwner[f]==c?-1:1;
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
        double sourceK=0,sourceOmega=0;
        if(!sstSourcesForCell(s,c,volumeFlux/s.V[c],sourceK,sourceOmega))return;
        r[5+Ns]+=sourceK;
        r[6+Ns]=constrainedOmega?0:r[6+Ns]+sourceOmega;
    }
    for(int k=0;k<7+Ns;++k)s.gasSteadyResidual[k*s.nCells+c]=fabs(r[k]);
}
