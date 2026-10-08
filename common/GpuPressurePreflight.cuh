#pragma once
// Fixed-pass transaction: read-only preview, all-cell decision, then writers.
// Failure is fatal to the stream; it must never silently discard a pressure kick.
#include "GpuPressureFailure.cuh"
__global__ void preparePressurePreflightKernel(DeviceState* sp,PressureTime dt)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    for(int k=0;k<10;++k)s.pressurePreviewMoments[10*static_cast<size_t>(c)+k]=PressureReal(0);
    PressureReal d[4];pressureDeltaFromLimitedFaces(s,c,dt,d);
    const PressureReal rho=s.momRhoP[c],px=s.momRhoUPx[c],py=s.momRhoUPy[c],pz=s.momRhoUPz[c],e=s.momRhoEP[c];
    if(!finiteDevice(rho)||!finiteDevice(px)||!finiteDevice(py)||!finiteDevice(pz)||!finiteDevice(e)
       ||rho<0||e<0||!finiteDevice(s.V[c])||s.V[c]<=0)
    {recordPressureFailure(s,pressureBadInitial,c);return;}
    if(rho==0)
    {
        if(px!=0||py!=0||pz!=0||e!=0)recordPressureFailure(s,pressureBadInitial,c);
        return;
    }
    const PressureReal i0=pressureKickInternalEnergy(rho,px,py,pz,e);
    if(!finiteDevice(i0)||i0<0){recordPressureFailure(s,pressureBadInitial,c);return;}
    if(rho<=s.epsSMin*s.rhoSolid)return;
    const PressureReal i1=pressureKickInternalEnergy(rho,px+d[0],py+d[1],pz+d[2],e+d[3]);
    const PressureReal floor=fmin(i0,PressureReal(1.5)*rho*s.thetaMin);
    const PressureReal maxDU=s.pressureKickFraction*s.cellLength[c]/dt;
    const PressureReal du=pressure_convex::norm3(d[0],d[1],d[2])/rho;
    if(!finiteDevice(d[0])||!finiteDevice(d[1])||!finiteDevice(d[2])||!finiteDevice(d[3])
       ||!finiteDevice(i1)||i1<floor||!finiteDevice(du)||du>maxDU
       ||(maxDU==0&&(d[0]!=0||d[1]!=0||d[2]!=0)))
        recordPressureFailure(s,pressureBadFinal,c);
}
__device__ inline PressureParameters pressurePreviewParameters(DeviceState& s,int c,bool unsorted)
{
    PressureReal d[4]={s.pressureDeltaMomX[c],s.pressureDeltaMomY[c],s.pressureDeltaMomZ[c],s.pressureDeltaEnergy[c]};
    PressureParameters q{};makePressureParameters(s,c,d,q);
    if(q.active&&unsorted)
    {
        const auto r=recoverUnsortedPressureKinematics(s.momRhoP[c],q.px1,q.py1,q.pz1,q.e1,d[0],d[1],d[2],d[3],s.thetaMin);
        q.ux0=r.ux0;q.uy0=r.uy0;q.uz0=r.uz0;q.ux1=r.ux1;q.uy1=r.uy1;q.uz1=r.uz1;
        q.theta1=r.theta1;q.thermalScale=r.thermalScale;q.thetaScale=r.thetaScale;q.resolved=r.resolved;
    }
    return q;
}
template<bool Compact>
__device__ inline void previewPressureParticle(DeviceState& s,int c,int i,const PressureParameters& q,PressureReal (&sum)[10])
{
    if(i<0||i>=s.particleCapacity){recordPressureFailure(s,pressureBadParticle,c);return;}
    const int status=Compact?s.compactPStatus[i]:s.pStatus[i];
    if(status==0)return;
    if((Compact?s.compactPCellId[i]:s.pCellId[i])!=c)
    {recordPressureFailure(s,pressureBadParticle,c);return;}
    // Every writer-eligible record is checked, even if a moment closure would omit it.
    const PressureReal mass=Compact?s.compactPm[i]:s.pm[i];
    const PressureReal ux=Compact?s.compactPux[i]:s.pux[i],uy=Compact?s.compactPuy[i]:s.puy[i],uz=Compact?s.compactPuz[i]:s.puz[i];
    const PressureReal theta=Compact?s.compactPTheta[i]:s.pTheta[i];
    if(!finiteDevice(mass)||mass<0||!finiteDevice(ux)||!finiteDevice(uy)||!finiteDevice(uz)||!finiteDevice(theta)||theta<0)
    {recordPressureFailure(s,pressureBadParticle,c);return;}
    const bool identity=!q.active||pressureDeltaIsZero(s,c);
    const bool stuck=PressureConstraintPolicy::template stuck<Compact>(s,i);
    const auto candidate=identity?PressureParticleCandidate<PressureReal>{ux,uy,uz,stuck?PressureReal(0):theta}:
        stuck?PressureParticleCandidate<PressureReal>{0,0,0,0}:
        pressureParticleCandidate(ux,uy,uz,theta,q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,q.theta1,q.thermalScale,q.thetaScale,q.resolved);
    const PressureReal e=mass*(PressureReal(.5)*sqr3(candidate.ux,candidate.uy,candidate.uz)+PressureReal(1.5)*candidate.theta);
    if(!finiteDevice(candidate.ux)||!finiteDevice(candidate.uy)||!finiteDevice(candidate.uz)||!finiteDevice(candidate.theta)||candidate.theta<0||!finiteDevice(e))
    {recordPressureFailure(s,pressureBadParticle,c);return;}
    // Production closures count status==1. Reject any other nonzero status on
    // an active transaction rather than writing an unaccounted-for particle.
    if(status!=1){recordPressureFailure(s,pressureBadParticle,c);return;}
    sum[0]+=mass*candidate.ux;sum[1]+=mass*candidate.uy;sum[2]+=mass*candidate.uz;
    sum[3]+=e;sum[4]+=mass;sum[5]+=PressureReal(1);
    sum[6]+=mass*ux;sum[7]+=mass*uy;sum[8]+=mass*uz;
    sum[9]+=mass*(PressureReal(.5)*sqr3(ux,uy,uz)+PressureReal(1.5)*(stuck?PressureReal(0):theta));
}
// Defined by each application's established scalar/warp reduction policy.
template<int N> __device__ void blockReduceComponentSums(PressureReal (&sum)[N],PressureReal* partials);
template<bool Split,bool Compact>
__global__ void previewPressureSortedKernel(DeviceState* sp)
{
    DeviceState& s=*sp;const int c=blockIdx.x;if(c>=s.nCells)return;
    __shared__ PressureParameters q;
    if(threadIdx.x==0)q=pressurePreviewParameters(s,c,false);
    __syncthreads();
    PressureReal sum[10]={};
    if constexpr(Split)
        for(int i=s.preBaseCellOffset[c]+threadIdx.x;i<s.preBaseCellOffset[c+1];i+=blockDim.x)
            if((Compact?s.compactPCellId[i]:s.pCellId[i])==c)previewPressureParticle<Compact>(s,c,i,q,sum);
    for(int pos=s.cellParticleOffset[c]+threadIdx.x;pos<s.cellParticleOffset[c+1];pos+=blockDim.x)
    {
        const int i=Compact?pos:s.sortedParticleIndex[pos];
        if constexpr(Split) {if(i>=0&&i<s.particleCapacity&&(Compact?s.compactPCellId[i]:s.pCellId[i])!=c)continue;}
        previewPressureParticle<Compact>(s,c,i,q,sum);
    }
    // Fixed 128-thread launch: four warp partials per component. Reuse the
    // application's native reduction, avoiding 1280 atomics per sorted cell.
    __shared__ PressureReal partials[10*4];
    blockReduceComponentSums<10>(sum,partials);
    if(threadIdx.x==0)
        for(int k=0;k<10;++k)s.pressurePreviewMoments[10*static_cast<size_t>(c)+k]=sum[k];
}
__global__ void previewPressureUnsortedKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    const int n=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
    {
        if(s.pStatus[i]==0)continue;
        const int c=s.pCellId[i];if(c<0||c>=s.nCells)continue;
        const auto q=pressurePreviewParameters(s,c,true);
        PressureReal sum[10]={};previewPressureParticle<false>(s,c,i,q,sum);
        for(int k=0;k<10;++k)atomicAdd(s.pressurePreviewMoments+10*static_cast<size_t>(c)+k,sum[k]);
    }
}
__global__ void auditPressurePreviewKernel(DeviceState* sp,PressureTime dt)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    const PressureReal* a=s.pressurePreviewMoments+10*static_cast<size_t>(c);
    if(s.momRhoP[c]==PressureReal(0))
    {
        if(a[0]!=0||a[1]!=0||a[2]!=0||a[3]!=0||a[4]!=0)
            recordPressureFailure(s,pressureUnrealizableParticles,c);
        return;
    }

    const PressureReal rho=a[4]/s.V[c],px=a[0]/s.V[c],py=a[1]/s.V[c],pz=a[2]/s.V[c],e=a[3]/s.V[c];
    const PressureReal i0=pressureKickInternalEnergy(s.momRhoP[c],s.momRhoUPx[c],s.momRhoUPy[c],s.momRhoUPz[c],s.momRhoEP[c]);
    const PressureReal floor=fmin(i0,PressureReal(1.5)*s.momRhoP[c]*s.thetaMin);
    const PressureReal internal=pressureKickInternalEnergy(rho,px,py,pz,e);
    const PressureReal dux=a[0]/a[4]-a[6]/a[4],duy=a[1]/a[4]-a[7]/a[4],duz=a[2]/a[4]-a[8]/a[4];
    const PressureReal du=pressure_convex::norm3(dux,duy,duz);
    const PressureReal maxDU=s.pressureKickFraction*s.cellLength[c]/dt;
    if(!finiteDevice(rho)||rho<=0||!finiteDevice(px)||!finiteDevice(py)||!finiteDevice(pz)||!finiteDevice(e)
       ||(!pressureDeltaIsZero(s,c)&&(!finiteDevice(internal)||internal<floor||!finiteDevice(du)||du>maxDU
          ||(maxDU==0&&(dux!=0||duy!=0||duz!=0))
          ||!finiteDevice(px/rho)||!finiteDevice(py/rho)||!finiteDevice(pz/rho)
          ||!finiteDevice(internal/(PressureReal(1.5)*rho))||!finiteDevice(rho/s.rhoSolid))))
    {recordPressureFailure(s,pressureUnrealizableParticles,c);return;}

    const PressureReal target[5]={s.momRhoUPx[c]+s.pressureDeltaMomX[c],s.momRhoUPy[c]+s.pressureDeltaMomY[c],s.momRhoUPz[c]+s.pressureDeltaMomZ[c],s.momRhoEP[c]+s.pressureDeltaEnergy[c],s.momRhoP[c]};
    const PressureReal eps=sizeof(PressureReal)==4?PressureReal(1.1920928955078125e-7):PressureReal(2.220446049250313e-16);
    const PressureReal tol=fmin(PressureReal(.0001),PressureReal(32)*eps*(PressureReal(1)+a[5]));
    // Cap tolerance at 0.01%; difficult reductions may fail closed rather
    // than granting arbitrarily large conservation error to large cells.
    // A floating-point count cannot certify integers beyond its exact range.
    // Refuse huge cells rather than overflow the published integer count.
    if(!finiteDevice(a[5])||a[5]<0||a[5]>PressureReal(2147483647.0)
       ||(sizeof(PressureReal)==4&&a[5]>=PressureReal(16777216.0)))
    {recordPressureFailure(s,pressureUnrealizableParticles,c);return;}
    const PressureReal mass=s.momRhoP[c]*s.V[c];
    const PressureReal energy=target[3]*s.V[c];
    const PressureReal momentumScale=pressure_convex::momentumClosureScale(mass,energy);
    if(!finiteDevice(momentumScale)){recordPressureFailure(s,pressureUnrealizableParticles,c);return;}
    const PressureReal initial[4]={s.momRhoUPx[c],s.momRhoUPy[c],s.momRhoUPz[c],s.momRhoEP[c]};
    const PressureReal initialMomentumScale=pressure_convex::momentumClosureScale(mass,initial[3]*s.V[c]);
    if(!finiteDevice(initialMomentumScale)){recordPressureFailure(s,pressureUnrealizableParticles,c);return;}
    for(int k=0;k<4;++k)
    {
        const PressureReal t=initial[k]*s.V[c];
        const PressureReal scale=k<3?fmax(fabs(t),initialMomentumScale):fabs(t);
        if(!finiteDevice(a[6+k])||fabs(a[6+k]-t)>tol*scale)
            recordPressureFailure(s,pressureUnrealizableParticles,c);
    }
    for(int k=0;k<5;++k)
    {
        const PressureReal t=target[k]*s.V[c];
        const PressureReal scale=k<3?fmax(fabs(t),momentumScale):fabs(t);
        if(!finiteDevice(a[k])||!finiteDevice(t)||fabs(a[k]-t)>tol*scale)
            recordPressureFailure(s,pressureUnrealizableParticles,c);
    }
}
// This is the same aggregate the preflight accepted, including density.
// In L0 it runs only after the target-q1 based particle reconstruction.
__global__ void publishPressureCanonicalMomentsKernel(DeviceState* sp)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells||s.pressureFailure[0]||pressureDeltaIsZero(s,c))return;
    const PressureReal* a=s.pressurePreviewMoments+10*static_cast<size_t>(c);
    const PressureReal rho=a[4]/s.V[c];
    const PressureReal px=a[0]/s.V[c],py=a[1]/s.V[c],pz=a[2]/s.V[c],e=a[3]/s.V[c];
    const PressureReal internal=pressureKickInternalEnergy(rho,px,py,pz,e);
    s.momRhoP[c]=rho;s.epsS[c]=rho/s.rhoSolid;s.cellParticleCount[c]=static_cast<int>(a[5]);
    publishPressureCellState(s,c,rho,px,py,pz,e,internal/(PressureReal(1.5)*rho));
}
__global__ void enforcePressurePreflightKernel(DeviceState* sp)
{
    DeviceState& s=*sp;
    if(s.pressureFailure[0])
    {
        printf("collisional pressure preflight failed: mask=%u firstCell=%u firstReason=%u; no pressure state committed\n",s.pressureFailure[0],s.pressureFailure[1]-1,s.pressureFailure[2]);
        asm volatile("trap;");
    }
}
inline cudaError_t launchPressurePreflight(DeviceState* s,PressureTime dt,int grid,int block,bool split,bool compact)
{
    preparePressurePreflightKernel<<<grid,block>>>(s->deviceState,dt);
    cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return err;
    if(s->csrCellLocalPathEnabled)
    {
        if(split)previewPressureSortedKernel<true,false><<<s->nCells,128>>>(s->deviceState);
        else if(compact)previewPressureSortedKernel<false,true><<<s->nCells,128>>>(s->deviceState);
        else previewPressureSortedKernel<false,false><<<s->nCells,128>>>(s->deviceState);
    }
    else previewPressureUnsortedKernel<<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState);
    err=cudaGetLastError();if(err!=cudaSuccess)return err;
    auditPressurePreviewKernel<<<grid,block>>>(s->deviceState,dt);
    err=cudaGetLastError();if(err!=cudaSuccess)return err;
    enforcePressurePreflightKernel<<<1,1>>>(s->deviceState);
    return cudaGetLastError();
}
