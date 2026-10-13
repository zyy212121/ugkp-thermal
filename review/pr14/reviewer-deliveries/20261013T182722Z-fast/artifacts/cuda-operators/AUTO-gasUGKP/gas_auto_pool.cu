#include "GpuResidentStrict.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <cmath>
using TestDirectory=HeavyDirectoryKind;
void selectDirectory(DeviceState*s,TestDirectory k){s->useSplitPreDirectory=k!=TestDirectory::full;s->preInjectionSegmentActive=k==TestDirectory::splitBaseAndInjection;}
int prepareTasks(DeviceState*s,int b,TestDirectory k){return prepareCsrSegmentedReductionTasks(s,b,k);}
int decide(DeviceState*s,int b,TestDirectory k){return runToolB3(s,b,k);}
void clearPool(DeviceState*s,double dt){clearPoissonThermalPoolKernel<<<1,32>>>(s->deviceState,dt);}
int heavyPool(DeviceState*s,double dt,int b){return launchCsrHeavyPoolReduction(s,dt,true,b,s->useSplitPreDirectory);}
int lightPool(DeviceState*s,TestDirectory k,double dt,int b){if(k==TestDirectory::full){accumulatePoissonPoolParticlesByCellKernel<false><<<4,b,8*b*sizeof(double)>>>(s->deviceState,dt);return 0;}return launchSplitPrePoissonPoolLightReduction(s,dt,b,8*b*sizeof(double));}
int expectedSource(TestDirectory k){return int(k==TestDirectory::full?CsrReductionTaskSource::fullIndexed:(k==TestDirectory::baseOnly?CsrReductionTaskSource::splitBaseDirect:CsrReductionTaskSource::splitLogical));}
int directoryTag(TestDirectory k){return int(k);}
constexpr double tolerance=2e-12;


void require(bool ok,const char* message){if(!ok){printf("FAIL %s\n",message);std::exit(2);}}
void check(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
void finish(){check(cudaDeviceSynchronize());check(cudaGetLastError());}
template<class T> __attribute__((noinline)) void mem(T*&p,int n){check(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;i++)::new(static_cast<void*>(p+i)) T{};}
__attribute__((noinline)) DeviceState* make(int block){
    auto*s=new DeviceState{};
    mem(s->csrMaximumOccupancy,1);
mem(s->csrHeavyTaskCount,1);
mem(s->csrHeavyCellCount,1);
mem(s->csrHeavyTaskCursor,1);
mem(s->particleCountDevice,4096);
mem(s->faceOwner,4096);
mem(s->faceNeighbour,4096);
mem(s->facePeriodicPair,4096);
mem(s->facePeriodicDx,4096);
mem(s->facePeriodicDy,4096);
mem(s->facePeriodicDz,4096);
mem(s->V,4096);
mem(s->Cx,4096);
mem(s->Cy,4096);
mem(s->Cz,4096);
mem(s->faceCx,4096);
mem(s->faceCy,4096);
mem(s->faceCz,4096);
mem(s->Sfx,4096);
mem(s->Sfy,4096);
mem(s->Sfz,4096);
mem(s->magSf,4096);
mem(s->deltaCoeffs,4096);
mem(s->faceWeight,4096);
mem(s->cellLength,4096);
mem(s->cellPlaneStart,4096);
mem(s->cellPlaneCount,4096);
mem(s->cellFaceId,4096);
mem(s->cellFaceNeighbor,4096);
mem(s->cellFaceKind,4096);
mem(s->cellFaceRestitution,4096);
mem(s->cellFaceTangential,4096);
mem(s->planeNx,4096);
mem(s->planeNy,4096);
mem(s->planeNz,4096);
mem(s->planeD,4096);
mem(s->rho,4096);
mem(s->rhoUx,4096);
mem(s->rhoUy,4096);
mem(s->rhoUz,4096);
mem(s->rhoE,4096);
mem(s->rhoNext,4096);
mem(s->rhoUxNext,4096);
mem(s->rhoUyNext,4096);
mem(s->rhoUzNext,4096);
mem(s->rhoENext,4096);
mem(s->gasPhiRho,4096);
mem(s->gasPhiRhoUx,4096);
mem(s->gasPhiRhoUy,4096);
mem(s->gasPhiRhoUz,4096);
mem(s->gasPhiRhoE,4096);
mem(s->gasFluxPositivityScale,4096);
mem(s->gasHllcAdcSensor,4096);
mem(s->gasBoundaryKind,4096);
mem(s->gasBoundaryRhoFix,4096);
mem(s->gasBoundaryUFix,4096);
mem(s->gasBoundaryPFix,4096);
mem(s->gasBoundaryTFix,4096);
mem(s->gasBoundaryPWave,4096);
mem(s->gasBoundaryPWaveGamma,4096);
mem(s->gasBoundaryPWaveFieldInf,4096);
mem(s->gasBoundaryPWaveLInf,4096);
mem(s->gasBoundaryRho,4096);
mem(s->gasBoundaryUx,4096);
mem(s->gasBoundaryUy,4096);
mem(s->gasBoundaryUz,4096);
mem(s->gasBoundaryP,4096);
mem(s->gasBoundaryT,4096);
mem(s->riemannBoundaryKind,4096);
mem(s->riemannBoundaryRhoFix,4096);
mem(s->riemannBoundaryUFix,4096);
mem(s->riemannBoundaryPFix,4096);
mem(s->riemannBoundaryTFix,4096);
mem(s->riemannBoundaryPWave,4096);
mem(s->riemannBoundaryPWaveGamma,4096);
mem(s->riemannBoundaryPWaveFieldInf,4096);
mem(s->riemannBoundaryPWaveLInf,4096);
mem(s->riemannBoundaryRho,4096);
mem(s->riemannBoundaryUx,4096);
mem(s->riemannBoundaryUy,4096);
mem(s->riemannBoundaryUz,4096);
mem(s->riemannBoundaryP,4096);
mem(s->riemannBoundaryT,4096);
mem(s->scheduledInletFaceMask,4096);
mem(s->pressureScheduleTimes,4096);
mem(s->pressureScheduleValues,4096);
mem(s->volumeFractionScheduleTimes,4096);
mem(s->volumeFractionScheduleValues,4096);
mem(s->gradPx,4096);
mem(s->gradPy,4096);
mem(s->gradPz,4096);
mem(s->gradRhoX,4096);
mem(s->gradRhoY,4096);
mem(s->gradRhoZ,4096);
mem(s->gradUxX,4096);
mem(s->gradUxY,4096);
mem(s->gradUxZ,4096);
mem(s->gradUyX,4096);
mem(s->gradUyY,4096);
mem(s->gradUyZ,4096);
mem(s->gradUzX,4096);
mem(s->gradUzY,4096);
mem(s->gradUzZ,4096);
mem(s->gradTX,4096);
mem(s->gradTY,4096);
mem(s->gradTZ,4096);
mem(s->gasGradientLimiterRho,4096);
mem(s->gasGradientLimiterUx,4096);
mem(s->gasGradientLimiterUy,4096);
mem(s->gasGradientLimiterUz,4096);
mem(s->gasGradientLimiterP,4096);
mem(s->gasGradientLimiterT,4096);
mem(s->nut,4096);
mem(s->gasDiffusionNumber,4096);
mem(s->rhoK,4096);
mem(s->rhoOmega,4096);
mem(s->rhoKInitial,4096);
mem(s->rhoOmegaInitial,4096);
mem(s->k,4096);
mem(s->omega,4096);
mem(s->sstWallDistance,4096);
mem(s->sstF1,4096);
mem(s->sstF2,4096);
mem(s->gradKX,4096);
mem(s->gradKY,4096);
mem(s->gradKZ,4096);
mem(s->gradOmegaX,4096);
mem(s->gradOmegaY,4096);
mem(s->gradOmegaZ,4096);
mem(s->sstPhiRhoK,4096);
mem(s->sstPhiRhoOmega,4096);
mem(s->sstBoundaryKMode,4096);
mem(s->sstBoundaryOmegaMode,4096);
mem(s->sstBoundaryK,4096);
mem(s->sstBoundaryOmega,4096);
mem(s->sstSourceNumber,4096);
mem(s->Ux,4096);
mem(s->Uy,4096);
mem(s->Uz,4096);
mem(s->p,4096);
mem(s->Tgas,4096);
mem(s->couplingRhoOld,4096);
mem(s->couplingUxOld,4096);
mem(s->couplingUyOld,4096);
mem(s->couplingUzOld,4096);
mem(s->couplingTgasOld,4096);
mem(s->epsS,4096);
mem(s->rhoUsx,4096);
mem(s->rhoUsy,4096);
mem(s->rhoUsz,4096);
mem(s->rhoEs,4096);
mem(s->rhoDs,4096);
mem(s->rhoHp,4096);
mem(s->Usx,4096);
mem(s->Usy,4096);
mem(s->Usz,4096);
mem(s->theta,4096);
mem(s->Tp,4096);
mem(s->dMeanCell,4096);
mem(s->epsGPrev,4096);
mem(s->collisionalPressure,4096);
mem(s->pressureKickScale,4096);
mem(s->pressureDeltaMomX,4096);
mem(s->pressureDeltaMomY,4096);
mem(s->pressureDeltaMomZ,4096);
mem(s->pressureDeltaEnergy,4096);
mem(s->solidPressurePhiMomX,4096);
mem(s->solidPressurePhiMomY,4096);
mem(s->solidPressurePhiMomZ,4096);
mem(s->solidPressurePhiEnergy,4096);
mem(s->mobilePackingRho,4096);
mem(s->packingStuckRho,4096);
mem(s->mobilePackingMomX,4096);
mem(s->mobilePackingMomY,4096);
mem(s->mobilePackingMomZ,4096);
mem(s->mobilePackingActiveCellMask,4096);
mem(s->mobilePackingCorrectionCellMask,4096);
mem(s->mobilePackingActiveCellList,4096);
mem(s->mobilePackingCorrectionCellList,4096);
mem(s->mobilePackingFrontierCurrent,4096);
mem(s->mobilePackingFrontierNext,4096);
mem(s->mobilePackingActiveCellCount,4096);
mem(s->mobilePackingCorrectionCellCount,4096);
mem(s->mobilePackingFrontierCurrentCount,4096);
mem(s->mobilePackingFrontierNextCount,4096);
mem(s->thetaDragAlpha,4096);
mem(s->momRhoP,4096);
mem(s->momRhoUPx,4096);
mem(s->momRhoUPy,4096);
mem(s->momRhoUPz,4096);
mem(s->momRhoEP,4096);
mem(s->momRhoPD,4096);
mem(s->momRhoHpP,4096);
mem(s->poolThermalCount,4096);
mem(s->poolThermalSumUx,4096);
mem(s->poolThermalSumUy,4096);
mem(s->poolThermalSumUz,4096);
mem(s->poolThermalSumU2,4096);
mem(s->poissonPoolSampleTargetCount,4096);
mem(s->poissonPoolMass,4096);
mem(s->poissonPoolMomX,4096);
mem(s->poissonPoolMomY,4096);
mem(s->poissonPoolMomZ,4096);
mem(s->poissonPoolEnergy,4096);
mem(s->poissonPoolDiameter,4096);
mem(s->poissonPoolDiameter2,4096);
mem(s->px,4096);
mem(s->py,4096);
mem(s->pz,4096);
mem(s->pux,4096);
mem(s->puy,4096);
mem(s->puz,4096);
mem(s->puxOld,4096);
mem(s->puyOld,4096);
mem(s->puzOld,4096);
mem(s->pT,4096);
mem(s->pTheta,4096);
mem(s->pd,4096);
mem(s->pm,4096);
mem(s->pCellId,4096);
mem(s->pStatus,4096);
mem(s->pRng,4096);
mem(s->pOrigId,4096);
mem(s->sortedParticleIndex,4096);
mem(s->cellParticleCount,4096);
mem(s->cellParticleOffset,4096);
mem(s->preBaseCellOffset,4096);
mem(s->preBaseParticleCountDevice,4096);
mem(s->compactCellOffset,4096);
mem(s->cellParticleWrite,4096);
mem(s->cellScanTempStorage,4096);
mem(s->compactCountDevice,4096);
mem(s->compactSelectTempStorage,4096);
mem(s->csrCellTaskCount,4096);
mem(s->csrCellTaskOffset,4096);
mem(s->csrMultiTaskCellList,4);
mem(s->csrHeavyCellList,4096);
mem(s->csrHeavyTaskCell,4096);
mem(s->csrHeavyTaskBegin,4096);
mem(s->csrHeavyTaskEnd,4096);
mem(s->csrHeavyCellTaskStart,4096);
mem(s->csrHeavyCellTaskCount,4096);
mem(s->csrHeavyPartials,1024);
mem(s->preInjectionHeavyTaskCount,4096);
mem(s->preInjectionHeavyTaskCell,4096);
mem(s->preInjectionHeavyTaskBegin,4096);
mem(s->preInjectionHeavyTaskEnd,4096);
mem(s->preInjectionHeavyCellTaskStart,4096);
mem(s->preInjectionHeavyCellTaskCount,4096);
mem(s->preInjectionHeavyPartials,4096);
mem(s->sourceCell,4096);
mem(s->sourceFace,4096);
mem(s->sourcePx,4096);
mem(s->sourcePy,4096);
mem(s->sourcePz,4096);
mem(s->sourceUx,4096);
mem(s->sourceUy,4096);
mem(s->sourceUz,4096);
mem(s->sourceT,4096);
mem(s->sourceTheta,4096);
mem(s->sourceD,4096);
mem(s->sourceMassRate,4096);
mem(s->sourceResidualMass,4096);
mem(s->compactPx,4096);
mem(s->compactPy,4096);
mem(s->compactPz,4096);
mem(s->compactPux,4096);
mem(s->compactPuy,4096);
mem(s->compactPuz,4096);
mem(s->compactPT,4096);
mem(s->compactPTheta,4096);
mem(s->compactPd,4096);
mem(s->compactPm,4096);
mem(s->compactPCellId,4096);
mem(s->compactPStatus,4096);
mem(s->compactPRng,4096);
mem(s->compactPOrigId,4096);
mem(s->poissonCellCollisionProbability,4096);
    mem(s->csrReductionTasks,128);
    mem(s->deviceState,1);
    s->nCells=4;s->particleCapacity=4096;s->particleWorkGrid=32;
    s->particleBlockThreads=128;s->reductionBlockThreads=block;
    s->multiprocessorCount=2;s->lightBlocksPerSm=2;s->csrHeavyWorkerGrid=2;
    s->csrHeavyTaskCapacity=128;s->csrCellLocalPathEnabled=1;
    s->rhoSolid=1000;s->epsSMin=1e-12;s->thetaMin=1e-12;
    s->particleDiameterFallback=.001;s->particleDiameterMin=1e-6;s->particleDiameterMax=1.;s->injectionParcelMass=1.;
    check(cudaFree(s->cellScanTempStorage));s->cellScanTempStorage=nullptr;
    check(cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage,s->cellScanTempBytes,s->csrCellTaskCount,s->csrCellTaskOffset,5));
    check(cudaMalloc(&s->cellScanTempStorage,s->cellScanTempBytes));
    return s;
}

void populate(DeviceState*s,TestDirectory kind,const int counts[4]){
    int base=0,injected=0,n=0;
    selectDirectory(s,kind);
    s->cellParticleOffset[0]=s->preBaseCellOffset[0]=0;
    for(int c=0;c<4;c++){
        const int b=kind==TestDirectory::full?0:(kind==TestDirectory::baseOnly?counts[c]:counts[c]/2);
        base+=b;injected+=counts[c]-b;
        s->preBaseCellOffset[c+1]=base;s->cellParticleOffset[c+1]=injected;n+=counts[c];
        s->momRhoP[c]=20;s->momRhoEP[c]=30;s->momRhoPD[c]=.02;s->thetaDragAlpha[c]=1;
    }
    *s->particleCountDevice=n;*s->preBaseParticleCountDevice=base;
    for(int c=0;c<4;c++){
        for(int i=s->preBaseCellOffset[c];i<s->preBaseCellOffset[c+1];i++)s->pCellId[i]=c;
        for(int j=s->cellParticleOffset[c];j<s->cellParticleOffset[c+1];j++){int i=base+j;s->sortedParticleIndex[j]=i;s->pCellId[i]=c;}
    }
    for(int i=0;i<n;i++){
        s->pStatus[i]=1;s->pRng[i]=123456789ULL+i*333ULL;
        s->pm[i]=1+(i%3)*.25;s->pux[i]=(i%17-8)*.125;s->puy[i]=(i%7-3)*.25;s->puz[i]=(i%5-2)*.25;
        s->pTheta[i]=.75;s->pd[i]=.001;s->pT[i]=1000;
    }
    *s->deviceState=*s;
}

void validateTasks(DeviceState*s,TestDirectory kind,int expectedTasks,int expectedMulti){
    printf("TASKS count=%d multi=%d expected=%d/%d\n",*s->csrHeavyTaskCount,*s->csrHeavyCellCount,expectedTasks,expectedMulti);
    require(*s->csrHeavyTaskCount==expectedTasks,"auto must publish current directory task count");
    require(*s->csrHeavyCellCount==expectedMulti,"auto must preserve multi-task cell count");
    require(*s->csrHeavyCellCount<=s->nCells,"finalizer count exceeds allocated cell list");
    require(s->csrTasksReady&&s->csrPreparedDirectoryKind==directoryTag(kind),"task directory is not ready for consumer");
    int countedMulti=0,total=0;
    for(int c=0;c<4;c++){
        const int tasks=s->csrCellTaskCount[c],first=s->csrCellTaskOffset[c];
        require(first==total,"task offsets must be contiguous");total+=tasks;
        int begin,end,source;
        source=expectedSource(kind);
        if(source==int(CsrReductionTaskSource::fullIndexed)){begin=s->cellParticleOffset[c];end=s->cellParticleOffset[c+1];}
        else if(source==int(CsrReductionTaskSource::splitBaseDirect)){begin=s->preBaseCellOffset[c];end=s->preBaseCellOffset[c+1];}
        else{begin=0;end=s->preBaseCellOffset[c+1]-s->preBaseCellOffset[c]+s->cellParticleOffset[c+1]-s->cellParticleOffset[c];}
        for(int j=0;j<tasks;j++){
            auto&q=s->csrReductionTasks[first+j];
            require(q.cell==c&&q.source==source,"task cell/source mismatch");
            require(q.begin==begin&&q.end>q.begin&&q.end<=end,"task range invalid or incomplete");
            begin=q.end;
        }
        require(begin==end,"directory particles are not fully covered");
        if(tasks>1){bool found=false;for(int j=0;j<expectedMulti;j++)if(s->csrMultiTaskCellList[j]==c)found=true;require(found,"multi-task cell missing from finalizer list");++countedMulti;}
    }
    require(total==expectedTasks&&countedMulti==expectedMulti,"task metadata totals disagree");
}

void consume(DeviceState*s,DeviceState*r,TestDirectory kind,int block){
    constexpr double dt=.005;
    clearPool(s,dt);
    if(s->csrHeavyReductionEnabled)require(heavyPool(s,dt,block)==0,"actual heavy pool launch");
    else require(lightPool(s,kind,dt,block)==0,"actual light launch");
    clearPool(r,dt);
    require(lightPool(r,kind,dt,block)==0,"reference light launch");
    finish();
    const int n=*s->particleCountDevice;int selected=0;
    for(int i=0;i<n;i++){
        require(s->pStatus[i]==r->pStatus[i],"L1/L2 must select identical particles");
        require(s->pRng[i]==r->pRng[i],"L1/L2 RNG stream mismatch");
        selected+=s->pStatus[i]==2;
    }
    if(n>0)require(selected>0&&selected<n,"fixture must exercise nonempty partial collision selection");
    for(int c=0;c<4;c++){
        double sum[7]={};int count=0;
        for(int i=0;i<n;i++)if(s->pStatus[i]==2&&s->pCellId[i]==c){
            ++count;double m=s->pm[i],ux=s->pux[i],uy=s->puy[i],uz=s->puz[i];
            sum[0]+=m;sum[1]+=m*ux;sum[2]+=m*uy;sum[3]+=m*uz;
            sum[4]+=m*(.5*(ux*ux+uy*uy+uz*uz)+1.5*s->pTheta[i]);sum[5]+=m*s->pd[i];sum[6]+=m*s->pd[i]*s->pd[i];
        }
        require(s->poolThermalCount[c]==count&&r->poolThermalCount[c]==count,"published collision pool count mismatch");
        double got[]={s->poissonPoolMass[c],s->poissonPoolMomX[c],s->poissonPoolMomY[c],s->poissonPoolMomZ[c],s->poissonPoolEnergy[c],s->poissonPoolDiameter[c],s->poissonPoolDiameter2[c]};
        double ref[]={r->poissonPoolMass[c],r->poissonPoolMomX[c],r->poissonPoolMomY[c],r->poissonPoolMomZ[c],r->poissonPoolEnergy[c],r->poissonPoolDiameter[c],r->poissonPoolDiameter2[c]};
        for(int j=0;j<7;j++){require(std::isfinite(got[j]),"nonfinite pool moment");require(std::fabs(got[j]-sum[j])<=tolerance*(1+std::fabs(sum[j])),"CPU collision pool moments disagree");require(std::fabs(got[j]-ref[j])<=tolerance*(1+std::fabs(ref[j])),"actual L1/L2 collision pool moments disagree");}
    }
    printf("PASS actual collision pool, CPU moments and L1/RNG parity N=%d selected=%d\n",n,selected);
}

int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    int directory=argc>1?atoi(argv[1]):0,prebuilt=argc>2?atoi(argv[2]):1,block=argc>3?atoi(argv[3]):32;
    TestDirectory kind=directory==0?TestDirectory::full:(directory==1?TestDirectory::baseOnly:TestDirectory::splitBaseAndInjection);
    DeviceState*s=make(block),*r=make(block);
    s->csrHeavyReductionMode=2;s->csrHeavyReductionEnabled=prebuilt;s->csrHeavyAutoInterval=3;
    const int table[9][4]={{block+1,block,block,block-1},{block+3,block+2,block-2,block-3},
        {3*block+4,2*block-12,2*block-12,block+20},{block,block,block,block},{0,0,0,0},
        {block,block,block,block},{block+1,block,block,block-1},{block+1,block,block,block-1},{block+1,block,block,block-1}};
    // Tile-changing row totals 8*block and has exactly two tasks in cell zero.
    // These are hand-derived expected task/multi-cell totals, not builder output.
    const int expectedTasks[]={5,6,5,4,0,0,0,0,5};
    const int expectedMulti[]={1,2,1,0,0,0,0,0,1};
    for(int step=0;step<9;step++){
        populate(s,kind,table[step]);populate(r,kind,table[step]);
        require(prepareTasks(s,block,kind)==0,"production directory task preparation");finish();
        printf("STEP %d directory=%d previous=%d\n",step+1,directory,s->csrHeavyReductionEnabled);
        require(decide(s,block,kind)==0,"automatic decision status");finish();
        const bool heavy=step<5||step==8;
        require(s->csrHeavyReductionEnabled==int(heavy)&&s->deviceState->csrHeavyReductionEnabled==int(heavy),"automatic enabled flag mismatch");
        if(heavy)validateTasks(s,kind,expectedTasks[step],expectedMulti[step]);
        consume(s,r,kind,block);
    }
    s->csrHeavyReductionMode=1;s->csrHeavyReductionEnabled=1;
    populate(s,kind,table[0]);populate(r,kind,table[0]);
    require(prepareTasks(s,block,kind)==0,"fixed L2 tasks");finish();
    auto advance=s->schedulingAdvanceCount;
    require(decide(s,block,kind)==0,"fixed L2 decision");finish();
    require(s->schedulingAdvanceCount==advance&&s->csrHeavyReductionEnabled==1,"fixed L2 must bypass auto");
    validateTasks(s,kind,5,1);consume(s,r,kind,block);
    s->csrHeavyReductionMode=2;s->csrHeavyAutoInterval=1;
    for(auto changed:{TestDirectory::full,TestDirectory::baseOnly,TestDirectory::splitBaseAndInjection}){
        populate(s,changed,table[0]);populate(r,changed,table[0]);
        require(prepareTasks(s,block,changed)==0,"changed directory producer");finish();
        require(decide(s,block,changed)==0,"changed directory auto decision");finish();
        validateTasks(s,changed,5,1);consume(s,r,changed,block);
    }
    puts("PASS gas automatic scheduling followed by actual task consumers");return 0;
}
