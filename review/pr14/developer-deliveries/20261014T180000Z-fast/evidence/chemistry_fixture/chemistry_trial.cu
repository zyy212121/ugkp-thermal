// Actual native CUDA owner, kernels, storage, rollback and validation policy.
// Host code only supplies inputs, observes returned results and compares fields.
#include "GpuResidentStrict.cu"
#include <new>
#include <cstring>
#include <vector>

namespace chemistryTrialTest {
using namespace ugkwp;
std::vector<void*> allocated;
std::vector<DeviceState*> fixtures;
void check(bool value,const char* message) {
    if(!value){std::fprintf(stderr,"FAIL %s: %s\n",message,lastError);std::exit(11);}
}
void cudaCheck(cudaError_t error) {
    if(error!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(error));std::exit(12);}
}
void sync(){cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());}
template<class T> void testMem(T*& pointer,int count=8) {
    cudaCheck(cudaMallocManaged(&pointer,std::size_t(count)*sizeof(T)));
    allocated.push_back(pointer);
    for(int i=0;i<count;++i)::new(static_cast<void*>(pointer+i)) T{};
}
template<class T> void testTable(const T*& pointer,const T* input,int count) {
    T* data=nullptr;testMem(data,count);std::memcpy(data,input,std::size_t(count)*sizeof(T));pointer=data;
}
DeviceState* fixture() {
    static_assert(ugkwp::compiledGasSpecies==2,"fixture uses a two-species mechanism");
    // The native backend keeps its canonical launch/owner state on the host.
    // A managed host owner aliases device memory on limited-UVM systems (WSL)
    // and cannot be read between an asynchronous kernel launch and its sync.
    DeviceState* s=new DeviceState;fixtures.push_back(s);
    cudaCheck(cudaMalloc(&s->deviceState,sizeof(DeviceState)));
    allocated.push_back(s->deviceState);
    testMem(s->csrMaximumOccupancy);
    testMem(s->csrHeavyTaskCount);
    testMem(s->csrHeavyCellCount);
    testMem(s->csrHeavyTaskCursor);
    testMem(s->particleCountDevice);
    testMem(s->faceOwner);
    testMem(s->faceNeighbour);
    testMem(s->facePeriodicPair);
    testMem(s->facePeriodicDx);
    testMem(s->facePeriodicDy);
    testMem(s->facePeriodicDz);
    testMem(s->V);
    testMem(s->Cx);
    testMem(s->Cy);
    testMem(s->Cz);
    testMem(s->faceCx);
    testMem(s->faceCy);
    testMem(s->faceCz);
    testMem(s->Sfx);
    testMem(s->Sfy);
    testMem(s->Sfz);
    testMem(s->magSf);
    testMem(s->deltaCoeffs);
    testMem(s->faceWeight);
    testMem(s->cellLength);
    testMem(s->cellPlaneStart);
    testMem(s->cellPlaneCount);
    testMem(s->cellFaceId);
    testMem(s->cellFaceNeighbor);
    testMem(s->cellFaceKind);
    testMem(s->cellFaceRestitution);
    testMem(s->cellFaceTangential);
    testMem(s->planeNx);
    testMem(s->planeNy);
    testMem(s->planeNz);
    testMem(s->planeD);
    testMem(s->rho);
    testMem(s->rhoUx);
    testMem(s->rhoUy);
    testMem(s->rhoUz);
    testMem(s->rhoE);
    testMem(s->rhoNext);
    testMem(s->rhoUxNext);
    testMem(s->rhoUyNext);
    testMem(s->rhoUzNext);
    testMem(s->rhoENext);
    testMem(s->gasPhiRho);
    testMem(s->gasPhiRhoUx);
    testMem(s->gasPhiRhoUy);
    testMem(s->gasPhiRhoUz);
    testMem(s->gasPhiRhoE);
    testMem(s->gasFluxPositivityScale);
    testMem(s->gasHllcAdcSensor);
    testMem(s->gasBoundaryKind);
    testMem(s->gasBoundaryRhoFix);
    testMem(s->gasBoundaryUFix);
    testMem(s->gasBoundaryPFix);
    testMem(s->gasBoundaryTFix);
    testMem(s->gasBoundaryPWave);
    testMem(s->gasBoundaryPWaveGamma);
    testMem(s->gasBoundaryPWaveFieldInf);
    testMem(s->gasBoundaryPWaveLInf);
    testMem(s->gasBoundaryRho);
    testMem(s->gasBoundaryUx);
    testMem(s->gasBoundaryUy);
    testMem(s->gasBoundaryUz);
    testMem(s->gasBoundaryP);
    testMem(s->gasBoundaryT);
    testMem(s->riemannBoundaryKind);
    testMem(s->riemannBoundaryRhoFix);
    testMem(s->riemannBoundaryUFix);
    testMem(s->riemannBoundaryPFix);
    testMem(s->riemannBoundaryTFix);
    testMem(s->riemannBoundaryPWave);
    testMem(s->riemannBoundaryPWaveGamma);
    testMem(s->riemannBoundaryPWaveFieldInf);
    testMem(s->riemannBoundaryPWaveLInf);
    testMem(s->riemannBoundaryRho);
    testMem(s->riemannBoundaryUx);
    testMem(s->riemannBoundaryUy);
    testMem(s->riemannBoundaryUz);
    testMem(s->riemannBoundaryP);
    testMem(s->riemannBoundaryT);
    testMem(s->scheduledInletFaceMask);
    testMem(s->pressureScheduleTimes);
    testMem(s->pressureScheduleValues);
    testMem(s->volumeFractionScheduleTimes);
    testMem(s->volumeFractionScheduleValues);
    testMem(s->gradPx);
    testMem(s->gradPy);
    testMem(s->gradPz);
    testMem(s->gradRhoX);
    testMem(s->gradRhoY);
    testMem(s->gradRhoZ);
    testMem(s->gradUxX);
    testMem(s->gradUxY);
    testMem(s->gradUxZ);
    testMem(s->gradUyX);
    testMem(s->gradUyY);
    testMem(s->gradUyZ);
    testMem(s->gradUzX);
    testMem(s->gradUzY);
    testMem(s->gradUzZ);
    testMem(s->gradTX);
    testMem(s->gradTY);
    testMem(s->gradTZ);
    testMem(s->gasGradientLimiterRho);
    testMem(s->gasGradientLimiterUx);
    testMem(s->gasGradientLimiterUy);
    testMem(s->gasGradientLimiterUz);
    testMem(s->gasGradientLimiterP);
    testMem(s->gasGradientLimiterT);
    testMem(s->nut);
    testMem(s->gasDiffusionNumber);
    testMem(s->rhoK);
    testMem(s->rhoOmega);
    testMem(s->rhoKInitial);
    testMem(s->rhoOmegaInitial);
    testMem(s->k);
    testMem(s->omega);
    testMem(s->sstWallDistance);
    testMem(s->sstF1);
    testMem(s->sstF2);
    testMem(s->gradKX);
    testMem(s->gradKY);
    testMem(s->gradKZ);
    testMem(s->gradOmegaX);
    testMem(s->gradOmegaY);
    testMem(s->gradOmegaZ);
    testMem(s->sstPhiRhoK);
    testMem(s->sstPhiRhoOmega);
    testMem(s->sstBoundaryKMode);
    testMem(s->sstBoundaryOmegaMode);
    testMem(s->sstBoundaryK);
    testMem(s->sstBoundaryOmega);
    testMem(s->sstSourceNumber);
    testMem(s->Ux);
    testMem(s->Uy);
    testMem(s->Uz);
    testMem(s->p);
    testMem(s->Tgas);
    testMem(s->couplingRhoOld);
    testMem(s->couplingUxOld);
    testMem(s->couplingUyOld);
    testMem(s->couplingUzOld);
    testMem(s->couplingTgasOld);
    testMem(s->epsS);
    testMem(s->rhoUsx);
    testMem(s->rhoUsy);
    testMem(s->rhoUsz);
    testMem(s->rhoEs);
    testMem(s->rhoDs);
    testMem(s->rhoHp);
    testMem(s->Usx);
    testMem(s->Usy);
    testMem(s->Usz);
    testMem(s->theta);
    testMem(s->Tp);
    testMem(s->dMeanCell);
    testMem(s->epsGPrev);
    testMem(s->collisionalPressure);
    testMem(s->pressureKickScale);
    testMem(s->pressureDeltaMomX);
    testMem(s->pressureDeltaMomY);
    testMem(s->pressureDeltaMomZ);
    testMem(s->pressureDeltaEnergy);
    testMem(s->solidPressurePhiMomX);
    testMem(s->solidPressurePhiMomY);
    testMem(s->solidPressurePhiMomZ);
    testMem(s->solidPressurePhiEnergy);
    testMem(s->mobilePackingRho);
    testMem(s->packingStuckRho);
    testMem(s->mobilePackingMomX);
    testMem(s->mobilePackingMomY);
    testMem(s->mobilePackingMomZ);
    testMem(s->mobilePackingActiveCellMask);
    testMem(s->mobilePackingCorrectionCellMask);
    testMem(s->mobilePackingActiveCellList);
    testMem(s->mobilePackingCorrectionCellList);
    testMem(s->mobilePackingFrontierCurrent);
    testMem(s->mobilePackingFrontierNext);
    testMem(s->mobilePackingActiveCellCount);
    testMem(s->mobilePackingCorrectionCellCount);
    testMem(s->mobilePackingFrontierCurrentCount);
    testMem(s->mobilePackingFrontierNextCount);
    testMem(s->thetaDragAlpha);
    testMem(s->momRhoP);
    testMem(s->momRhoUPx);
    testMem(s->momRhoUPy);
    testMem(s->momRhoUPz);
    testMem(s->momRhoEP);
    testMem(s->momRhoPD);
    testMem(s->momRhoHpP);
    testMem(s->poolThermalCount);
    testMem(s->poolThermalSumUx);
    testMem(s->poolThermalSumUy);
    testMem(s->poolThermalSumUz);
    testMem(s->poolThermalSumU2);
    testMem(s->poissonPoolSampleTargetCount);
    testMem(s->poissonPoolMass);
    testMem(s->poissonPoolMomX);
    testMem(s->poissonPoolMomY);
    testMem(s->poissonPoolMomZ);
    testMem(s->poissonPoolEnergy);
    testMem(s->poissonPoolDiameter);
    testMem(s->poissonPoolDiameter2);
    testMem(s->px);
    testMem(s->py);
    testMem(s->pz);
    testMem(s->pux);
    testMem(s->puy);
    testMem(s->puz);
    testMem(s->puxOld);
    testMem(s->puyOld);
    testMem(s->puzOld);
    testMem(s->pT);
    testMem(s->pTheta);
    testMem(s->pd);
    testMem(s->pm);
    testMem(s->pCellId);
    testMem(s->pStatus);
    testMem(s->pRng);
    testMem(s->pOrigId);
    testMem(s->sortedParticleIndex);
    testMem(s->cellParticleCount);
    testMem(s->cellParticleOffset);
    testMem(s->preBaseCellOffset);
    testMem(s->preBaseParticleCountDevice);
    testMem(s->compactCellOffset);
    testMem(s->cellParticleWrite);
    testMem(s->cellScanTempStorage);
    testMem(s->compactCountDevice);
    testMem(s->compactSelectTempStorage);
    testMem(s->csrCellTaskCount);
    testMem(s->csrCellTaskOffset);
    testMem(s->csrMultiTaskCellList);
    testMem(s->csrHeavyCellList);
    testMem(s->csrHeavyTaskCell);
    testMem(s->csrHeavyTaskBegin);
    testMem(s->csrHeavyTaskEnd);
    testMem(s->csrHeavyCellTaskStart);
    testMem(s->csrHeavyCellTaskCount);
    testMem(s->csrHeavyPartials);
    testMem(s->preInjectionHeavyTaskCount);
    testMem(s->preInjectionHeavyTaskCell);
    testMem(s->preInjectionHeavyTaskBegin);
    testMem(s->preInjectionHeavyTaskEnd);
    testMem(s->preInjectionHeavyCellTaskStart);
    testMem(s->preInjectionHeavyCellTaskCount);
    testMem(s->preInjectionHeavyPartials);
    testMem(s->sourceCell);
    testMem(s->sourceFace);
    testMem(s->sourcePx);
    testMem(s->sourcePy);
    testMem(s->sourcePz);
    testMem(s->sourceUx);
    testMem(s->sourceUy);
    testMem(s->sourceUz);
    testMem(s->sourceT);
    testMem(s->sourceTheta);
    testMem(s->sourceD);
    testMem(s->sourceMassRate);
    testMem(s->sourceResidualMass);
    testMem(s->compactPx);
    testMem(s->compactPy);
    testMem(s->compactPz);
    testMem(s->compactPux);
    testMem(s->compactPuy);
    testMem(s->compactPuz);
    testMem(s->compactPT);
    testMem(s->compactPTheta);
    testMem(s->compactPd);
    testMem(s->compactPm);
    testMem(s->compactPCellId);
    testMem(s->compactPStatus);
    testMem(s->compactPRng);
    testMem(s->compactPOrigId);
    testMem(s->poissonCellCollisionProbability);
    s->nCells=2;s->nFaces=3;s->nInternalFaces=1;
    s->fixedCellBlockThreads=s->fixedFaceBlockThreads=32;
    s->gasTimeIntegrator=s->hostGasTimeIntegrator=1;
    s->gasFluxScheme=s->hostGasFluxScheme=1;s->gasReconstruction=s->hostGasReconstruction=0;
    s->gasLimiter=s->hostGasLimiter=0;s->hostTurbulenceModel=s->turbulenceModel=0;
    s->gammaGas=1.4;s->Rgas=universalGasConstant<double>()/.028;s->gasCp=1000;
    s->gasMu=0;s->gasPrClamped=.7;s->rhoMin=1e-12;s->TgasMin=1e-9;s->maxDiffusionNumber=.25;
    s->sstCoefficients=defaultSstCoefficients();s->sstKMin=1e-12;s->sstOmegaMin=1e-9;
    s->sstMaxSourceNumber=.25;s->turbulentPrandtl=.9;
    auto& g=s->gasSpecies;g.mode=GasMode::MixtureChemistry;
#define CHEM_MEM(field) testMem(g.field)
    CHEM_MEM(rho);CHEM_MEM(initial);CHEM_MEM(flux);CHEM_MEM(gradX);CHEM_MEM(gradY);CHEM_MEM(gradZ);
    CHEM_MEM(limiter);CHEM_MEM(boundaryMassFraction);CHEM_MEM(compositionBoundaryFixed);
    CHEM_MEM(positivityScale);CHEM_MEM(soundSpeed);CHEM_MEM(heatCapacity);CHEM_MEM(gasConstant);
    CHEM_MEM(cellStatus);CHEM_MEM(faceStatus);CHEM_MEM(chemistryStatus);CHEM_MEM(chemistryAudit);
#undef CHEM_MEM
    SpeciesThermoData<double> species[2];double coefficients[6]={1000,0,0,1000,0,0};
    double elements[2]={1,1},basis[2]={-1,1};
    for(int k=0;k<2;++k){species[k].coefficientOffset=3*k;species[k].molarMass=.028;
        species[k].minTemperature=200;species[k].maxTemperature=4000;
        species[k].referencePressure=101325;species[k].referenceTemperature=300;
        species[k].referenceEntropy=1500;species[k].hasEntropyReference=true;}
    testTable(g.thermo.species,species,2);testTable(g.thermo.coefficients,coefficients,6);
    testTable(g.thermo.elementComposition,elements,2);g.thermo.coefficientCount=6;g.thermo.elementCount=1;
    GasReactionData<double> reaction[1];GasStoichTerm<double> reactants[1],products[1];
    reaction[0].reactantCount=reaction[0].productCount=1;reaction[0].highRate.preExponential=2;
    reactants[0].species=0;reactants[0].coefficient=1;products[0].species=1;products[0].coefficient=1;
    testTable(g.mechanism.reactions,reaction,1);testTable(g.mechanism.reactants,reactants,1);
    testTable(g.mechanism.products,products,1);testTable(g.mechanism.stoichiometricBasis,basis,2);
    g.mechanism.reactionCount=g.mechanism.reactantTermCount=g.mechanism.productTermCount=1;
    g.mechanism.independentRank=1;g.mechanism.referencePressure=101325;
    g.chemistryControls.initialStep=g.chemistryControls.maximumStep=1e-6;
    g.chemistryControls.maximumSteps=1;
    for(int c=0;c<2;++c){s->V[c]=.001;s->Cx[c]=c;s->cellLength[c]=1;
        s->rho[c]=28;s->Tgas[c]=700;s->p[c]=28*s->Rgas*700;
        s->rhoE[c]=28*(1000-s->Rgas)*700;g.rho[c]=28;
        s->cellPlaneStart[c]=2*c;s->cellPlaneCount[c]=2;s->sstWallDistance[c]=.5;
        g.chemistryAudit[c].integratedTime=99;g.chemistryAudit[c].acceptedSteps=7;}
    s->cellFaceId[0]=0;s->cellFaceId[1]=1;s->cellFaceId[2]=0;s->cellFaceId[3]=2;
    for(int f=0;f<3;++f){s->faceOwner[f]=f==2?1:0;s->faceNeighbour[f]=f==0?1:-1;
        s->facePeriodicPair[f]=-1;s->magSf[f]=1;s->Sfx[f]=f==1?-1:1;
        s->faceCx[f]=f==0?.5:f==1?-.5:1.5;s->faceWeight[f]=.5;s->deltaCoeffs[f]=f==0?1:2;
        s->gasBoundaryKind[f]=s->riemannBoundaryKind[f]=f==0?0:1;s->gasHllcAdcSensor[f]=1;}
    s->gasBudgetAudit.enabled=true;s->gasBudgetAudit.speciesCount=2;
    testMem(s->gasBudgetAudit.transport);testMem(s->gasBudgetAudit.initialTransport);
    testMem(s->gasBudgetAudit.chemistrySpecies);
    s->gasBudgetHost=new GasBudgetHostLedger;
    s->gasBudgetHost->output=std::tmpfile();check(s->gasBudgetHost->output!=nullptr,"open diagnostic ledger");
    s->gasBudgetHost->speciesCount=2;s->gasBudgetHost->elementCount=1;
    s->gasBudgetHost->elementMassFactors={1/.028,1/.028};
    s->gasBudgetHost->identity="\"fixture\":\"chemistry_trial\"";
    g.rho[7]=918273.125;g.cellStatus[7]=123456;g.chemistryAudit[7].acceptedSteps=123456;
    cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));
    return s;
}
bool sameBits(double x,double y){return std::memcmp(&x,&y,sizeof(double))==0;}
bool sameAudit(const ChemistryAudit<double,compiledGasSpecies>& x,
               const ChemistryAudit<double,compiledGasSpecies>& y) {
    // Compare semantic members, never uninitialized struct padding.
    return std::memcmp(x.speciesMassChange,y.speciesMassChange,sizeof(x.speciesMassChange))==0
        &&sameBits(x.massResidual,y.massResidual)
        &&sameBits(x.maximumElementResidual,y.maximumElementResidual)
        &&sameBits(x.energyResidual,y.energyResidual)
        &&sameBits(x.maximumLocalError,y.maximumLocalError)
        &&sameBits(x.integratedTime,y.integratedTime)
        &&x.acceptedSteps==y.acceptedSteps&&x.rejectedSteps==y.rejectedSteps
        &&x.nonlinearIterations==y.nonlinearIterations;
}
void compare(const DeviceState* left,const DeviceState* right) {
    sync();auto a=sharedGasTrialDetail::cellFields(const_cast<DeviceState*>(left));
    auto b=sharedGasTrialDetail::cellFields(const_cast<DeviceState*>(right));
    check(a.size()==b.size(),"accepted cell field count differs");
    for(std::size_t k=0;k<a.size();++k)
        check(a[k].count==b[k].count && bool(a[k].pointer)==bool(b[k].pointer) && (!a[k].pointer ||
            std::memcmp(a[k].pointer,b[k].pointer,a[k].count*sizeof(double))==0),"accepted cell fields differ");
    a=sharedGasTrialDetail::faceFields(const_cast<DeviceState*>(left));
    b=sharedGasTrialDetail::faceFields(const_cast<DeviceState*>(right));
    check(a.size()==b.size(),"accepted face field count differs");
    for(std::size_t k=0;k<a.size();++k)
        check(a[k].count==b[k].count && bool(a[k].pointer)==bool(b[k].pointer) && (!a[k].pointer ||
            std::memcmp(a[k].pointer,b[k].pointer,a[k].count*sizeof(double))==0),"accepted face fields differ");
    check(std::memcmp(left->gasSpecies.faceStatus,right->gasSpecies.faceStatus,
        left->nFaces*sizeof(int))==0,"accepted face statuses differ");
    // cellFields already includes every transport/species budget channel.
    check(std::memcmp(left->gasBudgetAudit.initialTransport,right->gasBudgetAudit.initialTransport,
        left->nCells*(left->gasBudgetAudit.speciesCount+2)*sizeof(double))==0,"initial transport budget differs");
    for(int c=0;c<2;++c){const auto& x=left->gasSpecies.chemistryAudit[c];const auto& y=right->gasSpecies.chemistryAudit[c];
        check(sameAudit(x,y),"accepted full semantic chemistry audit differs");
        const auto& sx=left->gasSpecies.chemistryStatus[c];const auto& sy=right->gasSpecies.chemistryStatus[c];
        check(sx.code==sy.code&&sx.thermoCode==sy.thermoCode&&sx.reaction==sy.reaction
            &&sx.attemptedSteps==sy.attemptedSteps&&sx.rejectedSteps==sy.rejectedSteps
            &&sx.nonlinearIterations==sy.nonlinearIterations,"accepted full semantic chemistry status differs");
        check(left->gasSpecies.cellStatus[c]==right->gasSpecies.cellStatus[c],"accepted cell status differs");
        std::printf("FULL_AUDIT c=%d mass=%a element=%a energy=%a local=%a time=%a accepted=%d rejected=%d nonlinear=%d status=%d thermo=%d reaction=%d attempted=%d statusRejected=%d statusNonlinear=%d\n",
            c,x.massResidual,x.maximumElementResidual,x.energyResidual,x.maximumLocalError,x.integratedTime,
            x.acceptedSteps,x.rejectedSteps,x.nonlinearIterations,int(sx.code),int(sx.thermoCode),
            sx.reaction,sx.attemptedSteps,sx.rejectedSteps,sx.nonlinearIterations);
        for(int k=0;k<compiledGasSpecies;++k)
            std::printf("FULL_SPECIES_AUDIT c=%d species=%d massChange=%a\n",c,k,x.speciesMassChange[k]);
    }
}
std::string semanticLedger(const DeviceState* s){
    auto* output=s->gasBudgetHost->output;std::rewind(output);
    std::string result;char row[8192];
    while(std::fgets(row,sizeof(row),output)){
        std::string line(row);
        // Host wall-clock cost is not a physical/audit equivalence field.
        // All inventories, residuals, sources, accepted times and byte counts stay.
        const auto cost=line.find("\"audit_host_seconds\":");
        if(cost!=std::string::npos){const auto end=line.find(',',cost);
            check(end!=std::string::npos,"malformed budget cost field");line.erase(cost,end-cost+1);}
        result+=line;
    }
    check(!std::ferror(output),"read accepted budget ledger");return result;
}
void compareBudget(const DeviceState* left,const DeviceState* right){
    const auto& x=*left->gasBudgetHost;const auto& y=*right->gasBudgetHost;
    check(x.initialized==y.initialized&&x.requestActive==y.requestActive&&x.sequence==y.sequence
        &&x.speciesCount==y.speciesCount&&x.elementCount==y.elementCount
        &&sameBits(x.originTime,y.originTime)&&sameBits(x.acceptedTime,y.acceptedTime)
        &&sameBits(x.requestedEnd,y.requestedEnd)&&sameBits(x.requestedDt,y.requestedDt)
        &&x.origin==y.origin&&x.cumulativeBoundary==y.cumulativeBoundary
        &&x.cumulativeChemistry==y.cumulativeChemistry&&x.volumes==y.volumes
        &&x.elementMassFactors==y.elementMassFactors
        &&x.deviceBytes==y.deviceBytes&&x.hostReadBytes==y.hostReadBytes
        &&x.identity==y.identity&&x.path==y.path,"accepted host budget state differs");
    check(semanticLedger(left)==semanticLedger(right),"accepted full semantic budget records differ");
    std::printf("OWNER_LEDGER %s",semanticLedger(left).c_str());
}
struct TracePolicy:SharedGasTrialPolicy {
    static int begins,commits,rollbacks,stepLimits,terminalStages;
    static std::vector<double> sourceDurations,acceptedDurations;
    static double currentDuration;
    static void clear(){begins=commits=rollbacks=stepLimits=terminalStages=0;
        currentDuration=0.;sourceDurations.clear();acceptedDurations.clear();}
    static int begin(DeviceState* s){++begins;return SharedGasTrialPolicy::begin(s);}
    static int validate(DeviceState* s,bool native=false){
        const int rc=SharedGasTrialPolicy::validate(s,native);
        if(rc && native && s->gasSpecies.chemistryStatus[0].code==ChemistryCode::StepLimit){
            ++stepLimits;check(retryable(s)&&continueTerminal(s),"chemistry StepLimit did not reach owner retry");}
        return rc;
    }
    static int applySources(DeviceState* s,double dt,bool native=false){
        currentDuration=dt;sourceDurations.push_back(dt);if(!native)++terminalStages;
        return SharedGasTrialPolicy::applySources(s,dt,native);
    }
    static int commit(DeviceState* s){const int rc=SharedGasTrialPolicy::commit(s);
        if(!rc){++commits;acceptedDurations.push_back(currentDuration);}return rc;}
    static void rollback(DeviceState* s){++rollbacks;SharedGasTrialPolicy::rollback(s);}
};
int TracePolicy::begins=0,TracePolicy::commits=0,TracePolicy::rollbacks=0,TracePolicy::stepLimits=0,TracePolicy::terminalStages=0;
std::vector<double> TracePolicy::sourceDurations,TracePolicy::acceptedDurations;
double TracePolicy::currentDuration=0.;
double prepareOwner(DeviceState* s,double dt,double time){
    double co=0.;check(ugkwpGpuResidentStrictComputeGasCourant(s,dt,.5,time,&co)==0,
        "real native Courant preparation");sync();
    check(s->gasTrial.targetMaxCo==.5,"original targetMaxCo changed");
    std::printf("OWNER_INPUT request=%a targetMaxCo=%a measuredCo=%a\n",dt,s->gasTrial.targetMaxCo,co);
    return co;
}
void startReference(DeviceState* reference,double requestedDt,double time){
    check(prepareGasBudgetRequest(reference,requestedDt,time)==0,"prepare reference requested budget");
    check(SharedGasTrialPolicy::beginInterval(reference)==0,"begin reference requested interval");
}
void finishReference(DeviceState* reference){
    check(SharedGasTrialPolicy::commitInterval(reference)==0,"commit reference requested interval");
}
void ownerTests(){
    const double requestedDt=4e-6,time=2.;
    auto* actual=fixture();auto* reference=fixture();
    check(prepareOwner(actual,requestedDt,time)>.5,"original requested interval lost actual CFL refinement");
    prepareOwner(reference,requestedDt,time);TracePolicy::clear();
    GasRequestedIntervalControls<double> controls;
    check(prepareGasBudgetRequest(actual,requestedDt,time)==0,"prepare full requested budget");
    check(advanceGasRequestedInterval<TracePolicy>(actual,requestedDt,time,controls)==0,"requested interval did not complete");
    std::printf("OWNER_TRACE begins=%d commits=%d rollbacks=%d stepLimits=%d terminal=%d\n",
        TracePolicy::begins,TracePolicy::commits,TracePolicy::rollbacks,TracePolicy::stepLimits,TracePolicy::terminalStages);
    check(TracePolicy::begins==TracePolicy::commits+TracePolicy::rollbacks&&TracePolicy::commits>1
        &&TracePolicy::rollbacks>0&&TracePolicy::stepLimits>0&&TracePolicy::terminalStages==0,
        "owner did not complete genuine native refinement after chemistry StepLimit");
    check(TracePolicy::acceptedDurations.size()==std::size_t(TracePolicy::commits),"accepted trace count differs");
    startReference(reference,requestedDt,time);
    double elapsed=0.;for(double dt:TracePolicy::acceptedDurations){
        std::printf("OWNER_ACCEPT dt=%a stamp=%a\n",dt,time+elapsed);
        check(dt>0.&&dt<requestedDt,"accepted owner duration did not shrink");
        check(advanceGasTrial<SharedGasTrialPolicy>(reference,dt,time+elapsed,true)==0,
            "explicit accepted native partition failed");elapsed+=dt;
    }
    check(elapsed==requestedDt,"accepted native partition did not cover exact requested interval");
    finishReference(reference);compare(actual,reference);compareBudget(actual,reference);
    check(actual->gasBudgetHost->sequence==1&&!actual->gasBudgetHost->requestActive
        &&actual->gasBudgetHost->acceptedTime==time+requestedDt,"accepted requested ledger did not cover interval");
    std::puts("PASS requested owner accepted trace: full interval, cell/face fields, semantic audits/statuses and budgets");

    actual=fixture();reference=fixture();prepareOwner(actual,requestedDt,time);prepareOwner(reference,requestedDt,time);
    TracePolicy::clear();controls.maximumAttempts=1;
    check(prepareGasBudgetRequest(actual,requestedDt,time)==0,"prepare terminal budget");
    check(advanceGasRequestedInterval<TracePolicy>(actual,requestedDt,time,controls)==0,"exhausted owner did not complete terminal interval");
    check(TracePolicy::begins==2&&TracePolicy::rollbacks==1&&TracePolicy::commits==1
        &&TracePolicy::stepLimits==1&&TracePolicy::terminalStages==1,"terminal fallback trace differs");
    check(TracePolicy::acceptedDurations.size()==1&&TracePolicy::acceptedDurations[0]==requestedDt,
        "terminal accepted duration differs");
    startReference(reference,requestedDt,time);
    check(advanceGasTrial<SharedGasTrialPolicy>(reference,requestedDt,time,false)==0,"direct public completion failed");
    finishReference(reference);compare(actual,reference);compareBudget(actual,reference);
    check(actual->gasBudgetHost->acceptedTime==time+requestedDt,"terminal requested endpoint differs");
    std::puts("PASS terminal owner completion: complete original public state, semantic audits/statuses and budgets");

    // Preserve the original full requested-owner success witness as well as the
    // direct chemistry kernel control. Both use the unchanged 1e-6 interval.
    const double successfulDt=1e-6;
    actual=fixture();reference=fixture();
    check(prepareOwner(actual,successfulDt,time)<=.5,"original success control exceeds actual CFL");
    prepareOwner(reference,successfulDt,time);TracePolicy::clear();controls={};
    check(prepareGasBudgetRequest(actual,successfulDt,time)==0,"prepare original-success budget");
    check(advanceGasRequestedInterval<TracePolicy>(actual,successfulDt,time,controls)==0,"original successful requested interval failed");
    check(TracePolicy::begins==1&&TracePolicy::commits==1&&TracePolicy::rollbacks==0
        &&TracePolicy::stepLimits==0&&TracePolicy::terminalStages==0,"original successful interval partition changed");
    check(TracePolicy::acceptedDurations.size()==1&&TracePolicy::acceptedDurations[0]==successfulDt,
        "original successful interval duration changed");
    startReference(reference,successfulDt,time);
    check(advanceGasTrial<SharedGasTrialPolicy>(reference,successfulDt,time,false)==0,"original success public control failed");
    finishReference(reference);compare(actual,reference);compareBudget(actual,reference);
    check(actual->gasBudgetHost->acceptedTime==time+successfulDt,"original successful endpoint differs");
    std::puts("PASS original successful requested interval: real Courant owner/public control, full fields/audits/statuses/budgets");

    actual=fixture();reference=fixture();
    advanceGasChemistryKernel<<<1,32>>>(actual->deviceState,successfulDt,actual->V,true);
    advanceGasChemistryKernel<<<1,32>>>(reference->deviceState,successfulDt,reference->V,false);sync();
    for(int c=0;c<2;++c)check(actual->gasSpecies.chemistryStatus[c]&&reference->gasSpecies.chemistryStatus[c],
        "original direct chemistry success control failed");
    compare(actual,reference);compareBudget(actual,reference);
    std::puts("PASS original direct chemistry success: native/public fields, full semantic audits/statuses and budget channels");
}
void failedCellTests(){
    for(int kind=0;kind<16;++kind){auto* s=fixture();auto& g=s->gasSpecies;
        g.chemistryControls.maximumRejectedSteps=0;
        if(kind==1)g.thermo.species=nullptr;
        if(kind==2)g.thermo.coefficients=nullptr;
        if(kind==3)g.mechanism.stoichiometricBasis=nullptr;
        if(kind==4)g.mechanism.reactions=nullptr;
        if(kind==5)g.mechanism.reactants=nullptr;
        if(kind==6)g.mechanism.products=nullptr;
        if(kind==7)g.thermo.coefficientCount=1; // Supplied backing remains complete.
        if(kind==8){auto* basis=const_cast<double*>(g.mechanism.stoichiometricBasis);basis[0]=basis[1]=0;}
        if(kind==9)const_cast<GasReactionData<double>*>(g.mechanism.reactions)[0].highRate.preExponential=NAN;
        // The next outer local step cap must retain this actual failed reaction.
        if(kind==10){g.mechanism.reactants=nullptr;g.chemistryControls.maximumRejectedSteps=100;}
        if(kind==11){auto& reaction=const_cast<GasReactionData<double>*>(g.mechanism.reactions)[0];
            reaction.type=GasReactionType::ThirdBody;reaction.efficiencyCount=1;g.mechanism.efficiencyCount=1;}
        if(kind==12)const_cast<SpeciesThermoData<double>*>(g.thermo.species)[0].model=static_cast<SpeciesThermoModel>(255);
        if(kind==13)g.chemistryControls.maximumNewtonIterations=0;
        if(kind==15){g.thermo.coefficientCount=1;g.mechanism.stoichiometricBasis=nullptr;}
        const double savedSpecies[4]={g.rho[0],g.rho[1],g.rho[2],g.rho[3]};
        const auto savedAudit0=g.chemistryAudit[0],savedAudit1=g.chemistryAudit[1];
        double savedBudget[4];std::memcpy(savedBudget,s->gasBudgetAudit.chemistrySpecies,sizeof(savedBudget));
        const double* volume=kind==14?nullptr:s->V;
        const double interval=kind==15?0.:2e-6;
        cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));
        advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,volume,true);sync();
        check(!g.chemistryStatus[0],"failed native cell reported success");
        check(std::memcmp(savedSpecies,g.rho,sizeof(savedSpecies))==0&&sameAudit(savedAudit0,g.chemistryAudit[0])
            &&sameAudit(savedAudit1,g.chemistryAudit[1]),"failed native cell published species/audit");
        check(std::memcmp(savedBudget,s->gasBudgetAudit.chemistrySpecies,sizeof(savedBudget))==0,
            "failed native cell published any species budget channel");
        check(SharedGasTrialPolicy::validate(s,true)!=0,"failed cell passed shared validation");
        const bool unavailable=(kind>=1&&kind<=6)||kind==10||kind==11||kind==12||kind==14;
        check((g.cellStatus[0]==int(GasTransportCode::InvalidStorage))==unavailable,"actual unavailable model classification differs");
        check(SharedGasTrialPolicy::continueTerminal(s)==!unavailable,"metadata/numerical failure lost terminal route");
        if(kind==0||kind==9)check(SharedGasTrialPolicy::retryable(s),"numerical chemistry lost native retry");
        if(unavailable)check(!SharedGasTrialPolicy::retryable(s),"missing consumed backing became retryable");
        // Keep the original native failure visible before public replay replaces it.
        std::printf("PASS failed-cell kind=%d code=%d thermo=%d retry=%d terminal=%d\n",kind,int(g.chemistryStatus[0].code),int(g.chemistryStatus[0].thermoCode),int(SharedGasTrialPolicy::retryable(s)),int(SharedGasTrialPolicy::continueTerminal(s)));
        if(kind==7||kind==15){check(g.chemistryStatus[0].code==ChemistryCode::InvalidModel,"thermo model status lost");
            check(SharedGasTrialPolicy::resetErrors(s)==0,"reset metadata control");
            advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,s->V,false);sync();
            for(int c=0;c<s->nCells;++c)
                check(g.chemistryAudit[c].integratedTime==interval&&g.cellStatus[c]!=int(GasTransportCode::InvalidStorage),"metadata-only trial failure blocked public completion");
            std::printf("PASS public-replay kind=%d code=%d thermo=%d\n",kind,int(g.chemistryStatus[0].code),int(g.chemistryStatus[0].thermoCode));}
    }
}
// Formalize the S301/S302 real-CUDA negative transaction witness. Each
// configuration gets an independent accepted-state reference and real policy
// begin/validate/rollback; diagnostics are checked before rollback replaces them.
void failedTransactionTests(){
    auto* successful=fixture();
    advanceGasChemistryKernel<<<1,32>>>(successful->deviceState,1e-6,successful->V,true);sync();
    check(bool(successful->gasSpecies.chemistryStatus[0]),"independent original direct control failed");
    const int expectedNewton=successful->gasSpecies.chemistryStatus[0].nonlinearIterations;
    for(int kind=0;kind<16;++kind){auto* s=fixture();auto& g=s->gasSpecies;
        g.chemistryControls.maximumRejectedSteps=0;
        if(kind==1)g.thermo.species=nullptr;
        if(kind==2)g.thermo.coefficients=nullptr;
        if(kind==3)g.mechanism.stoichiometricBasis=nullptr;
        if(kind==4)g.mechanism.reactions=nullptr;
        if(kind==5)g.mechanism.reactants=nullptr;
        if(kind==6)g.mechanism.products=nullptr;
        if(kind==7)g.thermo.coefficientCount=1; // Supplied backing remains complete.
        if(kind==8){auto* basis=const_cast<double*>(g.mechanism.stoichiometricBasis);basis[0]=basis[1]=0;}
        if(kind==9)const_cast<GasReactionData<double>*>(g.mechanism.reactions)[0].highRate.preExponential=NAN;
        // The next outer local step cap must retain this actual failed reaction.
        if(kind==10){g.mechanism.reactants=nullptr;g.chemistryControls.maximumRejectedSteps=100;}
        if(kind==11){auto& reaction=const_cast<GasReactionData<double>*>(g.mechanism.reactions)[0];
            reaction.type=GasReactionType::ThirdBody;reaction.efficiencyCount=1;g.mechanism.efficiencyCount=1;}
        if(kind==12)const_cast<SpeciesThermoData<double>*>(g.thermo.species)[0].model=static_cast<SpeciesThermoModel>(255);
        if(kind==13)g.chemistryControls.maximumNewtonIterations=0;
        if(kind==15){g.thermo.coefficientCount=1;g.mechanism.stoichiometricBasis=nullptr;}

        auto* reference=fixture();const double interval=kind==15?0.:2e-6;
        startReference(s,interval,2.);startReference(reference,interval,2.);
        check(SharedGasTrialPolicy::begin(s)==0,"real failed transaction begin");
        const double* volume=kind==14?nullptr:s->V;
        cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));
        advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,volume,true);sync();
        using TC=decltype(g.chemistryStatus[0].thermoCode);
        auto code=ChemistryCode::InvalidModel;auto thermo=TC::Success;
        int reaction=-1,attempted=0,rejected=0,nonlinear=0;
        if(kind==0){code=ChemistryCode::StepLimit;attempted=1;nonlinear=expectedNewton;}
        if(kind==1||kind==2||kind==7||kind==12||kind==15)thermo=TC::InvalidModel;
        if(kind==4){attempted=rejected=nonlinear=1;}
        if(kind==5||kind==6||kind==9||kind==11){code=ChemistryCode::RateFailure;reaction=0;attempted=rejected=nonlinear=1;}
        if(kind==10){code=ChemistryCode::StepLimit;reaction=0;attempted=rejected=nonlinear=1;}
        if(kind==13)code=ChemistryCode::InvalidControls;
        const bool unavailable=(kind>=1&&kind<=6)||kind==10||kind==11||kind==12||kind==14;
        const bool retry=kind==0||kind==9;
        for(int c=0;c<s->nCells;++c){const auto& q=g.chemistryStatus[c];
            std::printf("FAIL_STATUS kind=%d cell=%d code=%d thermo=%d reaction=%d attempted=%d rejected=%d nonlinear=%d cellStatus=%d expectedNewton=%d\n",
                kind,c,int(q.code),int(q.thermoCode),q.reaction,q.attemptedSteps,q.rejectedSteps,q.nonlinearIterations,g.cellStatus[c],expectedNewton);
            check(q.code==code&&q.thermoCode==thermo&&q.reaction==reaction
                &&q.attemptedSteps==attempted&&q.rejectedSteps==rejected&&q.nonlinearIterations==nonlinear,
                "full failure-specific ChemistryStatus mismatch");
            check(g.cellStatus[c]==int(unavailable?GasTransportCode::InvalidStorage:GasTransportCode::ChemistryFailure),"full failure-specific cell status");
        }
        auto fields=sharedGasTrialDetail::cellFields(s),control=sharedGasTrialDetail::cellFields(reference);
        check(fields.size()==control.size(),"failure cell field count");
        for(std::size_t k=0;k<fields.size();++k)check(fields[k].count==control[k].count
            &&bool(fields[k].pointer)==bool(control[k].pointer)&&(!fields[k].pointer
            ||std::memcmp(fields[k].pointer,control[k].pointer,fields[k].count*sizeof(double))==0),"failure leaked transaction cell/budget state");
        fields=sharedGasTrialDetail::faceFields(s);control=sharedGasTrialDetail::faceFields(reference);
        check(fields.size()==control.size(),"failure face field count");
        for(std::size_t k=0;k<fields.size();++k)check(fields[k].count==control[k].count
            &&bool(fields[k].pointer)==bool(control[k].pointer)&&(!fields[k].pointer
            ||std::memcmp(fields[k].pointer,control[k].pointer,fields[k].count*sizeof(double))==0),"failure leaked transaction face state");
        check(std::memcmp(s->gasBudgetAudit.initialTransport,reference->gasBudgetAudit.initialTransport,
            s->nCells*(s->gasBudgetAudit.speciesCount+2)*sizeof(double))==0,"failure leaked initial transport budget");
        check(std::memcmp(s->gasSpecies.faceStatus,reference->gasSpecies.faceStatus,
            s->nFaces*sizeof(int))==0,"failure changed untouched face status");
        for(int c=0;c<s->nCells;++c)check(sameAudit(g.chemistryAudit[c],reference->gasSpecies.chemistryAudit[c]),"failure leaked accepted audit");
        compareBudget(s,reference);
        check(SharedGasTrialPolicy::validate(s,true)!=0,"failure transaction validate falsely succeeds");
        check(SharedGasTrialPolicy::retryable(s)==retry&&SharedGasTrialPolicy::continueTerminal(s)==!unavailable,"failure native decision flags");
        const bool savedRetry=s->gasTrial.retryableFailure,savedTerminal=s->gasTrial.terminalContinuation;
        SharedGasTrialPolicy::rollback(s);sync();compare(s,reference);compareBudget(s,reference);
        check(s->gasTrial.retryableFailure==savedRetry&&s->gasTrial.terminalContinuation==savedTerminal
            &&!s->gasModelPoisoned,"rollback changed native decision flags or poisoned owner");
        std::printf("PASS failed-transaction kind=%d bothCellsFullStatus=1 allCellFaceBudgetBytes=1 fullSemanticAudit=1 hostLedger=1 rollbackRestored=1 retry=%d terminal=%d\n",kind,int(savedRetry),int(savedTerminal));
    }
}

void cleanup(){sync();for(auto* s:fixtures){
    check(s->gasSpecies.rho[7]==918273.125&&s->gasSpecies.cellStatus[7]==123456
        &&s->gasSpecies.chemistryAudit[7].acceptedSteps==123456,"canary overwritten");
    check(releaseSharedGasTrialStorage(s->gasTrial)==0,"snapshot release failed");
    delete s->gasBudgetHost;s->gasBudgetHost=nullptr;delete s;}
    for(auto it=allocated.rbegin();it!=allocated.rend();++it)cudaCheck(cudaFree(*it));
}
} // namespace chemistryTrialTest
int main(){setvbuf(stdout,nullptr,_IONBF,0);chemistryTrialTest::ownerTests();
    chemistryTrialTest::failedCellTests();chemistryTrialTest::failedTransactionTests();chemistryTrialTest::cleanup();
    std::puts("PASS real CUDA chemistry Trial routing, owner retry, terminal completion and unavailable-resource controls");}
