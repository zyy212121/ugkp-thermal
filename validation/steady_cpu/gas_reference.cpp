// Included after the shared host CUDA-index shim by build.py. No physics copies.
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "gasTransport/GasBuildConfig.H"
struct DeviceState:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{
 ugkwp::GasSpeciesState<Real,2> gasSpecies;
 double referenceGasBoundaryBudget[5]{};
 double *steadySampleSum=nullptr,*steadySampleSquare=nullptr,*steadySampleBegin=nullptr,*steadySampleMean=nullptr,*steadySampleScratch=nullptr;
 double* gasPseudoDt=nullptr; double* gasSteadyResidual=nullptr;
 double gasPseudoReferenceDt=1; bool gasSteadyMode=true; double gasSteadyBoundaryTime=0;
 Real gravityX=0,gravityY=0,gravityZ=0,gasPr=.7;
};
#include "applications/gasUGKPSteady/private_backend/SteadyGasKernels.cuh"
struct Input {
 int cells=32,iterations=20000,scheme=1,reconstruction=0,limiter=0,sst=0,mixture=0,local=1,outputEvery=100;
 int particleCapacity=10000,seed=12345;
 double massRate=.01,parcelMass=1e-5,particleVelocity=10,particleTemperature=300,diameter=.001,particleTheta=0;
 double wallDistance=4503599627370496.;
 double yPerturbation=0,diffusivity0=0,diffusivity1=0;
 double length=2,courant=.15,dt=1e-6,maxDt=.001,rho=1,pressure=100000,velocity=20,perturbation=.1,mu=0,k=.1,omega=10,y0=.3;
};
void readInput(const char*path,Input& a){
 std::ifstream in(path);if(!in)throw std::runtime_error("cannot open input");
 std::string key;double value;
 while(in>>key){if(key[0]=='#'){std::getline(in,key);continue;}in>>value;
#define PARAM(x) if(key==#x){a.x=value;continue;}
 PARAM(cells) PARAM(iterations) PARAM(scheme) PARAM(reconstruction) PARAM(limiter) PARAM(sst) PARAM(mixture) PARAM(local) PARAM(outputEvery)
 PARAM(particleCapacity) PARAM(seed) PARAM(massRate) PARAM(parcelMass) PARAM(particleVelocity) PARAM(particleTemperature) PARAM(diameter) PARAM(particleTheta)
 PARAM(wallDistance) PARAM(yPerturbation) PARAM(diffusivity0) PARAM(diffusivity1)
 PARAM(length) PARAM(courant) PARAM(dt) PARAM(maxDt) PARAM(rho) PARAM(pressure) PARAM(velocity) PARAM(perturbation) PARAM(mu) PARAM(k) PARAM(omega) PARAM(y0)
#undef PARAM
 throw std::runtime_error("unknown input key: "+key);
 }
}
template<class F>void cells(DeviceState&s,F fn){for(int c=0;c<s.nCells;++c){threadIdx.x=c;fn();}}
template<class F>void faces(DeviceState&s,F fn){for(int f=0;f<s.nFaces;++f){threadIdx.x=f;fn();}}
void initialise(DeviceState&s,const Input&a){
 const int n=a.cells,storage=4*n+8;
 // GENERATED_ALLOCATIONS
 s.nCells=n;s.nFaces=n+1;s.nInternalFaces=n-1;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.rhoMin=1e-12;s.TgasMin=1e-9;
 s.gasMu=a.mu;s.gasPrClamped=s.gasPr=.7;s.turbulentPrandtl=.9;s.gasFluxScheme=a.scheme;s.gasReconstruction=a.reconstruction;s.gasLimiter=a.limiter;
 s.sstConfigured=a.sst;s.turbulenceModel=a.sst?3:0;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.sstKMin=1e-12;s.sstOmegaMin=1e-9;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
 s.gasSteadyResidual=new double[9*n]{};if(a.local)s.gasPseudoDt=new double[n]{};
 auto&g=s.gasSpecies;g.cellStatus=new int[n]{};g.faceStatus=new int[n+1]{};
 if(a.mixture){
 g.mode=ugkwp::GasMode::MixtureFrozen;g.rho=new Real[2*n]{};g.initial=new Real[2*n]{};g.flux=new Real[2*(n+1)]{};g.gradX=new Real[2*n]{};g.gradY=new Real[2*n]{};g.gradZ=new Real[2*n]{};g.limiter=new Real[n]{};g.boundaryMassFraction=new Real[2*(n+1)]{};g.compositionBoundaryFixed=new int[n+1]{};g.positivityScale=new Real[2*n]{};g.soundSpeed=new Real[n]{};g.heatCapacity=new Real[n]{};g.gasConstant=new Real[n]{};g.diffusivity=new Real[2]{a.diffusivity0,a.diffusivity1};
 auto*meta=new ugkwp::SpeciesThermoData<Real>[2];g.thermo.species=meta;g.thermo.coefficients=new Real[6]{1000,0,-1e6,1000,0,2e6};g.thermo.coefficientCount=6;
 for(int j=0;j<2;++j){meta[j].molarMass=.028;meta[j].minTemperature=100;meta[j].maxTemperature=3000;meta[j].referencePressure=101325;meta[j].coefficientOffset=3*j;}
 }
 const double dx=a.length/n;
 for(int c=0;c<n;++c){
 s.Cx[c]=(c+.5)*dx;s.V[c]=dx;s.cellLength[c]=std::cbrt(dx);s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;
 s.cellFaceId[2*c]=c==0?n-1:c-1;s.cellFaceId[2*c+1]=c==n-1?n:c;
 const double disturbance=a.perturbation*std::sin(6.283185307179586*s.Cx[c]/a.length);
 s.rho[c]=a.rho*(1+disturbance);s.rhoUx[c]=s.rho[c]*a.velocity*(1-disturbance);s.rhoE[c]=a.pressure*(1+disturbance)/(s.gammaGas-1)+.5*s.rhoUx[c]*s.rhoUx[c]/s.rho[c];
 if(a.mixture){const double Y=a.y0+a.yPerturbation*std::sin(6.283185307179586*s.Cx[c]/a.length);g.rho[c]=s.rho[c]*Y;g.rho[n+c]=s.rho[c]*(1-Y);Real rhos[2]={g.rho[c],g.rho[n+c]};s.rhoE[c]=ugkwp::mixtureEnergy(rhos,Real(700)*(1+disturbance),g.thermo)+.5*s.rhoUx[c]*s.rhoUx[c]/s.rho[c];}
 s.k[c]=a.k;s.omega[c]=a.omega;s.sstWallDistance[c]=a.wallDistance;
 }
 for(int f=0;f<=n;++f){
 const bool boundary=f>=n-1;const bool left=f==n-1;
 s.faceOwner[f]=boundary?(left?0:n-1):f;s.faceNeighbour[f]=boundary?-1:f+1;s.facePeriodicPair[f]=-1;
 s.Sfx[f]=left?-1:1;s.magSf[f]=1;s.faceCx[f]=boundary?(left?0:a.length):(f+1)*dx;s.faceWeight[f]=.5;s.deltaCoeffs[f]=(boundary?2:1)/dx;s.gasHllcAdcSensor[f]=1;
 s.riemannBoundaryKind[f]=s.gasBoundaryKind[f]=0;
 s.riemannBoundaryRhoFix[f]=s.riemannBoundaryPFix[f]=s.riemannBoundaryUFix[f]=boundary;
 s.riemannBoundaryRho[f]=a.rho;s.riemannBoundaryP[f]=a.pressure;s.riemannBoundaryUx[f]=a.velocity;
 s.sstBoundaryKMode[f]=s.sstBoundaryOmegaMode[f]=1;s.sstBoundaryK[f]=a.k;s.sstBoundaryOmega[f]=a.omega;
 if(a.mixture){g.compositionBoundaryFixed[f]=boundary;g.boundaryMassFraction[f]=a.y0;g.boundaryMassFraction[n+1+f]=1-a.y0;s.riemannBoundaryP[f]=a.rho*(8314.46261815324/28)*700;}
 s.gasBoundaryP[f]=s.riemannBoundaryP[f];s.gasBoundaryRho[f]=s.riemannBoundaryRho[f];s.gasBoundaryUx[f]=s.riemannBoundaryUx[f];s.gasBoundaryT[f]=a.mixture?700:a.pressure/(a.rho*s.Rgas);
 }
 blockDim.x=1;blockIdx.x=0;
 cells(s,[&](){recoverGasPrimitivesKernel(&s);initialiseSstConservativeStateKernel(&s);});
}
void rawFlux(DeviceState&s){
 cells(s,[&](){recoverGasPrimitivesKernel(&s);if(s.sstConfigured)recoverSstPrimitivesKernel(&s);});
 cells(s,[&](){computeGasPrimitiveGradientsKernel(&s);if(s.sstConfigured)computeSstGradientsKernel(&s);});
 cells(s,[&](){computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);computeGasHllcAdcSensorKernel(&s);});
 faces(s,[&](){computeGasInternalFaceFluxKernel<true>(&s,0.);if(s.sstConfigured)computeSstFaceFluxKernel(&s);});
}
void step(DeviceState&s,const Input&a){
 cells(s,[&](){recoverGasPrimitivesKernel(&s);if(s.sstConfigured)recoverSstPrimitivesKernel(&s);});
 if(a.local){
 faces(s,[&](){computeGasCourantFieldKernel(&s,1.);});
 cells(s,[&](){computeGasConvectiveCourantByCellKernel(&s,1.);prepareSteadyGasTimeStepKernel(&s,a.courant,a.maxDt);});
 }
 rawFlux(s);const double dt=a.local?1:a.dt;
 cells(s,[&](){computeGasFluxPositivityScaleKernel(&s,dt);});faces(s,[&](){applyGasFluxPositivityScaleKernel(&s);});
 if(!a.local)for(int f=s.nInternalFaces;f<s.nFaces;++f){const double q[5]={s.gasPhiRho[f],s.gasPhiRhoUx[f],s.gasPhiRhoUy[f],s.gasPhiRhoUz[f],s.gasPhiRhoE[f]};for(int j=0;j<5;++j)s.referenceGasBoundaryBudget[j]-=dt*q[j];}
 if(s.sstConfigured){faces(s,[&](){computeSstFaceFluxKernel(&s);});cells(s,[&](){applySstFluxAndSourceKernel(&s,dt);});}
 cells(s,[&](){applyGasFluxDivergenceByCellKernel(&s,dt);});
}
void output(DeviceState&s,std::ostream&out,int iteration){
 rawFlux(s);cells(s,[&](){steadyGasSpatialResidualKernel(&s);});
 for(int c=0;c<s.nCells;++c){out<<iteration<<','<<c<<','<<s.Cx[c]<<','<<s.V[c]<<','<<s.rho[c]<<','<<s.rhoUx[c]<<','<<s.rhoUy[c]<<','<<s.rhoUz[c]<<','<<s.rhoE[c]<<','<<s.Ux[c]<<','<<s.Uy[c]<<','<<s.Uz[c]<<','<<s.p[c]<<','<<s.Tgas[c]<<','<<s.k[c]<<','<<s.omega[c]<<','<<s.nut[c]<<','<<(s.gasSpecies.rho?s.gasSpecies.rho[c]:0)<<','<<(s.gasSpecies.rho?s.gasSpecies.rho[s.nCells+c]:0)<<','<<(s.gasPseudoDt?s.gasPseudoDt[c]:0);for(int j=0;j<9;++j)out<<','<<s.gasSteadyResidual[j*s.nCells+c];out<<','<<s.gasSpecies.cellStatus[c]<<'\n';}
}
// PARTICLE_FRAGMENT
#ifdef STEADY_CPU_PARTICLES
#include "applications/gasUGKPSteady/private_backend/SteadySamplingMoments.cuh"
void outputWindow(ParticleState&s,std::ostream&out,int iteration,double duration){
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;finishSteadySampleWindowKernel(&s,duration);}
 const int size=GpuSteadySamplingReport::fields*s.nCells;
 for(int c=0;c<s.nCells;++c){out<<iteration<<','<<c;for(int k=0;k<GpuSteadySamplingReport::fields;++k){const int j=k*s.nCells+c;out<<','<<s.steadySampleMean[j]<<','<<s.steadySampleScratch[j]<<','<<s.steadySampleScratch[size+j]<<','<<s.steadySampleScratch[2*size+j]<<','<<s.steadySampleScratch[3*size+j]<<','<<s.steadySampleScratch[4*size+j];}out<<'\n';}
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;beginSteadySampleWindowKernel(&s);}
}
void outputParticles(ParticleState&s,std::ostream&out,int iteration){
 for(int c=0;c<s.nCells;++c){
 double mass=0,age=0,temperature=0,x=0;int count=0;
 for(int i=0;i<s.particleCount;++i)if(s.pStatus[i]==1&&s.pCellId[i]==c){++count;mass+=s.pm[i];age+=s.pm[i]*s.referenceAge[i];temperature+=s.pm[i]*s.pT[i];x+=s.pm[i]*s.px[i];}
 out<<iteration<<','<<c<<','<<count<<','<<mass<<','<<(mass?x/mass:0)<<','<<(mass?age/mass:0)<<','<<(mass?temperature/mass:0)<<','<<s.momRhoP[c]<<','<<s.momRhoUPx[c]<<','<<s.momRhoUPy[c]<<','<<s.momRhoUPz[c]<<','<<s.momRhoEP[c]<<','<<s.momRhoHpP[c]<<','<<s.momRhoPD[c]<<','<<s.injected<<','<<s.outflow<<','<<s.referenceMaxCollisionProbability<<','<<s.injectedMass<<','<<s.injectedMomentumX<<','<<s.injectedMechanicalEnergy<<','<<s.injectedMaterialHeat<<','<<s.outflowMass<<','<<s.outflowMomentumX<<','<<s.outflowMechanicalEnergy<<','<<s.outflowMaterialHeat<<','<<s.gasDragMomentumX<<','<<s.gasDragEnergy<<','<<s.gasMaterialHeat<<','<<s.sourceResidualMass[0]<<','<<s.gasPairedMomentumX<<','<<s.gasPairedEnergy;for(int j=0;j<5;++j)out<<','<<s.referenceGasBoundaryBudget[j];out<<'\n';
 }
}
#endif
int main(int argc,char**argv){
 if(argc!=3){std::fprintf(stderr,"usage: %s input.cfg fields.csv\n",argv[0]);return 2;}
 try{Input a;readInput(argv[1],a);
#ifdef STEADY_CPU_PARTICLES
 ParticleState s;initialise(s,a);initialiseParticles(s,a.particleCapacity);
 s.sourceMassRate[0]=a.massRate;s.injectionParcelMass=a.parcelMass;s.sourceUx[0]=a.particleVelocity;s.sourceT[0]=a.particleTemperature;s.sourceD[0]=a.diameter;s.sourceTheta[0]=a.particleTheta;s.rngSeed=a.seed;s.particleDiameterFallback=s.particleDiameterMin=s.particleDiameterMax=a.diameter;
 std::ofstream particleOut(std::string(argv[2])+".particles.csv");particleOut<<std::setprecision(17)<<"iteration,cell,count,mass,xMean,ageMeanUpperBound,TMean,momRhoP,momRhoUPx,momRhoUPy,momRhoUPz,momRhoEP,momRhoHpP,momRhoPD,injected,outflow,maxCollisionProbability,injectedMass,injectedMomentumX,injectedMechanicalEnergy,injectedMaterialHeat,outflowMass,outflowMomentumX,outflowMechanicalEnergy,outflowMaterialHeat,gasDragMomentumX,gasDragEnergy,gasMaterialHeat,sourceResidualMass,gasPairedMomentumX,gasPairedEnergy,gasBoundaryMass,gasBoundaryMomentumX,gasBoundaryMomentumY,gasBoundaryMomentumZ,gasBoundaryEnergy\n";
 outputParticles(s,particleOut,0);
 const int sampleSize=GpuSteadySamplingReport::fields*s.nCells;
 s.steadySampleSum=new double[sampleSize]{};s.steadySampleSquare=new double[sampleSize]{};s.steadySampleBegin=new double[sampleSize]{};s.steadySampleMean=new double[sampleSize]{};s.steadySampleScratch=new double[5*sampleSize]{};
 std::ofstream windowOut(std::string(argv[2])+".windows.csv");windowOut<<std::setprecision(17)<<"iteration,cell";
 for(int k=0;k<GpuSteadySamplingReport::fields;++k)windowOut<<",q"<<k<<"Mean,q"<<k<<"MeanChange,q"<<k<<"Fluctuation,q"<<k<<"InventoryBegin,q"<<k<<"InventoryEnd,q"<<k<<"CharacteristicInventory";windowOut<<'\n';
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;beginSteadySampleWindowKernel(&s);}
 int windowSteps=0;
#else
 DeviceState s;initialise(s,a);
#endif
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<"iteration,cell,x,volume,rho,rhoUx,rhoUy,rhoUz,rhoE,Ux,Uy,Uz,p,T,k,omega,nut,rhoY0,rhoY1,pseudoDt,Rrho,Rmx,Rmy,Rmz,RE,RY0,RY1,Rk,Romega,cellStatus\n";
 output(s,out,0);for(int i=1;i<=a.iterations;++i){step(s,a);
#ifdef STEADY_CPU_PARTICLES
 particleStep(s,a.dt,0.);
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;accumulateSteadySampleWindowKernel(&s,a.dt);}++windowSteps;
 if(i%a.outputEvery==0||i==a.iterations){outputWindow(s,windowOut,i,windowSteps*a.dt);windowSteps=0;}
 if(i%a.outputEvery==0||i==a.iterations)outputParticles(s,particleOut,i);
#endif
 if(i%a.outputEvery==0||i==a.iterations)output(s,out,i);}return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
