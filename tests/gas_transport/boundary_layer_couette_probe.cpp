// Host-only sequential replay of the production operators; not a CUDA runtime test.
// Regression origin: reviewed common tree 040bf096d79755808bda04f1e2cca7a81421d901
// rejects Euler step 1378, cell 0 (InvalidThermodynamics), normal velocity
// 415.2714529614 m/s. SSPRK3 also rejects step 1378: changing integrators does
// not repair the nonlocal matching-pressure force. Restoring only local p*n
// reaches t=.05 with the original N128/.4 Co/.25 diffusion/growth1.05 settings.
// BVP matching pressure remains the second-cell donor pressure in walls().
#include "gasWall/WallModel.H"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
void check(State&s,const char*stage,int step){
 for(int c=0;c<s.nCells;c++)if(s.gasSpecies.cellStatus[c]){fprintf(stderr,"FAIL stage=%s step=%d cell=%d code=%d rho=%.17g T=%.17g p=%.17g U=%.17g,%.17g E=%.17g\n",stage,step,c,s.gasSpecies.cellStatus[c],s.rho[c],s.Tgas[c],s.p[c],s.Ux[c],s.Uy[c],s.rhoE[c]);exit(3);}
 for(int f=0;f<s.nFaces;f++)if(s.gasSpecies.faceStatus[f]){fprintf(stderr,"FAIL stage=%s step=%d face=%d own=%d nei=%d kind=%d code=%d\n",stage,step,f,s.faceOwner[f],s.faceNeighbour[f],s.riemannBoundaryKind[f],s.gasSpecies.faceStatus[f]);exit(4);}
}
void setup(State&s,int n){initialise(s);s.nCells=n;s.nInternalFaces=n-1;s.nFaces=5*n+1;s.gasMu=.1;s.gasPrClamped=.72;s.gasThermalConductivity=-1;s.maxDiffusionNumber=.25;
 auto*m=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);auto*a=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
 for(int k=0;k<2;++k){m[k].molarMass=.0289702530249;m[k].maxTemperature=4000;a[3*k]=1004.5;a[3*k+1]=a[3*k+2]=0;}
 const double h=1./n,A=1e-4;
 for(int c=0;c<n;c++){s.V[c]=A*h;s.Cx[c]=s.Cz[c]=.005;s.Cy[c]=(c+.5)*h;s.rho[c]=1;s.rhoUx[c]=s.rhoUy[c]=s.rhoUz[c]=0;s.gasSpecies.rho[c]=s.gasSpecies.rho[n+c]=.5;Real rr[2]={.5,.5};s.rhoE[c]=ugkwp::mixtureEnergy(rr,Real(300),s.gasSpecies.thermo);s.cellPlaneStart[c]=6*c;s.cellPlaneCount[c]=6;
  int*fs=s.cellFaceId+6*c;fs[0]=c?c-1:n-1;fs[1]=c+1<n?c:n;for(int j=0;j<4;j++)fs[j+2]=n+1+4*c+j;
  threadIdx.x=c;recoverGasPrimitivesKernel(&s);
 }
 for(int f=0;f<s.nFaces;f++){s.faceOwner[f]=f<n-1?f:f==n-1?0:f==n?n-1:(f-n-1)/4;s.faceNeighbour[f]=f<n-1?f+1:-1;s.facePeriodicPair[f]=-1;s.faceWeight[f]=.5;s.Sfx[f]=s.Sfy[f]=s.Sfz[f]=0;s.faceCx[f]=s.faceCz[f]=.005;s.faceCy[f]=f<n-1?(f+1)*h:f==n-1?0:f==n?1:s.Cy[s.faceOwner[f]];s.magSf[f]=A;s.deltaCoeffs[f]=f<n-1?1/h:2/h;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f<n-1?0:f<=n?2:3;
  if(f<=n)s.Sfy[f]=f==n-1?-A:A;
  else {int j=(f-n-1)%4;if(j<2){s.Sfx[f]=j? .01*h:-.01*h;s.faceCx[f]=j?.01:0;s.facePeriodicDx[f]=j?.01:-.01;s.facePeriodicPair[f]=j?f-1:f+1;s.faceNeighbour[f]=s.faceOwner[f];s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=0;}else{s.Sfz[f]=j==2?-.01*h:.01*h;s.faceCz[f]=j==2?0:.01;}s.magSf[f]=.01*h;s.deltaCoeffs[f]=100;}
  if(f==n-1||f==n){s.gasBoundaryUFix[f]=s.riemannBoundaryUFix[f]=1;s.gasBoundaryUx[f]=s.riemannBoundaryUx[f]=f==n?1:0;s.gasBoundaryTFix[f]=s.riemannBoundaryTFix[f]=1;s.gasBoundaryT[f]=s.riemannBoundaryT[f]=300;}
  threadIdx.x=f;updateLegacyGasBoundaryMirrorKernel(&s,0);updateRiemannBoundaryMirrorKernel(&s);
 }
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=2;w.faceSlot=new int[s.nFaces];w.ownerSlot=new int[n];for(int f=0;f<s.nFaces;f++)const_cast<int*>(w.faceSlot)[f]=-1;for(int c=0;c<n;c++)const_cast<int*>(w.ownerSlot)[c]=-1;const_cast<int*>(w.faceSlot)[n-1]=0;const_cast<int*>(w.faceSlot)[n]=1;const_cast<int*>(w.ownerSlot)[0]=0;const_cast<int*>(w.ownerSlot)[n-1]=1;w.status=new int[2]{};w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[2];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[2];w.speciesFlux=new Real[4]{};
}
void walls(State&s){for(int slot=0;slot<2;slot++){int own=slot?s.nCells-1:0,donor=slot?s.nCells-2:1;ugkwp::gaswall::WallInput<Real,2> in;in.temperature=300;in.pressure=s.p[donor];in.matchingDistance=1.5/s.nCells;in.ownerDistance=.5/s.nCells;in.velocity[0]=slot?1:0;in.normal[1]=slot?-1:1;in.matching.temperature=s.Tgas[donor];in.matching.velocity[0]=s.Ux[donor];in.matching.velocity[1]=s.Uy[donor];in.matching.velocity[2]=s.Uz[donor];for(int j=0;j<2;j++)in.matching.massFraction[j]=s.gasSpecies.rho[j*s.nCells+donor]/s.rho[donor];in.model.thermo=s.gasSpecies.thermo;in.model.mode=s.gasSpecies.mode;in.model.viscosity=s.gasMu;in.model.conductivity=s.gasMu*1004.5/.72;in.quadrature.volume=s.V[own];ugkwp::gaswall::WallOutput<Real,2> out;ugkwp::gaswall::WallStatus status;ugkwp::gaswall::WallModelConfig<Real> cfg;cfg.family=ugkwp::gaswall::WallFamily::BoundaryLayer;cfg.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;ck(ugkwp::gaswall::evaluateConstantTransportWall(in,cfg,out,status),"wall solve failed");ck(ugkwp::publishGasBoundaryLayerOutput(s,slot,in,out,Real(1e-4),Real(0)),"publish failed");}}

void euler(State&s,double dt,int step){
 for(int c=0;c<s.nCells;c++){threadIdx.x=c;recoverGasPrimitivesKernel(&s);}check(s,"recover",step);
 for(int f=0;f<s.nFaces;f++){threadIdx.x=f;updateLegacyGasBoundaryMirrorKernel(&s,0);updateRiemannBoundaryMirrorKernel(&s);}check(s,"boundary",step);walls(s);check(s,"wall",step);
 for(int c=0;c<s.nCells;c++){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);}check(s,"gradient",step);
 for(int c=0;c<s.nCells;c++){threadIdx.x=c;computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}check(s,"limiter",step);
 for(int f=0;f<s.nFaces;f++){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,dt);}check(s,"flux",step);
 for(int f=0;f<s.nFaces;f++){threadIdx.x=f;enforcePeriodicGasFluxAntisymmetryKernel(&s);}check(s,"periodic",step);
 for(int c=0;c<s.nCells;c++){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}check(s,"scale",step);
 for(int f=0;f<s.nFaces;f++){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);}check(s,"applyscale",step);
 for(int c=0;c<s.nCells;c++){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);}check(s,"update",step);
}
int main(int argc,char**argv){State s;setup(s,128);s.gasSpecies.thermoControls={};s.gasFluxScheme=2;s.gasReconstruction=1;s.gasLimiter=0;
 int rk=argc>1?atoi(argv[1]):1,step=0;double dt=1e-5,time=0,minDt=1e9,maxDt=0,maxCo=0,maxDiff=0;
 while(time<.05){
  for(int f=0;f<s.nFaces;f++){threadIdx.x=f;computeGasCourantFieldKernel(&s,dt);}check(s,"courant",step);
  double co=0,diff=0;for(int c=0;c<s.nCells;c++){threadIdx.x=c;computeGasConvectiveCourantByCellKernel(&s,dt);computeGasDiffusionNumberKernel(&s,dt,.4);co=fmax(co,s.gasFluxPositivityScale[c]);diff=fmax(diff,s.gasDiffusionNumber[c]);}
  double next=fmin(dt*.4/fmax(co,diff),fmin(1e-4,dt*1.05));next=fmin(next,.05-time);maxCo=fmax(maxCo,co*next/dt);maxDiff=fmax(maxDiff,diff*next/dt*.25/.4);dt=next;minDt=fmin(minDt,dt);maxDt=fmax(maxDt,dt);
  if(step%1000==0){double maxV=0,loP=1e300,hiP=0;for(int c=0;c<s.nCells;c++){maxV=fmax(maxV,fabs(s.Uy[c]));loP=fmin(loP,s.p[c]);hiP=fmax(hiP,s.p[c]);}fprintf(stderr,"TRACE rk=%d step=%d time=%.12g dt=%.12g maxV=%.12g pRange=%.12g,%.12g\n",rk,step,time,dt,maxV,loP,hiP);}
  if(rk>1)for(int c=0;c<s.nCells;c++){threadIdx.x=c;saveGasConservativeStateKernel(&s);}
  euler(s,dt,step);
  if(rk>1){euler(s,dt,step);for(int c=0;c<s.nCells;c++){threadIdx.x=c;blendGasConservativeStateKernel(&s,rk==2?.5:.75,rk==2?.5:.25);recoverGasPrimitivesKernel(&s);}check(s,"blend",step);}
  if(rk==3){euler(s,dt,step);for(int c=0;c<s.nCells;c++){threadIdx.x=c;blendGasConservativeStateKernel(&s,1./3.,2./3.);recoverGasPrimitivesKernel(&s);}check(s,"blend3",step);}
  time+=dt;step++;if(step>200000){fprintf(stderr,"TOO MANY time=%g\n",time);return 5;}
 }
 printf("PASS rk=%d steps=%d time=%.17g dtRange=%.12g,%.12g maxCo=%.12g maxDiff=%.12g\n",rk,step,time,minDt,maxDt,maxCo,maxDiff);
 for(int c=0;c<s.nCells;c++)printf("DATA %.17g %.17g %.17g %.17g\n",s.Cy[c],s.Ux[c],s.Uy[c],s.p[c]);
}
