// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuLesAlgebra.cuh"
using Real=GpuReal;
using Time=double;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME Time
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_R(1e-30)
#define __host__
#define __global__
#define __device__
#define __forceinline__ inline
// PTX traps are represented by process failure in this serial host emulator.
#define asm(...) std::abort()
struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;
constexpr Real OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr Real OfVSmall=GPU_R(1e-30);
constexpr Real OfGreat=GPU_R(1.0)/OfSmall;
struct GasPrimDevice { Real rho,ux,uy,uz,p,T; };
#include "legacy_gas_fixture.hpp"
#ifdef LEGACY_REFERENCE
using DeviceState=LegacyGasFixture;
#else
// No application DeviceState definition: the actual operators must be generic.
#include "gasTransport/GasStateView.H"
#endif
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
#include "operators/computeSstGradientsKernel.cuh"
#include "operators/sstVelocityInvariants.cuh"
#include "operators/computeGasHllcAdcSensorKernel.cuh"
#include "operators/updateBarthLimiter.cuh"
#include "operators/computeGasGradientLimiterKernel.cuh"
#include "operators/computeGasEddyViscosityKernel.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "operators/gasFaceSubgridTransportProperties.cuh"
#include "operators/computeRiemannGasFaceFluxDevice.cuh"
#include "operators/computeGasInternalFaceFluxKernel.cuh"
#include "operators/computeGasFluxPositivityScaleKernel.cuh"
#include "operators/computeSstFaceFluxKernel.cuh"

#ifdef LEGACY_OPTIONAL_SPECIES
struct TestLegacyGasFixture:LegacyGasFixture{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
#else
using TestLegacyGasFixture=LegacyGasFixture;
#endif
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
namespace fs=std::filesystem;
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2>gasSpecies;};
template<class T>void fieldStorage(T&v,const fs::path&p,std::size_t count){if constexpr(std::is_pointer_v<T>){using E=std::remove_const_t<std::remove_pointer_t<T>>;auto*data=new E[count]{};v=data;if(fs::exists(p)){auto size=fs::file_size(p);if(size>count*sizeof(E)||size%sizeof(E))std::abort();std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(data),size);if(!f)std::abort();}}}
template<class T>void read(fs::path p,T*data,std::size_t n){std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(data),n*sizeof(T));if(!f){std::cerr<<"read failed "<<p<<'\n';std::abort();}}
void diagnostic(ugkp::diagnostics::DeviceContext&d,const char*where){if(d.fatalClaim){auto&r=d.firstFatal;std::cerr<<where<<" fatal code="<<int(r.code)<<" site="<<r.sourceSiteId<<" entity="<<r.entityIndex<<" value="<<r.observed<<'\n';std::exit(2);}}
int main(int argc,char**argv){
 if(argc<3)return 1;fs::path root=argv[1],output=argv[2];fs::create_directories(output);bool layer=root.filename().string().rfind("baseline-",0)==0;
 int nc,nf,ni;std::ifstream(root/"counts.txt")>>nc>>nf>>ni;State s;std::size_t cap=2*nf;
#define UGKWP_GAS_STATE_FIELD(Type,Name) fieldStorage(s.Name,root/(#Name ".bin"),cap);
#include "common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
 s.nCells=nc;s.nFaces=nf;s.nInternalFaces=ni;s.gasFluxScheme=argc>5?std::atoi(argv[5]):2;s.gasReconstruction=argc>7?std::atoi(argv[7]):1;s.gasLimiter=argc>6?std::atoi(argv[6]):1;s.gasMu=1.7244421537657099e-5;s.gasPrClamped=.72;s.gasCp=1004.675;s.Rgas=287.0500676;s.gammaGas=s.gasCp/(s.gasCp-s.Rgas);s.rhoMin=1e-12;s.TgasMin=1e-9;s.maxDiffusionNumber=.25;
 s.sstConfigured=1;s.turbulenceModel=3;s.sstWallTreatment=layer?2:1;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.sstKMin=1e-12;s.sstOmegaMin=1e-6;s.sstMaxSourceNumber=.25;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;s.turbulentPrandtl=.9;s.sstJayatillekeP=ugkpwall::jayatillekeSmoothP(.72/.9);s.sstThermalYPlus=ugkpwall::jayatillekeThermalYPlus(.72/.9);
 ugkp::diagnostics::DeviceContext diag{};s.diagnostics=&diag;CapturedSstWall properties;auto&g=s.gasSpecies;g.mode=ugkwp::GasMode::MixtureFrozen;g.thermo=properties.input.model.thermo;g.thermo.diagnostics=&diag;
 g.rho=new Real[2*nc]{};g.initial=new Real[2*nc]{};g.flux=new Real[2*nf]{};g.gradX=new Real[2*nc]{};g.gradY=new Real[2*nc]{};g.gradZ=new Real[2*nc]{};g.limiter=new Real[nc]{};g.boundaryMassFraction=new Real[2*nf]{};g.compositionBoundaryFixed=new int[nf]{};g.positivityScale=new Real[2*nc]{};g.soundSpeed=new Real[nc]{};g.heatCapacity=new Real[nc]{};g.gasConstant=new Real[nc]{};g.cellStatus=new int[nc]{};g.faceStatus=new int[nf]{};g.diffusivity=properties.input.model.diffusivity;
 read(root/"speciesRhoAir.bin",g.rho,nc);read(root/"speciesRhoCO2.bin",g.rho+nc,nc);std::copy(g.rho,g.rho+2*nc,g.initial);std::fill(g.positivityScale,g.positivityScale+2*nc,1);
 for(int c=0;c<nc;++c){Real Y[2]={g.rho[c]/s.rho[c],g.rho[nc+c]/s.rho[c]};g.heatCapacity[c]=Y[0]*1004.675+Y[1]*843.68062566560013;g.gasConstant[c]=ugkwp::mixtureGasConstant(Y,g.thermo);g.soundSpeed[c]=std::sqrt(g.heatCapacity[c]/(g.heatCapacity[c]-g.gasConstant[c])*g.gasConstant[c]*s.Tgas[c]);s.rhoK[c]=s.rho[c]*s.k[c];s.rhoOmega[c]=s.rho[c]*s.omega[c];}
 for(int f=0;f<nf;++f){g.boundaryMassFraction[f]=1;if(s.riemannBoundaryKind[f]==0&&s.riemannBoundaryUFix[f])g.compositionBoundaryFixed[f]=1;}
 if(layer){
  auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=int(fs::file_size(root/"wall-exchange.bin")/(9*sizeof(double)));auto*fs=new int[nf];auto*cs=new int[nc];std::fill(fs,fs+nf,-1);std::fill(cs,cs+nc,-1);w.faceSlot=fs;w.ownerSlot=cs;w.status=new int[w.count]{};w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[w.count]{};w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[w.count]{};w.speciesFlux=new Real[2*w.count]{};
  std::vector<double>values(w.count*9);read(root/"wall-exchange.bin",values.data(),values.size());
  for(int j=0;j<w.count;++j){auto*v=values.data()+9*j;int f=int(v[0]),c=int(v[1]);fs[f]=cs[c]=j;auto&e=w.exchange[j];e.matchingPressure=v[2];e.momentumX=s.magSf[f]*v[4];e.momentumY=-s.magSf[f]*v[2]+s.magSf[f]*v[5];e.momentumZ=s.magSf[f]*v[6];e.energy=-s.magSf[f]*v[3];e.ready=e.massReady=true;w.sst[j]={std::sqrt(v[4]*v[4]+v[6]*v[6]),v[7],s.V[c],v[8]};}
 }
 blockIdx.x=0;blockDim.x=1;
 for(int c=0;c<nc;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);}diagnostic(diag,"primitive gradients");
 for(int c=0;c<nc;++c){threadIdx.x=c;computeGasGradientLimiterKernel(&s);computeSstGradientsKernel(&s);}diagnostic(diag,"limiter/SST gradients");
 // Face transport consumes saved nut; compare its own same-state producer separately.
 std::vector<Real>savedNut(s.nut,s.nut+nc);Real nutRel=0;
 for(int c=0;c<nc;++c){threadIdx.x=c;computeGasEddyViscosityKernel(&s);nutRel=std::max(nutRel,std::abs(s.nut[c]-savedNut[c])/std::max(1e-30,std::abs(savedNut[c])));}
 diagnostic(diag,"nut");if(argc<9||std::string(argv[8])!="recompute-nut")std::copy(savedNut.begin(),savedNut.end(),s.nut);
 if(argc>4&&std::string(argv[4])=="unsupported-dispatch"){
  s.gasFluxScheme=4;Real m,x,y,z,e;bool ok=computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e);
  std::cout<<"mixture_HLLC_success="<<ok<<" fatal="<<diag.fatalClaim<<" code="<<int(diag.firstFatal.code)<<" site="<<diag.firstFatal.sourceSiteId<<" face="<<diag.firstFatal.entityIndex<<" observed="<<diag.firstFatal.observed<<'\n';
  return !ok&&diag.fatalClaim&&diag.firstFatal.code==ugkp::diagnostics::Code::MissingOperator?0:7;
 }
 for(int f=0;f<nf;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);}diagnostic(diag,"full face flux");
 std::vector<std::array<Real,5>> full(nf);for(int f=0;f<nf;++f)full[f]={s.gasPhiRho[f],s.gasPhiRhoUx[f],s.gasPhiRhoUy[f],s.gasPhiRhoUz[f],s.gasPhiRhoE[f]};
 State inviscid=s;inviscid.gasMu=0;inviscid.turbulenceModel=0;inviscid.gasSpecies.diffusivity=nullptr;
 std::ofstream faces(output/"normal-face-budget.csv");faces<<std::setprecision(17)<<"face,owner,neighbour,x,y,full_mx,inviscid_mx,viscous_mx,advective_mx,numerical_mx,leftUx,rightUx,leftRho,rightRho,own_gradientUy,own_limiterUx,HLLC_constant_gamma_core_mx,SLAU2_constant_gamma_core_mx\n";
 Real decomposition=0;
 for(int f=0;f<ni;++f)if(std::abs(s.Sfy[f]/s.magSf[f])>.99&&s.faceCy[f]<(argc>3?std::atof(argv[3]):.003)){
  int a=s.faceOwner[f],b=s.faceNeighbour[f];Real mi,x,y,z,en;if(!computeRiemannGasFaceFluxDevice<false>(inviscid,f,mi,x,y,z,en))return 3;
  GasPrimDevice left{},right{};Real yl[2],yr[2];bool el=false,er=false;if(!reconstructGasMixtureFace(s,a,f,left,yl,false,&el)||!reconstructGasMixtureFace(s,b,f,right,yr,false,&er))return 4;
  auto sound=[&](GasPrimDevice v,Real*Y){Real R=ugkwp::mixtureGasConstant(Y,g.thermo),cv=ugkwp::mixtureHeatCapacity(Y,v.T,g.thermo);return std::sqrt((cv+R)/cv*R*v.T);};
  Real nx=s.Sfx[f]/s.magSf[f],ny=s.Sfy[f]/s.magSf[f],nz=s.Sfz[f]/s.magSf[f],unl=left.ux*nx+left.uy*ny+left.uz*nz,unr=right.ux*nx+right.uy*ny+right.uz*nz;
  Real ap=std::max(Real(0),std::max(unl+sound(left,yl),unr+sound(right,yr))),am=std::min(Real(0),std::min(unl-sound(left,yl),unr-sound(right,yr)));
  Real adv=ugkpriemann::hllComponentFromWaveSpeeds(left.rho*left.ux*unl+left.p*nx,right.rho*right.ux*unr+right.p*nx,0,0,ap,am);
  Real hllNum=ugkpriemann::hllComponentFromWaveSpeeds(0,0,left.rho*left.ux,right.rho*right.ux,ap,am);Real num=x/s.magSf[f]-adv;if(s.gasFluxScheme==2)decomposition=std::max(decomposition,std::abs((adv+hllNum)-x/s.magSf[f]));
  // Historical constant-gamma comparison diagnostic only. Production mixture
  // HLLC/SLAU2 are supported by the bundled source.
  ugkpriemann::Primitive pl{left.rho,left.ux,left.uy,left.uz,left.p},pr{right.rho,right.ux,right.uy,right.uz,right.p};
  auto hf=ugkpriemann::fluxUnitArea(pl,pr,nx,ny,nz,s.gammaGas,ugkpriemann::Scheme::HLLC);
  auto sf=ugkpriemann::fluxUnitArea(pl,pr,nx,ny,nz,s.gammaGas,ugkpriemann::Scheme::SLAU2);
  if(!hf.valid||!sf.valid)return 5;
  faces<<f<<','<<a<<','<<b<<','<<s.faceCx[f]<<','<<s.faceCy[f]<<','<<full[f][1]/s.magSf[f]<<','<<x/s.magSf[f]<<','<<(full[f][1]-x)/s.magSf[f]<<','<<adv<<','<<num<<','<<left.ux<<','<<right.ux<<','<<left.rho<<','<<right.rho<<','<<s.gradUxY[a]<<','<<s.gasGradientLimiterUx[a]<<','<<hf.flux[1]<<','<<sf.flux[1]<<'\n';
 }
 diagnostic(diag,"inviscid separation");
 // Restore full flux, including species channels overwritten by inviscid calls.
 for(int f=0;f<nf;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);}diagnostic(diag,"restore face flux");

 for(int f=0;f<nf;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}diagnostic(diag,"budget SST faces");
 std::ofstream kb(output/"macro-cell-budget.csv");kb<<std::setprecision(17)<<"cell,x,y,rho,Ux,T,k,omega,nut,tau,productionUsed,productionStrain,productionCap,destruction,compression,kAdvectionX,kAdvectionY,kDiffusionX,kDiffusionY,kRhs,kRhsKernel,momentumWall,momentumNormalInviscid,momentumNormalViscous,momentumStreamInviscid,momentumStreamViscous,momentumOther,momentumRhs,divU\n";
 std::vector<std::array<Real,5>> inv(nf);
 for(int f=0;f<nf;++f){
   if(f>=ni&&(s.riemannBoundaryKind[f]==1||s.riemannBoundaryKind[f]==3||s.riemannBoundaryKind[f]==4))continue;
   if(!computeRiemannGasFaceFluxDevice<false>(inviscid,f,inv[f][0],inv[f][1],inv[f][2],inv[f][3],inv[f][4])){std::cerr<<"inviscid budget failure face="<<f<<'\n';diagnostic(diag,"budget inviscid");return 3;}
 }
 // Restore species/full flux after diagnostic evaluation, without advancing state.
 for(int f=0;f<nf;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);}
 for(int c=0;c<nc;++c)if(s.Cy[c]<.0008&&s.Cx[c]>.34&&s.Cx[c]<.86){
   Real adv[2]={},dif[2]={},momWall=0,momNI=0,momNV=0,momSI=0,momSV=0,momOther=0,volumeFlux=0;
   const int start=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
   for(int i=0;i<count;++i){const int plane=start+i,f=s.cellFaceId[plane];
     const Real sign=s.faceOwner[f]==c?-1:1;const int a=s.faceOwner[f],b=coupledFaceNeighbour(s,f);
     const Real mass=s.gasPhiRho[f],ko=s.k[a],kn=b>=0?s.k[b]:sstBoundaryValue(s,f,a,false);
     Real ak=mass*(mass>=0?ko:kn),dk=s.sstPhiRhoK[f]-ak;
     const int axis=std::abs(s.Sfx[f])>std::abs(s.Sfy[f])?0:1;
     adv[axis]+=sign*ak/s.V[c];dif[axis]+=sign*dk/s.V[c];
     volumeFlux-=sign*mass/sstFaceDensity(s,f);
     if(f>=ni&&s.riemannBoundaryKind[f]==2)momWall+=sign*full[f][1]/s.V[c];
     else if(std::abs(s.Sfy[f]/s.magSf[f])>.99){momNI+=sign*inv[f][1]/s.V[c];momNV+=sign*(full[f][1]-inv[f][1])/s.V[c];}
     else if(std::abs(s.Sfx[f]/s.magSf[f])>.99){momSI+=sign*inv[f][1]/s.V[c];momSV+=sign*(full[f][1]-inv[f][1])/s.V[c];}
     else momOther+=sign*full[f][1]/s.V[c];
   }
   const Real div=volumeFlux/s.V[c];Real tr=0,s2=0,gby=0;sstVelocityInvariants(s,c,tr,s2,gby);
   const Real prod=s.rho[c]*sstKProductionForCell(s,c,gby),strain=s.rho[c]*s.nut[c]*gby;
   const Real cap=s.rho[c]*s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[c]*s.omega[c];
   const Real destruction=s.rho[c]*s.sstCoefficients.betaStar*s.k[c]*s.omega[c];
   const Real compression=Real(2)/3*s.rho[c]*s.k[c]*div;
   Real sourceK=0,sourceO=0;if(!sstSourcesForCell(s,c,div,sourceK,sourceO))return 5;
   Real flux=0;for(int i=0;i<count;++i){const int f=s.cellFaceId[start+i];flux+=(s.faceOwner[f]==c?-1:1)*s.sstPhiRhoK[f]/s.V[c];}
   const int slot=ugkwp::gasBoundaryLayerOwnerSlot(s,c);Real shear=slot>=0?s.gasBoundaryLayer.sst[slot].wallShearStress:0;
   kb<<c<<','<<s.Cx[c]<<','<<s.Cy[c]<<','<<s.rho[c]<<','<<s.Ux[c]<<','<<s.Tgas[c]<<','<<s.k[c]<<','<<s.omega[c]<<','<<s.nut[c]<<','<<shear<<','<<prod<<','<<strain<<','<<cap<<','<<destruction<<','<<compression<<','<<adv[0]<<','<<adv[1]<<','<<dif[0]<<','<<dif[1]<<','<<adv[0]+adv[1]+dif[0]+dif[1]+prod-destruction-compression<<','<<flux+sourceK<<','<<momWall<<','<<momNI<<','<<momNV<<','<<momSI<<','<<momSV<<','<<momOther<<','<<momWall+momNI+momNV+momSI+momSV+momOther<<','<<div<<'\n';
 }

 std::vector<int>selected;
 for(double station:{.35,.38,.58,.75,.85}){
   int best=-1;double score=1e99;
   for(int c=0;c<nc;++c){double v=std::abs(s.Cx[c]-station)+100*s.Cy[c];if(v<score){score=v;best=c;}}
   selected.push_back(best);
 }
 std::vector<bool>selectedFace(nf,false);
 for(int c:selected)for(int i=0;i<s.cellPlaneCount[c];++i)selectedFace[s.cellFaceId[s.cellPlaneStart[c]+i]]=true;
 std::ofstream detailed(output/"detailed-faces.csv");
 detailed<<std::setprecision(17)<<"face,owner,neighbour,x,y,z,area,Sfx,Sfy,Sfz,mass,momentumX,momentumY,momentumZ,inviscidMomentumX,inviscidMomentumY,inviscidMomentumZ,viscousMomentumX,viscousMomentumY,viscousMomentumZ,energy,energyAdvection,pressureWork,energyInviscid,viscousWork,conductionAndWallCorrection,cellStressWork,pressureMomentumX,pressureMomentumY,pressureMomentumZ\n";
 std::vector<Real> pressureWork(nf,0),pressureFace(nf,0);
 std::vector<Real>workOld(nf,0),workCentral(nf,0),workCell(nf,0),heat(nf,0);
 std::ofstream ef(output/"thermal-faces.csv");ef<<std::setprecision(17)<<"face,owner,neighbour,x,y,mass,energyInviscid,workReconstructed,workCentral,workCellStress,heat,velocityReconstructedX,velocityCentralX,tractionX\n";
 for(int f=0;f<nf;++f){
  int own=s.faceOwner[f],nei=coupledFaceNeighbour(s,f);Real vo[3]={},vc[3]={};
  if(nei>=0){
   GasPrimDevice l{},r{};Real yl[2],yr[2];bool le=false,re=false;
   if(!reconstructGasMixtureFace(s,own,f,l,yl,false,&le)||!reconstructGasMixtureFace(s,nei,f,r,yr,false,&re))return 6;
   // Exact pressure/energy split for the stationary general-mixture HLLC fan.
   // This is diagnostic only; the production flux was already computed above.
   const Real nx=s.Sfx[f]/s.magSf[f],ny=s.Sfy[f]/s.magSf[f],nz=s.Sfz[f]/s.magSf[f];
   const Real ul=l.ux*nx+l.uy*ny+l.uz*nz,ur=r.ux*nx+r.uy*ny+r.uz*nz;
   auto acoustic=[&](const GasPrimDevice& v,Real* y){const Real R=ugkwp::mixtureGasConstant(y,g.thermo),cv=ugkwp::mixtureHeatCapacity(y,v.T,g.thermo);return std::sqrt((cv+R)/cv*R*v.T);};
   const Real cl=acoustic(l,yl),cr=acoustic(r,yr);
   const Real sl=std::min(ul-cl,ur-cr),sr=std::max(ul+cl,ur+cr);
   Real up,pp;
   if(sl>=0){up=ul;pp=l.p;}else if(sr<=0){up=ur;pp=r.p;}
   else {const Real al=l.rho*(sl-ul),ar=r.rho*(sr-ur);up=(l.p-r.p-al*ul+ar*ur)/(ar-al);pp=l.p+al*(up-ul);}
   pressureWork[f]=pp*up*s.magSf[f];pressureFace[f]=pp;
   const Real w=s.faceWeight[f];
   vo[0]=w*l.ux+(1-w)*r.ux;vo[1]=w*l.uy+(1-w)*r.uy;vo[2]=w*l.uz+(1-w)*r.uz;
   vc[0]=w*s.Ux[own]+(1-w)*s.Ux[nei];vc[1]=w*s.Uy[own]+(1-w)*s.Uy[nei];vc[2]=w*s.Uz[own]+(1-w)*s.Uz[nei];
  }else if(s.riemannBoundaryKind[f]==2){vo[0]=vc[0]=s.riemannBoundaryUx[f];vo[1]=vc[1]=s.riemannBoundaryUy[f];vo[2]=vc[2]=s.riemannBoundaryUz[f];}
  else {vo[0]=vc[0]=s.Ux[own];vo[1]=vc[1]=s.Uy[own];vo[2]=vc[2]=s.Uz[own];}
  for(int j=0;j<3;++j){workOld[f]+=(full[f][j+1]-inv[f][j+1])*vo[j];workCentral[f]+=(full[f][j+1]-inv[f][j+1])*vc[j];}
  workCell[f]=workCentral[f];
  if(nei>=0){
   const ugkptransport::Vector3 n{s.Sfx[f]/s.magSf[f],s.Sfy[f]/s.magSf[f],s.Sfz[f]/s.magSf[f]};
   auto cellWork=[&](int c){
    ugkptransport::Vector3 gx{s.gradUxX[c],s.gradUxY[c],s.gradUxZ[c]},gy{s.gradUyX[c],s.gradUyY[c],s.gradUyZ[c]},gz{s.gradUzX[c],s.gradUzY[c],s.gradUzZ[c]};
    auto traction=ugkptransport::openFoamNewtonianTraction(s.gasMu+s.rho[c]*s.nut[c],n,{ugkptransport::dot(gx,n),ugkptransport::dot(gy,n),ugkptransport::dot(gz,n)},gx,gy,gz);
    return -(traction.x*s.Ux[c]+traction.y*s.Uy[c]+traction.z*s.Uz[c])*s.magSf[f];
   };
   workCell[f]=s.faceWeight[f]*cellWork(own)+(1-s.faceWeight[f])*cellWork(nei);
  }
  heat[f]=full[f][4]-inv[f][4]-workOld[f];
  if(nei<0 && s.riemannBoundaryKind[f]==2)
    pressureFace[f]=(inv[f][1]*s.Sfx[f]+inv[f][2]*s.Sfy[f]+inv[f][3]*s.Sfz[f])/(s.magSf[f]*s.magSf[f]);
  if(selectedFace[f]){
   detailed<<f<<','<<own<<','<<nei<<','<<s.faceCx[f]<<','<<s.faceCy[f]<<','<<s.faceCz[f]<<','<<s.magSf[f]<<','<<s.Sfx[f]<<','<<s.Sfy[f]<<','<<s.Sfz[f];
   for(int j=0;j<4;++j)detailed<<','<<full[f][j];
   for(int j=1;j<4;++j)detailed<<','<<inv[f][j];
   for(int j=1;j<4;++j)detailed<<','<<full[f][j]-inv[f][j];
   detailed<<','<<full[f][4]<<','<<inv[f][4]-pressureWork[f]<<','<<pressureWork[f]<<','<<inv[f][4]<<','<<workOld[f]<<','<<heat[f]<<','<<workCell[f]<<','<<pressureFace[f]*s.Sfx[f]<<','<<pressureFace[f]*s.Sfy[f]<<','<<pressureFace[f]*s.Sfz[f]<<'\n';
  }

  if(s.faceCy[f]<.0008&&s.faceCx[f]>.34&&s.faceCx[f]<.86)
   ef<<f<<','<<own<<','<<nei<<','<<s.faceCx[f]<<','<<s.faceCy[f]<<','<<full[f][0]/s.magSf[f]<<','<<inv[f][4]/s.magSf[f]<<','<<workOld[f]/s.magSf[f]<<','<<workCentral[f]/s.magSf[f]<<','<<workCell[f]/s.magSf[f]<<','<<heat[f]/s.magSf[f]<<','<<vo[0]<<','<<vc[0]<<','<<-(full[f][1]-inv[f][1])/s.magSf[f]<<'\n';
 }

 std::ofstream dc(output/"detailed-cells.csv");
 dc<<std::setprecision(17)<<"cell,x,y,z,volume,rho,Ux,Uy,Uz,T,p,k,omega,nut,cp,cv,R,mu,conductivityLaminar,Pr,Prt,massRhs,momentumRhsX,momentumRhsY,momentumRhsZ,energyRhs,energyVolumeSource,massVolumeSource,momentumVolumeSourceX,momentumVolumeSourceY,momentumVolumeSourceZ,Tdot,TdotCellStressWork\n";
 std::ofstream ec(output/"thermal-cells.csv");ec<<std::setprecision(17)<<"cell,x,y,T,energyRhs,energyInviscid,workReconstructed,workCentral,workCellStress,heat,wallHeat,Tdot,TdotCentralWork,TdotCellStressWork\n";
 for(int c=0;c<nc;++c)if(s.Cy[c]<.0008&&s.Cx[c]>.34&&s.Cx[c]<.86){
  Real rhs[5]={},ei=0,wo=0,wc=0,wt=0,h=0,wh=0;
  for(int i=0;i<s.cellPlaneCount[c];++i){int f=s.cellFaceId[s.cellPlaneStart[c]+i];Real sign=s.faceOwner[f]==c?-1:1;
   for(int j=0;j<5;++j)rhs[j]+=sign*full[f][j]/s.V[c];
   ei+=sign*inv[f][4]/s.V[c];wo+=sign*workOld[f]/s.V[c];wc+=sign*workCentral[f]/s.V[c];h+=sign*heat[f]/s.V[c];wt+=sign*workCell[f]/s.V[c];
   if(f>=ni&&s.riemannBoundaryKind[f]==2)wh+=sign*heat[f]/s.V[c];
  }
  const Real cv=s.gasCp-s.Rgas;
  const Real thermal=rhs[4]-s.Ux[c]*rhs[1]-s.Uy[c]*rhs[2]-s.Uz[c]*rhs[3]+(.5*(s.Ux[c]*s.Ux[c]+s.Uy[c]*s.Uy[c]+s.Uz[c]*s.Uz[c])-cv*s.Tgas[c])*rhs[0];

  if(std::find(selected.begin(),selected.end(),c)!=selected.end()){
   dc<<c<<','<<s.Cx[c]<<','<<s.Cy[c]<<','<<s.Cz[c]<<','<<s.V[c]<<','<<s.rho[c]<<','<<s.Ux[c]<<','<<s.Uy[c]<<','<<s.Uz[c]<<','<<s.Tgas[c]<<','<<s.p[c]<<','<<s.k[c]<<','<<s.omega[c]<<','<<s.nut[c]<<','<<s.gasCp<<','<<cv<<','<<s.Rgas<<','<<s.gasMu<<','<<s.gasMu*s.gasCp/s.gasPrClamped<<','<<s.gasPrClamped<<','<<s.turbulentPrandtl;
   for(int j=0;j<5;++j)dc<<','<<rhs[j];
   dc<<",0,0,0,0,0,"<<thermal/(s.rho[c]*cv)<<','<<(thermal-wo+wt)/(s.rho[c]*cv)<<'\n';
  }
  ec<<c<<','<<s.Cx[c]<<','<<s.Cy[c]<<','<<s.Tgas[c]<<','<<rhs[4]<<','<<ei<<','<<wo<<','<<wc<<','<<wt<<','<<h<<','<<wh<<','<<thermal/(s.rho[c]*cv)<<','<<(thermal-wo+wc)/(s.rho[c]*cv)<<','<<(thermal-wo+wt)/(s.rho[c]*cv)<<'\n';
 }
 diagnostic(diag,"source budgets");
 std::array<long double,5> before{},after{},boundary{},absolute{};Real*fields[]={s.rho,s.rhoUx,s.rhoUy,s.rhoUz,s.rhoE};
 std::vector<std::array<Real,5>> cells(nc);
 for(int c=0;c<nc;++c)for(int j=0;j<5;++j){before[j]+=static_cast<long double>(s.V[c])*fields[j][c];cells[c][j]=fields[j][c];}
 for(int f=ni;f<nf;++f)for(int j=0;j<5;++j){boundary[j]+=full[f][j];absolute[j]+=std::abs(full[f][j]);}
 for(int c=0;c<nc;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,1e-7);}diagnostic(diag,"Euler divergence");
 std::array<long double,5> changes{};
 for(int c=0;c<nc;++c)for(int j=0;j<5;++j){after[j]+=static_cast<long double>(s.V[c])*fields[j][c];changes[j]+=static_cast<long double>(s.V[c])*(fields[j][c]-cells[c][j]);}
 std::ofstream local(output/"first-layer-response.csv");local<<std::setprecision(17)<<"cell,x,y,dU_dt,drhoUx_dt,drho_dt\n";
 for(int c=0;c<nc;++c)if(s.Cy[c]<.0002)local<<c<<','<<s.Cx[c]<<','<<s.Cy[c]<<','<<(s.rhoUx[c]/s.rho[c]-cells[c][1]/cells[c][0])/1e-7<<','<<(s.rhoUx[c]-cells[c][1])/1e-7<<','<<(s.rho[c]-cells[c][0])/1e-7<<'\n';
 std::ofstream budget(output/"global-euler-budget.csv");budget<<std::setprecision(20)<<"component,inventory_before,inventory_after,measured_change,expected_change,closure_error,closure_over_absolute_boundary_step\n";for(int j=0;j<5;++j)budget<<j<<','<<before[j]<<','<<after[j]<<','<<changes[j]<<','<<-1e-7L*boundary[j]<<','<<changes[j]+1e-7L*boundary[j]<<','<<(changes[j]+1e-7L*boundary[j])/std::max(1e-30L,1e-7L*absolute[j])<<'\n';
 std::cout<<std::setprecision(17)<<"snapshot_nut_producer_max_relative_difference="<<nutRel<<"\nKurganov_split_max_absolute_difference="<<decomposition<<"\nfaces="<<nf<<" cells="<<nc<<" fatal="<<diag.fatalClaim<<'\n';
}
