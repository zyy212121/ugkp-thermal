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
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include "common/gasWall/WallGeometryBuilder.H"
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2>gasSpecies;ugkwp::GasBoundaryLayerModelState<Real,2,128>gasBoundaryLayerModel;};
template<class T>void fieldStorage(T&v,const fs::path&p,std::size_t count){if constexpr(std::is_pointer_v<T>){using E=std::remove_const_t<std::remove_pointer_t<T>>;auto*data=new E[count]{};v=data;if(fs::exists(p)){auto size=fs::file_size(p);if(size>count*sizeof(E)||size%sizeof(E))std::abort();std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(data),size);if(!f)std::abort();}}}
template<class T>void read(fs::path p,T*data,std::size_t n){std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(data),n*sizeof(T));if(!f){std::cerr<<"read failed "<<p<<'\n';std::abort();}}
void diagnostic(ugkp::diagnostics::DeviceContext&d,const char*where){if(d.fatalClaim){auto&r=d.firstFatal;std::cerr<<where<<" fatal code="<<int(r.code)<<" site="<<r.sourceSiteId<<" entity="<<r.entityIndex<<" value="<<r.observed<<'\n';std::exit(2);}}
int main(int argc,char**argv){
 if(argc<3)return 1;fs::path root=argv[1],output=argv[2];fs::create_directories(output);bool layer=true;
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
 blockIdx.x=0;blockDim.x=1;
 for(int c=0;c<nc;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);}diagnostic(diag,"primitive gradients");
 for(int c=0;c<nc;++c){threadIdx.x=c;computeGasGradientLimiterKernel(&s);computeSstGradientsKernel(&s);}diagnostic(diag,"limiter/SST gradients");

 using namespace ugkwp::gaswall;
 WallGeometryMesh<double> mesh;
 std::ifstream mf(argv[9],std::ios::binary);int mn[4];mf.read(reinterpret_cast<char*>(mn),sizeof(mn));
 auto rv=[&](auto& v,int n){v.resize(n);mf.read(reinterpret_cast<char*>(v.data()),n*sizeof(v[0]));};
 rv(mesh.points,mn[0]);rv(mesh.cellCentres,mn[1]);rv(mesh.volumes,mn[1]);rv(mesh.faceOffsets,mn[2]+1);rv(mesh.facePoints,mn[3]);rv(mesh.owner,mn[2]);rv(mesh.neighbour,mn[2]);rv(mesh.physicalWall,mn[2]);if(!mf)return 11;
 std::vector<int> wf;for(int f=0;f<nf;++f)if(mesh.physicalWall[f])wf.push_back(f);
 WallGeometryOptions<double> options;const int nodes=argc>10?std::atoi(argv[10]):24;options.reconstructMatching=true;options.constructQuadrature=true;options.profileNodes=nodes;options.profileStretch=2;
 WallGeometryCache<double> geometry;WallStatus gs;if(!geometry.rebuild(mesh,wf,gs,options))return 12;
 auto& wall=s.gasBoundaryLayer;wall.enabled=true;wall.count=wf.size();
 std::vector<int> fslot(nf,-1),cslot(nc,-1),offset(1,0),cells;
 std::vector<Real> weights;
 wall.faceSlot=fslot.data();wall.ownerSlot=cslot.data();wall.status=new int[wf.size()]{};
 wall.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[wf.size()]{};wall.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[wf.size()]{};wall.speciesFlux=new Real[2*wf.size()]{};
 auto& model=s.gasBoundaryLayerModel;model.config=properties.config;model.config.nodes=nodes;model.volumeSstClosure=true;
 model.input=new WallInput<Real,2>[wf.size()];model.output=new WallOutput<Real,2>[wf.size()];model.status=new WallStatus[wf.size()];
 model.workspace=new WallWorkspace<Real,2,128>;model.workspaceCapacity=128;model.workspaceCount=1;model.faces=wf.data();model.profile=new Real[wf.size()*nodes*9];
 for(int j=0;j<int(wf.size());++j){const auto&d=geometry.descriptors[j];fslot[wf[j]]=cslot[d.ownerCell]=j;
  auto& in=model.input[j];in=properties.input;in.ownerDistance=d.ownerDistance;in.matchingDistance=d.matchingDistance;in.quadrature=d.quadrature();
  for(int k=0;k<3;++k)in.normal[k]=d.normal[k];
  cells.insert(cells.end(),d.matchingCells.begin(),d.matchingCells.end());weights.insert(weights.end(),d.matchingWeights.begin(),d.matchingWeights.end());offset.push_back(cells.size());}
 model.matchingOffsets=offset.data();model.matchingCells=cells.data();model.matchingWeights=weights.data();
 std::ofstream closure(output/"closure.csv");closure<<std::setprecision(17)<<"face,cell,x,y,matchY,omegaInput,omegaPublished,omegaPoint,omegaDestruction,kInput,kContentMean,kDestructionRate,sourceIntegral,balanceResidual,Umatch,kmatch,omegaMatch,tau,iterations,residual\n";
 for(int j=0;j<int(wf.size());++j){
  if(!ugkwp::evaluateGasBoundaryLayerSlot<true,128>(s,j,0,detail::SerialWallTeam{})){diagnostic(diag,"wall model");return 13;}
  auto& in=model.input[j];auto& out=model.output[j];int c=mesh.owner[wf[j]];
  auto* p=model.profile+nodes*9*j;int l=0;while(l<nodes-2&&p[(l+1)*9]<in.ownerDistance)++l;
  double f=(in.ownerDistance-p[l*9])/(p[(l+1)*9]-p[l*9]);double z=(1-f)/std::sqrt(p[l*9+6])+f/std::sqrt(p[(l+1)*9+6]);
  closure<<wf[j]<<','<<c<<','<<s.Cx[c]<<','<<in.ownerDistance<<','<<in.matchingDistance<<','<<s.omega[c]<<','<<out.ownerOmega<<','<<1/(z*z)<<','<<out.ownerKDestruction/(.09*out.ownerKContent)<<','<<s.k[c]<<','<<out.ownerKContent/s.V[c]<<','<<out.ownerKDestruction/out.ownerKContent<<','<<out.integratedKSource<<','<<out.ownerKBalanceResidual<<','<<in.matching.velocity[0]<<','<<in.matching.k<<','<<in.matching.omega<<','<<wall.sst[j].wallShearStress<<','<<out.iterations<<','<<out.residual<<'\n';
 }
 for(int c=0;c<nc;++c)applySstWallFunctionStateCell(s,c);
 for(int c=0;c<nc;++c){threadIdx.x=c;computeSstGradientsKernel(&s);}
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
 std::ofstream kb(output/"macro-cell-budget.csv");kb<<std::setprecision(17)<<"cell,x,y,rho,Ux,T,k,omega,nut,tau,productionUsed,productionStrain,productionCap,destruction,compression,kAdvectionX,kAdvectionY,kDiffusionX,kDiffusionY,kRhs,kRhsKernel,momentumWall,momentumNormalInviscid,momentumNormalViscous,momentumStreamInviscid,momentumStreamViscous,momentumOther,momentumRhs,divU,omegaAdvectionX,omegaAdvectionY,omegaDiffusionX,omegaDiffusionY,omegaSource,omegaPdeRhs,omegaAlgebraic,kSourceApplied,kDecayApplied,kNonDestructiveApplied,omegaProduction,omegaDestruction,omegaCompression,omegaCrossDiffusion,omegaSourceDecompositionError,pressureGradientXForce,resolvedStreamAcceleration,resolvedNormalAcceleration\n";
 std::vector<std::array<Real,5>> inv(nf);
 for(int f=0;f<nf;++f){
   if(f>=ni&&(s.riemannBoundaryKind[f]==1||s.riemannBoundaryKind[f]==3||s.riemannBoundaryKind[f]==4))continue;
   if(!computeRiemannGasFaceFluxDevice<false>(inviscid,f,inv[f][0],inv[f][1],inv[f][2],inv[f][3],inv[f][4])){std::cerr<<"inviscid budget failure face="<<f<<'\n';diagnostic(diag,"budget inviscid");return 3;}
 }
 // Restore species/full flux after diagnostic evaluation, without advancing state.
 for(int f=0;f<nf;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);}
 for(int c=0;c<nc;++c)if(s.Cy[c]<.0008&&s.Cx[c]>.34&&s.Cx[c]<.86){
   Real adv[2]={},dif[2]={},oa[2]={},od[2]={},momWall=0,momNI=0,momNV=0,momSI=0,momSV=0,momOther=0,volumeFlux=0;
   const int start=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
   for(int i=0;i<count;++i){const int plane=start+i,f=s.cellFaceId[plane];
     const Real sign=s.faceOwner[f]==c?-1:1;const int a=s.faceOwner[f],b=coupledFaceNeighbour(s,f);
     const Real mass=s.gasPhiRho[f],ko=s.k[a],kn=b>=0?s.k[b]:sstBoundaryValue(s,f,a,false);
     Real ak=mass*(mass>=0?ko:kn),dk=s.sstPhiRhoK[f]-ak;
     const int axis=std::abs(s.Sfx[f])>std::abs(s.Sfy[f])?0:1;
     adv[axis]+=sign*ak/s.V[c];dif[axis]+=sign*dk/s.V[c];
     Real ao=mass*(mass>=0?s.omega[a]:(b>=0?s.omega[b]:sstBoundaryValue(s,f,a,true)));oa[axis]+=sign*ao/s.V[c];od[axis]+=sign*(s.sstPhiRhoOmega[f]-ao)/s.V[c];
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
   const Real gamma=ugkwp::sstGamma(s.sstF1[c],s.sstCoefficients);
   const Real op=s.rho[c]*gamma*std::min(gby,ugkwp::sstOmegaProductionLimit(s.omega[c],s2,s.sstF2[c],s.sstCoefficients));
   const Real oe=s.rho[c]*ugkwp::sstBeta(s.sstF1[c],s.sstCoefficients)*s.omega[c]*s.omega[c];
   const Real oc=Real(2)/3*s.rho[c]*gamma*div*s.omega[c];
   const Real cd=ugkwp::sstCrossDiffusion(s.omega[c],s.gradKX[c]*s.gradOmegaX[c]+s.gradKY[c]*s.gradOmegaY[c]+s.gradKZ[c]*s.gradOmegaZ[c],s.sstCoefficients);
   const Real ox=s.rho[c]*(1-s.sstF1[c])*cd;
   const Real kd=slot>=0?s.rho[c]*s.k[c]*s.gasBoundaryLayer.sst[slot].kDestructionRate:destruction;
   const Real kp=sourceK+kd;
   kb<<c<<','<<s.Cx[c]<<','<<s.Cy[c]<<','<<s.rho[c]<<','<<s.Ux[c]<<','<<s.Tgas[c]<<','<<s.k[c]<<','<<s.omega[c]<<','<<s.nut[c]<<','<<shear<<','<<prod<<','<<strain<<','<<cap<<','<<destruction<<','<<compression<<','<<adv[0]<<','<<adv[1]<<','<<dif[0]<<','<<dif[1]<<','<<adv[0]+adv[1]+dif[0]+dif[1]+prod-destruction-compression<<','<<flux+sourceK<<','<<momWall<<','<<momNI<<','<<momNV<<','<<momSI<<','<<momSV<<','<<momOther<<','<<momWall+momNI+momNV+momSI+momSV+momOther<<','<<div<<','<<oa[0]<<','<<oa[1]<<','<<od[0]<<','<<od[1]<<','<<sourceO<<','<<oa[0]+oa[1]+od[0]+od[1]+sourceO<<','<<(slot>=0)<<','<<sourceK<<','<<kd<<','<<kp<<','<<op<<','<<oe<<','<<oc<<','<<ox<<','<<sourceO-(op-oe-oc+ox)<<','<<-s.gradPx[c]<<','<<-s.rho[c]*s.Ux[c]*s.gradUxX[c]<<','<<-s.rho[c]*s.Uy[c]*s.gradUxY[c]<<'\n';
 }

 std::cout<<"complete "<<nc<<" cells\n";return 0;
}
