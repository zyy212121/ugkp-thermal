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
 s.nCells=nc;s.nFaces=nf;s.nInternalFaces=ni;s.gasFluxScheme=2;s.gasReconstruction=1;s.gasLimiter=1;s.gasMu=1.7244421537657099e-5;s.gasPrClamped=.72;s.gasCp=1004.675;s.Rgas=287.0500676;s.gammaGas=s.gasCp/(s.gasCp-s.Rgas);s.rhoMin=1e-12;s.TgasMin=1e-9;s.maxDiffusionNumber=.25;
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
 diagnostic(diag,"nut");std::copy(savedNut.begin(),savedNut.end(),s.nut);
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
  Real num=ugkpriemann::hllComponentFromWaveSpeeds(0,0,left.rho*left.ux,right.rho*right.ux,ap,am);decomposition=std::max(decomposition,std::abs((adv+num)-x/s.magSf[f]));
  // Pure-air constant-gamma core comparison only. The production mixture
  // dispatcher currently supports schemes 1/2 and cannot switch to HLLC/SLAU2.
  ugkpriemann::Primitive pl{left.rho,left.ux,left.uy,left.uz,left.p},pr{right.rho,right.ux,right.uy,right.uz,right.p};
  auto hf=ugkpriemann::fluxUnitArea(pl,pr,nx,ny,nz,s.gammaGas,ugkpriemann::Scheme::HLLC);
  auto sf=ugkpriemann::fluxUnitArea(pl,pr,nx,ny,nz,s.gammaGas,ugkpriemann::Scheme::SLAU2);
  if(!hf.valid||!sf.valid)return 5;
  faces<<f<<','<<a<<','<<b<<','<<s.faceCx[f]<<','<<s.faceCy[f]<<','<<full[f][1]/s.magSf[f]<<','<<x/s.magSf[f]<<','<<(full[f][1]-x)/s.magSf[f]<<','<<adv<<','<<num<<','<<left.ux<<','<<right.ux<<','<<left.rho<<','<<right.rho<<','<<s.gradUxY[a]<<','<<s.gasGradientLimiterUx[a]<<','<<hf.flux[1]<<','<<sf.flux[1]<<'\n';
 }
 diagnostic(diag,"inviscid separation");
 // Restore full flux, including species channels overwritten by inviscid calls.
 for(int f=0;f<nf;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);}diagnostic(diag,"restore face flux");
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


