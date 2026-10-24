#include "fvCFD.H"
#include "emptyFvPatch.H"
#include "materials/dry/BulkResponse.H"
#include "materials/dry/PoreTransport.H"
#include "materials/dry/DrySurfaceResponse.H"
#include <fstream>
#include <iomanip>
#include <vector>
using namespace Foam;
namespace dry=chmt::dry;
struct Gas {
 double internalEnergy(size_t s,double T) const{return (s?100000.:200000.)+700*T;}
 double heatCapacityCv(size_t,double) const{return 700;}
 double enthalpy(size_t s,double T) const{return internalEnergy(s,T)+300*T;}
 double gasConstant(size_t) const{return 300;}
};
#include "coupled.H"
#include "localCases.H"
#include "mixture.H"
#include "zeroSupport.H"
int main(int argc,char**argv) {
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 SolverPerformance<scalar>::debug=0;
 IOdictionary cfg(IOobject("validationProperties",runTime.system(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 word mode(cfg.lookup("mode"));
 scalar L=readScalar(cfg.lookup("length"));
 scalar dt=readScalar(cfg.lookup("dt")), end=readScalar(cfg.lookup("end"));
 volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 std::ofstream out((runTime.path()/"result.csv").c_str());out<<std::setprecision(17);
 const scalar pi=constant::mathematical::pi;
 Gas gas;
 if(mode=="conduction" || mode=="diffusion" || mode=="conduction3d" || mode=="diffusion3d") {
  const bool threeD=mode=="conduction3d" || mode=="diffusion3d";
  const bool diff=mode=="diffusion" || mode=="diffusion3d";
  scalar storage=diff?0.35:1.0, alpha=diff?0.003:0.01;
  // Thermal alpha comes from the SAME production caloric/mixture primitive.
  dry::MaterialState<1,1> q; q.condensedMass[0]=500;
  std::array<dry::CondensedThermo,1> solid{};solid[0].density=1000;solid[0].cp0=1000;solid[0].conductivity=10000;
  if(!diff) alpha=dry::primitive(q,1,300,solid,gas,0).conductivity/dry::caloricState(q,solid,gas,300).capacity;
  const scalar a=alpha/storage;
  forAll(T,i){scalar x=mesh.C()[i].x();scalar dx=threeD?std::cbrt(mesh.V()[i]):mesh.V()[i];scalar sinc=std::sin(pi*dx/(2*L))/(pi*dx/(2*L));T[i]=std::sin(pi*x/L)*sinc;if(threeD)T[i]*=std::sin(pi*mesh.C()[i].y()/L)*std::sin(pi*mesh.C()[i].z()/L)*sinc*sinc;}
  T.correctBoundaryConditions();
  scalar initial=gSum(T.primitiveField()*mesh.V()),boundary=0, maxFluxSum=0;
  const label steps=label(std::lround(end/dt));runTime.setDeltaT(dt);
  for(label step=0;step<steps;++step){
   ++runTime;
   solve(fvm::ddt(dimensionedScalar("s",dimless,storage),T)-fvm::laplacian(dimensionedScalar("alpha",dimViscosity,alpha),T));
   tmp<surfaceScalarField> grad=fvc::snGrad(T);
   forAll(mesh.boundary(),patch) if(!isA<emptyFvPatch>(mesh.boundary()[patch]))
    boundary+=dt*alpha*gSum(grad().boundaryField()[patch]*mesh.magSf().boundaryField()[patch]);
   if(diff) forAll(mesh.neighbour(),face){label l=mesh.owner()[face],r=mesh.neighbour()[face];
    std::array<double,2> y{{0.5*(T[l]+T[r]),1-0.5*(T[l]+T[r])}};
    double gradient=(T[r]-T[l])*mesh.deltaCoeffs()[face];
    auto J=dry::correctedDiffusion(1.0,y,{{gradient,-gradient}},{{alpha,alpha}});
    maxFluxSum=max(maxFluxSum,mag(J[0]+J[1]));
   }
  }
  scalar err=0,maxErr=0;
  forAll(T,i){scalar x=mesh.C()[i].x(),dx=threeD?std::cbrt(mesh.V()[i]):mesh.V()[i];scalar sinc=std::sin(pi*dx/(2*L))/(pi*dx/(2*L));scalar exact=std::sin(pi*x/L)*sinc*std::exp(-a*sqr(pi/L)*end*(threeD?3:1));if(threeD)exact*=std::sin(pi*mesh.C()[i].y()/L)*std::sin(pi*mesh.C()[i].z()/L)*sinc*sinc;scalar e=T[i]-exact;err+=e*e*mesh.V()[i];maxErr=max(maxErr,mag(e));}
  scalar final=gSum(T.primitiveField()*mesh.V());
  out<<"mode,N,dt,L2,Linf,conservation_abs,conservation_rel,zero_sum_flux\n"<<mode<<','<<mesh.nCells()<<','<<dt<<','<<std::sqrt(err/L)<<','<<maxErr<<','<<storage*(final-initial)-boundary<<','<<mag(storage*(final-initial)-boundary)/(storage*mag(initial)+mag(boundary))<<','<<maxFluxSum<<'\n';
  T.write();
 } else if(mode=="darcy") {
  const scalar RT=90000,K=1e-12,mu=1e-5,pL=200000,pR=100000;
  forAll(T,i)T[i]=pL+(pR-pL)*mesh.C()[i].x()/L;
  T.correctBoundaryConditions();
  surfaceScalarField gamma(IOobject("gamma",runTime.timeName(),mesh),mesh,dimensionedScalar("g",dimViscosity,1));
  scalar change=1;
  label iter=0;
  for(;iter<2000 && change>1e-10;++iter){
   scalarField old(T.primitiveField());
   forAll(mesh.neighbour(),f){label l=mesh.owner()[f],r=mesh.neighbour()[f];gamma[f]=K/mu*(T[l]>T[r]?T[l]:T[r])/RT;}
   forAll(mesh.boundary(),p)if(!isA<emptyFvPatch>(mesh.boundary()[p])){
    const labelUList& cells=mesh.boundary()[p].faceCells();
    forAll(cells,j) gamma.boundaryFieldRef()[p][j]=K/mu*max(T[cells[j]],T.boundaryField()[p][j])/RT;
   }
   solve(fvm::laplacian(gamma,T));
   change=gMax(mag(T.primitiveField()-old));
  }
  surfaceScalarField J(-gamma*fvc::snGrad(T)*mesh.magSf());
  scalar err=0,fluxError=0,coreError=0;scalar exactJ=K*(sqr(pL)-sqr(pR))/(2*mu*RT*L);
  forAll(T,i){scalar x=mesh.C()[i].x(),dx=mesh.V()[i],b=(sqr(pR)-sqr(pL))/L;
   scalar exact=2*(std::pow(sqr(pL)+b*(x+dx/2),1.5)-std::pow(sqr(pL)+b*(x-dx/2),1.5))/(3*b*dx);err+=sqr(T[i]-exact)*mesh.V()[i];}
  forAll(mesh.neighbour(),f){label l=mesh.owner()[f],r=mesh.neighbour()[f];double d=1/mesh.deltaCoeffs()[f];double u=dry::darcyVelocity(K,mu,(T[r]-T[l])/d,0);
   dry::PoreFaceState<1> a,b;a.massFraction={{1}};b.massFraction={{1}};a.density=T[l]/RT;b.density=T[r]/RT;a.temperature=b.temperature=300;
   auto flux=dry::poreFaceFlux(a,b,d,u,{{0}},0,gas);coreError=max(coreError,mag(flux.species[0]*mesh.magSf()[f]-J[f]));fluxError=max(fluxError,mag(J[f]/mesh.magSf()[f]-exactJ)/exactJ);}
  scalar divErr=gMax(mag(fvc::div(J)().primitiveField()));
  out<<"mode,N,iterations,L2_pressure,relative_flux_error,divergence,core_flux_difference\n"<<mode<<','<<mesh.nCells()<<','<<iter<<','<<std::sqrt(err/L)<<','<<fluxError<<','<<divErr<<','<<coreError<<'\n';T.write();
 } else if(mode=="coupled") {coupledCase(runTime,mesh,T,dt,end,out);
 } else if(mode=="isothermal" || mode=="adiabatic") {pyrolysisCase(runTime,mesh,T,dt,end,out,mode=="adiabatic");
 } else if(mode=="surface") {surfaceCase(runTime,mesh,T,out);
 } else if(mode=="mixture") {mixtureCase(runTime,mesh,T,out);
 } else if(mode=="zeroSupport") {zeroSupportCase(runTime,mesh,dt,end,out);
 } else {FatalErrorInFunction<<"Unknown mode "<<mode<<exit(FatalError);}
 return 0;
}
