#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace chmt;
namespace {
void require(bool ok,const std::string& message) { if(!ok) { std::cerr<<"FAIL: "<<message<<'\n'; std::exit(1); } }
bool near(Real a,Real b,Real rel=2e-11) { return std::abs(a-b)<=rel*std::max(1.0,std::max(std::abs(a),std::abs(b))); }
PhysicsConfig physics() {
    PhysicsConfig p; p.enableFilm=true;p.enableGas=false;p.spatialOrder=1;
    p.liquid.rho=1000;p.liquid.cp0=2000;p.liquid.e0=-1000000;
    p.liquid.Tmin=100;p.liquid.Tmax=2000;p.liquidViscosity=1;
    p.liquidReferencePressure=100000;
    return p;
}
const Vec3 normal=normalized(Vec3{1,1,1});
const Vec3 tangentX=normalized(Vec3{1,-1,0});
const Vec3 tangentY=cross(normal,tangentX);
HostState state(const PhysicsConfig& p,int nx,int ny,Vec3 velocity) {
    HostState h;const int n=nx*ny;
    h.surface.area.assign(n,1);h.surface.oldArea.assign(n,1);h.surface.normal.assign(n,normal);
    h.surface.gasFace.assign(n,-1);h.surface.baseVelocity.assign(n,velocity);
    h.film.resize(n);h.filmAux.resize(n);
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x) {
        const int i=y*nx+x; auto& q=h.film[i];q.mass=1+.1*x+.2*y;
        q.species[0]=q.mass*(.2+.1*x);q.species[1]=q.mass-q.species[0];
        q.enthalpy=q.mass*liquidH(p,300+5*x+7*y,100000);
        h.filmAux[i].baseVelocity=velocity;h.filmAux[i].normal=normal;
        require(recoverFilm(q,1,100000,p,h.filmAux[i]),"fixture EOS");
        h.filmAux[i].meanVelocity=velocity;h.filmAux[i].topVelocity=velocity;
        h.filmAux[i].kineticEnergy=.5*q.mass*dot(velocity,velocity);
        h.surface.centre.push_back(tangentX*x+tangentY*y);
        for(int direction=0;direction<2;++direction) {
            if((direction==0&&nx==1)||(direction==1&&ny==1))continue;
            Vec3 t=direction==0?tangentX:tangentY;
            int j=direction==0?y*nx+(x+1)%nx:((y+1)%ny)*nx+x;
            h.surface.edgeOwner.push_back(i);h.surface.edgeNeighbour.push_back(j);
            h.surface.edgeLength.push_back(1);h.surface.edgeConormal.push_back(t);
            h.surface.edgeOwnerOffset.push_back(t*.5);h.surface.edgeNeighbourOffset.push_back(t*-.5);
        }
    }
    return h;
}
std::vector<CpuFilmDrive> drive(const HostState& h) {
    std::vector<CpuFilmDrive> d(h.film.size());for(auto& a:d)a.pressure=100000;return d;
}
CpuFilmCandidate advance(const HostState& h,const PhysicsConfig& p,Real dt) {
    CpuFilmCandidate c;std::string error;
    require(advanceCpuFilmCandidate(h,p,dt,std::vector<CpuFilmForcing>(h.film.size()),drive(h),c,error),"advance: "+error);return c;
}
Real energy(const HostState& h,const PhysicsConfig& p) {Real sum=0;for(std::size_t i=0;i<h.film.size();++i)sum+=filmInternalEnergy(h.film[i],h.filmAux[i].pressure,p);return sum;}
void transportBothTangents() {
    const auto p=physics();const Real dt=.02;
    for(int axis=0;axis<3;++axis) {
        Real ux=axis==1?0:.2,uy=axis==0?0:.15;
        auto h=state(p,3,3,tangentX*ux+tangentY*uy);auto c=advance(h,p,dt);
        Real mass0=0,mass1=0,species0[Ns]{},species1[Ns]{};
        for(int y=0;y<3;++y)for(int x=0;x<3;++x) {
            const int i=3*y+x,ix=3*y+(x+2)%3,iy=3*((y+2)%3)+x;
            const Real expected=h.film[i].mass-dt*(ux*(h.film[i].mass-h.film[ix].mass)+uy*(h.film[i].mass-h.film[iy].mass));
            require(near(c.film[i].mass,expected),"periodic nonuniform transport in both oblique tangents");
            mass0+=h.film[i].mass;mass1+=c.film[i].mass;
            for(int s=0;s<Ns;++s) {
                const Real expectedSpecies=h.film[i].species[s]-dt*(ux*(h.film[i].species[s]-h.film[ix].species[s])+uy*(h.film[i].species[s]-h.film[iy].species[s]));
                require(near(c.film[i].species[s],expectedSpecies),"actual donor species transported");
                species0[s]+=h.film[i].species[s];species1[s]+=c.film[i].species[s];
            }
        }
        require(near(mass0,mass1),"closed mass conservation");
        for(int s=0;s<Ns;++s)require(near(species0[s],species1[s]),"closed species conservation");
        require(near(c.report.internalEnergyAfter,energy(h,p)),"closed physical energy conservation");
        require(std::abs(c.report.reducedEnergyResidual)<1e-7,"reduced energy residual");
    }
}
void pressureStorage() {
    const auto p=physics();auto h=state(p,1,1,{});auto d=drive(h);d[0].pressure=200000;d[0].endpointPressure=300000;
    CpuFilmCandidate c;std::string error;
    require(advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),d,c,error),"pressure update: "+error);
    require(near(c.film[0].enthalpy-h.film[0].enthalpy,200),"endpoint pV exactly once");
    require(near(c.filmAux[0].temperature,h.filmAux[0].temperature),"pressure change preserves temperature");
    require(near(c.report.internalEnergyAfter,c.report.internalEnergyBefore),"pressure change preserves physical U");
    require(near(c.report.budgetDelta.filmPressureVolume,200),"PV storage audit");
}
void donorRollback() {
    auto p=physics();auto h=state(p,3,3,tangentX*.75+tangentY*.75);
    CpuFilmCandidate c;c.film.resize(1);c.film[0].mass=987;std::string error;
    const auto original=h.film;
    require(!advanceCpuFilmCandidate(h,p,1,std::vector<CpuFilmForcing>(9),drive(h),c,error),"aggregate donor rejects despite simultaneous incoming credit");
    require(error.find("donor")!=std::string::npos,"donor rejection identified");
    require(c.film.size()==1&&c.film[0].mass==987,"failed candidate output rollback");
    for(std::size_t i=0;i<h.film.size();++i)require(h.film[i].mass==original[i].mass&&h.film[i].enthalpy==original[i].enthalpy,"accepted state immutable");
}
void conservativeConduction() {
    auto p=physics();p.liquid.conductivity=200;auto h=state(p,3,3,{});auto c=advance(h,p,.02);
    require(c.filmAux[0].temperature>h.filmAux[0].temperature,"tangential conduction heats cold face");
    require(c.filmAux[8].temperature<h.filmAux[8].temperature,"tangential conduction cools hot face");
    require(near(c.report.internalEnergyAfter,energy(h,p)),"conduction is antisymmetric");
    for(std::size_t f=0;f<h.film.size();++f)require(c.film[f].mass==h.film[f].mass,"conduction does not change mass");
}
void surfaceRhsOwnership() {
    auto p=physics();p.nElements=1;for(int species=0;species<Ns;++species)p.species[species].element[0]=1;auto h=state(p,1,1,tangentX*.2);h.surface.gasFace[0]=0;
    auto d=drive(h);d[0].shear=tangentX*10;d[0].hasCoupledNormalTrace=true;
    d[0].interfaceNormalVelocity=.01;d[0].solidNormalVelocity=-.02;
    std::vector<FilmQ> rates;std::vector<FilmRateBudget> budgets;std::string error;
    std::vector<FilmAux> aux;std::vector<Vec3> gradients;std::vector<Real> sideVolume;
    const bool ok=evaluateCpuFilmTransport(h,p,.1,d,rates,budgets,error,&aux,&gradients,&sideVolume);
    require(ok,"interface-owner RHS: "+error);
    require(aux.size()==1&&gradients.size()==1&&sideVolume.size()==1,"RHS refreshes all optional profile outputs");
    require(near(dot(aux[0].topVelocity,tangentX),.21),"refreshed profile uses actual recorded shear");
    require(rates[0].enthalpy==0,"RHS excludes top/bottom owned interface work");
    require(budgets[0].supportPower==0&&budgets[0].prescribedTopPower==0,"RHS ownership audit");
}
void forcingAndKineticSeparation() {
    auto p=physics();auto h=state(p,1,1,{});h.surface.gasFace[0]=0;auto d=drive(h);
    d[0].shear=tangentX*10;d[0].hasCoupledNormalTrace=true;
    d[0].interfaceNormalVelocity=.01;d[0].solidNormalVelocity=-.02;
    std::vector<CpuFilmForcing> f(1);f[0].energy=31;f[0].radiationEnergy=7;
    f[0].interfaceKineticOutflow=3;f[0].topWorkIncluded=f[0].bottomWorkIncluded=true;
    CpuFilmCandidate c;std::string error;const bool ok=advanceCpuFilmCandidate(h,p,.1,f,d,c,error);
    require(ok,"owned source: "+error);
    require(near(c.report.internalEnergyAfter-c.report.internalEnergyBefore,41),"physical source, radiation and kinetic advection added once");
    require(c.report.budgetDelta.supportWork==0,"no duplicated top or bottom work");
    require(near(c.report.budgetDelta.radiation,7),"radiation added once");
    require(near(c.report.budgetDelta.exchangeEnergy[FilmParticipant],31),"kinetic advection excluded from physical exchange");
    require(near(c.report.kineticDefect,c.filmAux[0].kineticEnergy-h.filmAux[0].kineticEnergy+3),"kinetic defect separate from heating");
    d[0].hasCoupledNormalTrace=false;
    require(!advanceCpuFilmCandidate(h,p,.1,f,d,c,error),"missing actual coupled trace refused");
}
void undeclaredInflowAndOutflow() {
    auto p=physics();p.nElements=1;for(int species=0;species<Ns;++species)p.species[species].element[0]=1;auto h=state(p,1,1,tangentX*.2);
    h.surface.edgeOwner={0};h.surface.edgeNeighbour={-1};h.surface.edgeLength={1};
    h.surface.edgeConormal={tangentX};h.surface.edgeOwnerOffset={tangentX*.5};h.surface.edgeNeighbourOffset={Vec3{}};
    auto c=advance(h,p,.1);require(near(c.film[0].mass,.98),"open edge outflow transports actual mass");
    require(near(c.report.budgetDelta.boundaryMass,.02),"outflow mass budget");require(near(c.report.budgetDelta.boundaryElements[0],.02),"open film outflow elemental budget");
    h.surface.baseVelocity[0]=-tangentX*.2;std::string error;
    require(!advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),drive(h),c,error),"undeclared edge inflow refused");
    require(error.find("inflow")!=std::string::npos,"inflow policy diagnostic");
}
void radiationAndSourceLimits() {
    auto p=physics();auto h=state(p,1,1,{});auto d=drive(h);std::vector<CpuFilmForcing> f(1);std::string error;
    Real dt0=0,dt1=0;require(estimateCpuFilmStep(h,p,f,d,1000,1000,dt0,error),"unforced step");
    p.enableRadiation=true;p.emissivity=1;p.ambientTemperature=100;
    require(estimateCpuFilmStep(h,p,f,d,1000,1000,dt1,error),"radiation step");
    require(dt1<dt0,"radiation local restriction");
    auto c=advance(h,p,.1);require(c.report.internalEnergyAfter<c.report.internalEnergyBefore,"gray radiation cools film");
    require(near(c.report.internalEnergyAfter-c.report.internalEnergyBefore,c.report.budgetDelta.radiation),"radiation budget closes");
    p.enableRadiation=false;f[0].energy=1e9;
    require(estimateCpuFilmStep(h,p,f,d,1,1,dt1,error)&&dt1<.01,"source temperature limit");
    f[0]=CpuFilmForcing{};f[0].withdrawnMass=1.1;f[0].withdrawnSpecies[0]=.22;f[0].withdrawnSpecies[1]=.88;
    require(!advanceCpuFilmCandidate(h,p,.1,f,d,c,error),"net-zero forcing cannot hide gross donor withdrawal");
}
void secondOrderAndPeriodicOffsets() {
    auto p=physics();p.spatialOrder=2;auto h=state(p,3,3,tangentX*.2+tangentY*.15);
    auto c=advance(h,p,.02);Real mass=0,expected=0;
    for(std::size_t f=0;f<h.film.size();++f){mass+=c.film[f].mass;expected+=h.film[f].mass;}
    require(near(mass,expected),"limited-linear mass conservation");
    require(near(c.report.internalEnergyAfter,energy(h,p)),"limited-linear energy conservation");
    // Centres are intentionally wrong: periodic image offsets, not wrapped centre differences, own the stencil.
    for(auto& centre:h.surface.centre)centre={999,999,999};
    auto other=advance(h,p,.02);
    for(std::size_t f=0;f<h.film.size();++f)require(other.film[f].mass==c.film[f].mass,"periodic image edge offsets used");
}
void exactDryoutRemainder() {
    auto p=physics();auto h=state(p,1,1,{});h.surface.gasFace[0]=0;auto d=drive(h);d[0].hasCoupledNormalTrace=true;
    std::vector<CpuFilmForcing> f(1);f[0].mass=-.03;
    f[0].energy=-40.17;f[0].interfaceKineticOutflow=.12;f[0].radiationEnergy=.18;
    f[0].phaseMass=-(h.film[0].mass+f[0].mass);
    for(int s=0;s<Ns;++s) {
        f[0].species[s]=-.03*h.film[0].species[s]/h.film[0].mass;
        f[0].phaseSpecies[s]=-(h.film[0].species[s]+f[0].species[s]);
    }
    f[0].phaseKineticOutflow=.13;
    const Real available=filmInternalEnergy(h.film[0],h.filmAux[0].pressure,p)+f[0].energy+f[0].interfaceKineticOutflow+f[0].radiationEnergy;
    f[0].phaseEnergy=-(available+f[0].phaseKineticOutflow);
    f[0].topWorkIncluded=f[0].bottomWorkIncluded=true;f[0].exactPhaseEvent=true;
    Real estimated=0;std::string estimateError;require(estimateCpuFilmStep(h,p,f,d,.1,.1,estimated,estimateError),"terminal phase estimate");require(estimated==.1,"proven exact terminal phase event is not asymptotically halved");
    CpuFilmCandidate c;std::string error;const bool ok=advanceCpuFilmCandidate(h,p,.1,f,d,c,error);
    require(ok,"exact physical dryout remainder: "+error);
    require(c.film[0].mass==0&&c.film[0].enthalpy==0,"dryout exactly empties H and species without clipping");
    for(int s=0;s<Ns;++s)require(c.film[0].species[s]==0,"exact dryout species");
    require(c.report.phaseEvent,"dryout reports phase event");
}
void exactGasDryout(){auto p=physics();auto h=state(p,1,1,{});h.surface.gasFace[0]=0;auto d=drive(h);d[0].hasCoupledNormalTrace=true;
    std::vector<CpuFilmForcing> f(1);f[0].mass=-h.film[0].mass;for(int species=0;species<Ns;++species)f[0].species[species]=-h.film[0].species[species];
    f[0].energy=-filmInternalEnergy(h.film[0],h.filmAux[0].pressure,p);f[0].topWorkIncluded=f[0].bottomWorkIncluded=true;f[0].exactPhaseEvent=true;
    Real dt=0;std::string error;require(estimateCpuFilmStep(h,p,f,d,1,1,dt,error)&&dt==1,"actual gas-driven exact dryout interval");
    CpuFilmCandidate candidate;require(advanceCpuFilmCandidate(h,p,dt,f,d,candidate,error),"gas dryout accepted: "+error);require(candidate.film[0].mass==0&&candidate.film[0].enthalpy==0,"gas dryout exactly empty");}
void edgeSweepVolumeRate() {
    const auto p=physics();auto h=state(p,3,3,tangentX*.2+tangentY*.15);
    h.surface.sweptEdgeArea.assign(h.surface.edgeOwner.size(),0);h.surface.sweptEdgeArea[0]=.01;
    std::vector<FilmQ> rates;std::vector<FilmRateBudget> budgets;std::vector<Real> side;std::string error;
    require(evaluateCpuFilmTransport(h,p,.1,drive(h),rates,budgets,error,nullptr,nullptr,&side),"swept-edge RHS");
    require(near(side[0],.0001)&&near(side[1],-.0001),"actual swept edge volume antisymmetric");
}
void bodySupportTopWorkOnce() {
    auto p=physics();p.gravity=tangentX*2;auto h=state(p,1,1,tangentX*.2);
    h.surface.prescribedTopTraction={tangentX*10};
    const auto profile=filmProfile(h.filmAux[0].thickness,p.liquidViscosity,h.surface.baseVelocity[0],tangentX*10,-p.gravity*p.liquid.rho);
    const Real body=h.film[0].mass*dot(p.gravity,profile.meanVelocity);
    const Real support=-dot(profile.bottomShear,h.surface.baseVelocity[0]);
    const Real top=dot(tangentX*10,profile.topVelocity);
    const auto c=advance(h,p,.1);
    require(near(c.report.budgetDelta.bodyWork,.1*body),"body power once");
    require(near(c.report.budgetDelta.supportWork,.1*(support+top)),"top and support power once");
    require(near(c.report.internalEnergyAfter-c.report.internalEnergyBefore,.1*(body+support+top),2e-10),"mechanical input is not doubled by dissipation or kinetic defect");
}
void rejectsMalformedGrossWithdrawals() {
    auto p=physics();auto h=state(p,1,1,{});std::vector<CpuFilmForcing> f(1);
    f[0].withdrawnMass=.1;f[0].withdrawnSpecies[0]=.2;
    CpuFilmCandidate c;std::string error;
    require(!advanceCpuFilmCandidate(h,p,.1,f,drive(h),c,error),"gross species must close to gross mass");
}
void localStep() {
    auto p=physics();auto h=state(p,3,3,tangentX*.75+tangentY*.75);auto d=drive(h);
    Real dt=999;std::string error;std::vector<CpuFilmForcing> f(9);
    require(estimateCpuFilmStep(h,p,f,d,10,10,dt,error),"CPU step estimate: "+error);
    require(dt>0&&dt<1,"aggregate advection limits CPU dt");
    p.cfl=1e-12;p.maxDt=1e-15;p.gasConductivity=1e30;
    Real other=0;require(estimateCpuFilmStep(h,p,f,d,10,10,other,error),"gas-independent estimate");
    require(dt==other,"gas CFL/maxDt/diffusion absent from film step");
}
}
int main(){transportBothTangents();pressureStorage();donorRollback();localStep();conservativeConduction();surfaceRhsOwnership();forcingAndKineticSeparation();undeclaredInflowAndOutflow();radiationAndSourceLimits();secondOrderAndPeriodicOffsets();exactDryoutRemainder();exactGasDryout();edgeSweepVolumeRate();bodySupportTopWorkOnce();rejectsMalformedGrossWithdrawals();std::cout<<"CPU film tests passed\n";}
