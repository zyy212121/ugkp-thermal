#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include "verification/FilmInventory.H"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
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
template<class State> Real energy(const State& h,const PhysicsConfig& p) {Real sum=0;for(std::size_t i=0;i<h.film.size();++i)sum+=h.film[i].enthalpy-h.filmAux[i].pressure*h.film[i].mass/p.liquid.rho;return sum;}
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
        require(near(energy(c,p),energy(h,p)),"closed physical energy conservation");
        require(std::abs(energy(c,p)-energy(h,p))<1e-7,"independent closed physical energy difference");
    }
}
void pressureStorage() {
    const auto p=physics();auto h=state(p,1,1,{});auto d=drive(h);d[0].pressure=200000;d[0].endpointPressure=300000;
    CpuFilmCandidate c;std::string error;
    require(advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),d,c,error),"pressure update: "+error);
    require(near(c.film[0].enthalpy-h.film[0].enthalpy,200),"endpoint pV exactly once");
    require(near(c.filmAux[0].temperature,h.filmAux[0].temperature),"pressure change preserves temperature");
    require(near(energy(c,p),energy(h,p)),"pressure change preserves physical U");
    require(near((c.filmAux[0].pressure*c.film[0].mass-h.filmAux[0].pressure*h.film[0].mass)/p.liquid.rho,200),"actual endpoint pressure-volume storage");
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
    require(near(energy(c,p),energy(h,p)),"conduction is antisymmetric");
    for(std::size_t f=0;f<h.film.size();++f)require(c.film[f].mass==h.film[f].mass,"conduction does not change mass");
}
void surfaceRhsOwnership() {
    auto p=physics();p.nElements=1;for(int species=0;species<Ns;++species)p.species[species].element[0]=1;auto h=state(p,1,1,tangentX*.2);h.surface.gasFace[0]=0;
    auto d=drive(h);d[0].shear=tangentX*10;d[0].hasCoupledNormalTrace=true;
    d[0].interfaceNormalVelocity=.01;d[0].solidNormalVelocity=-.02;
    std::vector<FilmQ> rates;std::string error;
    std::vector<FilmAux> aux;std::vector<Vec3> gradients;std::vector<Real> sideVolume;
    const bool ok=evaluateCpuFilmTransport(h,p,.1,d,rates,error,&aux,&gradients,&sideVolume);
    require(ok,"interface-owner RHS: "+error);
    require(aux.size()==1&&gradients.size()==1&&sideVolume.size()==1,"RHS refreshes all optional profile outputs");
    require(near(dot(aux[0].topVelocity,tangentX),.21),"refreshed profile uses actual recorded shear");
    require(rates[0].enthalpy==0,"RHS excludes top/bottom owned interface work");
    require(rates[0].mass==0,"interface-owner transport excludes integrated mass sources");
}
void forcingAndKineticSeparation() {
    auto p=physics();auto h=state(p,1,1,{});h.surface.gasFace[0]=0;auto d=drive(h);
    d[0].shear=tangentX*10;d[0].hasCoupledNormalTrace=true;
    d[0].interfaceNormalVelocity=.01;d[0].solidNormalVelocity=-.02;
    std::vector<CpuFilmForcing> f(1);f[0].energy=31;f[0].radiationEnergy=7;
    f[0].interfaceKineticOutflow=3;f[0].topWorkIncluded=f[0].bottomWorkIncluded=true;
    CpuFilmCandidate c;std::string error;const bool ok=advanceCpuFilmCandidate(h,p,.1,f,d,c,error);
    require(ok,"owned source: "+error);
    require(near(energy(c,p)-energy(h,p),41),"physical source, radiation and kinetic advection added once");
    require(near(energy(c,p)-energy(h,p)-f[0].radiationEnergy-f[0].interfaceKineticOutflow,f[0].energy),"physical source remains distinct from radiation and kinetic channel");
    require(c.film[0].mass==h.film[0].mass,"energy forcing does not create film mass");
    d[0].hasCoupledNormalTrace=false;
    require(!advanceCpuFilmCandidate(h,p,.1,f,d,c,error),"missing actual coupled trace refused");
}
void signedOpenEdgeInflowAndOutflow() {
    auto p=physics();p.nElements=1;for(int species=0;species<Ns;++species)p.species[species].element[0]=1;auto h=state(p,1,1,tangentX*.2);
    h.surface.edgeOwner={0};h.surface.edgeNeighbour={-1};h.surface.edgeLength={1};
    h.surface.edgeConormal={tangentX};h.surface.edgeOwnerOffset={tangentX*.5};h.surface.edgeNeighbourOffset={Vec3{}};
    auto c=advance(h,p,.1);require(near(c.film[0].mass,.98),"open edge outflow transports actual mass");
    require(near(h.film[0].mass-c.film[0].mass,.02),"actual outflow mass loss");
    Real elementalOut=0;for(int species=0;species<Ns;++species)elementalOut+=(h.film[0].species[species]-c.film[0].species[species])*p.species[species].element[0];
    require(near(elementalOut,.02),"outflow actual elemental loss");
    h.surface.baseVelocity[0]=-tangentX*.2;std::string error;
    require(advanceCpuFilmCandidate(h,p,.1,std::vector<CpuFilmForcing>(1),drive(h),c,error),"signed owner-extrapolated inflow: "+error);
    require(near(c.film[0].mass,1.02),"signed inflow adds owner-extrapolated mass");
    require(near(c.film[0].enthalpy,1.02*h.film[0].enthalpy),"signed inflow thermal inventory follows owner enthalpy");
    std::vector<FilmQ> rates;
    require(evaluateCpuFilmTransport(h,p,.1,drive(h),rates,error),"observe actual inflow transport RHS: "+error);
    require(rates[0].mass>0&&near(rates[0].mass,.2),"signed negative outward flux gives positive mass RHS");
    Real elementalIn=0;
    for(int species=0;species<Ns;++species){
        require(near(c.film[0].species[species],1.02*h.film[0].species[species]),"signed inflow species inventory");
        require(near(c.film[0].species[species]-h.film[0].species[species],.1*rates[0].species[species]),"species RHS and actual inventory close");
        elementalIn+=(c.film[0].species[species]-h.film[0].species[species])*p.species[species].element[0];
    }
    require(near(elementalIn,.02),"signed inflow actual elemental gain");
    require(near((-rates[0].enthalpy),-.2*h.film[0].enthalpy),"actual RHS edge energy matches independent owner enthalpy flux");

    const Real pressureWork=-h.filmAux[0].pressure*rates[0].mass/p.liquid.rho;
    require(near(energy(c,p)-energy(h,p)+.1*(-rates[0].enthalpy),.1*pressureWork),"signed inflow physical energy plus pressure work closes");
}
void radiationAndSourceLimits() {
    auto p=physics();auto h=state(p,1,1,{});auto d=drive(h);std::vector<CpuFilmForcing> f(1);std::string error;
    Real dt0=0,dt1=0;require(estimateCpuFilmStep(h,p,f,d,1000,1000,dt0,error),"unforced step");
    p.enableRadiation=true;p.emissivity=1;p.ambientTemperature=100;
    require(estimateCpuFilmStep(h,p,f,d,1000,1000,dt1,error),"radiation step");
    require(dt1<dt0,"radiation local restriction");
    auto c=advance(h,p,.1);require(energy(c,p)<energy(h,p),"gray radiation cools film");
    require(near(energy(c,p)-energy(h,p),.1*h.surface.area[0]*5.670374419e-8*p.emissivity*(std::pow(p.ambientTemperature,4)-std::pow(h.filmAux[0].temperature,4))),"independent Stefan-Boltzmann energy closes");
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
    require(near(energy(c,p),energy(h,p)),"limited-linear energy conservation");
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
    require(h.film[0].mass>0&&c.film[0].mass==0,"actual wet-to-dry phase transition");
}
void exactGasDryout(){auto p=physics();auto h=state(p,1,1,{});h.surface.gasFace[0]=0;auto d=drive(h);d[0].hasCoupledNormalTrace=true;
    std::vector<CpuFilmForcing> f(1);f[0].mass=-h.film[0].mass;for(int species=0;species<Ns;++species)f[0].species[species]=-h.film[0].species[species];
    f[0].energy=-filmInternalEnergy(h.film[0],h.filmAux[0].pressure,p);f[0].topWorkIncluded=f[0].bottomWorkIncluded=true;f[0].exactPhaseEvent=true;
    Real dt=0;std::string error;require(estimateCpuFilmStep(h,p,f,d,1,1,dt,error)&&dt==1,"actual gas-driven exact dryout interval");
    CpuFilmCandidate candidate;require(advanceCpuFilmCandidate(h,p,dt,f,d,candidate,error),"gas dryout accepted: "+error);require(candidate.film[0].mass==0&&candidate.film[0].enthalpy==0,"gas dryout exactly empty");}
void edgeSweepVolumeRate() {
    const auto p=physics();auto h=state(p,3,3,tangentX*.2+tangentY*.15);
    h.surface.sweptEdgeArea.assign(h.surface.edgeOwner.size(),0);h.surface.sweptEdgeArea[0]=.01;
    std::vector<FilmQ> rates;std::vector<Real> side;std::string error;
    require(evaluateCpuFilmTransport(h,p,.1,drive(h),rates,error,nullptr,nullptr,&side),"swept-edge RHS");
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
    require(mag(c.filmAux[0].meanVelocity-profile.meanVelocity)<1e-12,"actual endpoint mean velocity agrees with independent mechanical profile");
    require(mag(c.filmAux[0].topVelocity-profile.topVelocity)<1e-12,"actual endpoint top velocity agrees with independent mechanical profile");
    require(near(energy(c,p)-energy(h,p),.1*(body+support+top),2e-10),"mechanical input is not doubled by dissipation or kinetic defect");
}
void rejectsMalformedGrossWithdrawals() {
    auto p=physics();auto h=state(p,1,1,{});std::vector<CpuFilmForcing> f(1);
    f[0].withdrawnMass=.1;f[0].withdrawnSpecies[0]=.2;
    CpuFilmCandidate c;std::string error;
    ugkp::diagnostics::DeviceContext diagnostics{};p.diagnostics=&diagnostics;
    require(advanceCpuFilmCandidate(h,p,.1,f,drive(h),c,error),"finite gross-species discrepancy retains update");
    require(diagnostics.warnings[0].count>0&&!diagnostics.fatalClaim,"gross species discrepancy warns");
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

void twoUlpPressureOpenEdge() {
    auto p=physics();p.liquid.cp0=1000;p.liquid.e0=100000;p.liquidViscosity=.01;p.liquidReferencePressure=171600;
    HostState h;const Vec3 n{1,0,0},t{0,1,0};
    const Real pressure0=178167.05269620317,pressure1=std::nextafter(std::nextafter(pressure0,0.0),0.0);
    h.surface.area=h.surface.oldArea={.25,.25};h.surface.normal={n,n};h.surface.baseVelocity={{},{}};h.surface.gasFace={0,1};
    h.surface.centre={{1,.25,.25},{1,.75,.25}};
    h.surface.edgeOwner={0,0};h.surface.edgeNeighbour={-1,1};h.surface.edgeLength={.5,.5};
    h.surface.edgeConormal={-t,t};h.surface.edgeOwnerOffset={t*-.25,t*.25};h.surface.edgeNeighbourOffset={{},t*-.25};h.surface.sweptEdgeArea={0,0};
    h.film.resize(2);h.filmAux.resize(2);std::vector<CpuFilmDrive> d(2);std::vector<CpuFilmForcing> forcing(2);
    for(int f=0;f<2;++f){const Real pressure=f?pressure1:pressure0;auto& q=h.film[f];q.mass=.0025;q.species[0]=q.mass*.2;q.species[1]=q.mass-q.species[0];q.enthalpy=q.mass*liquidH(p,350,pressure);h.filmAux[f].normal=n;require(recoverFilm(q,.25,pressure,p,h.filmAux[f]),"two-ulp fixture thermal inventory");d[f].pressure=d[f].endpointPressure=pressure;d[f].hasCoupledNormalTrace=true;forcing[f].topWorkIncluded=forcing[f].bottomWorkIncluded=true;}
    const Real fullDt=1.2207031249998656e-7;
    for(int halves:{0,30}){const Real requested=std::ldexp(fullDt,-halves);std::string error;std::vector<FilmQ> rates;std::vector<FilmAux> profiles;
        require(evaluateCpuFilmTransport(h,p,requested,d,rates,error,&profiles,nullptr,nullptr,nullptr),"two-ulp actual profile observation: "+error);
        const Real signedLengthRate=h.surface.edgeLength[0]*dot(profiles[0].meanVelocity,h.surface.edgeConormal[0]);
        require(std::isfinite(signedLengthRate)&&signedLengthRate<0,"two-ulp pressure difference produces strictly negative finite open-edge rate");
        const bool ordinaryTransport=evaluateCpuFilmTransport(h,p,requested,d,rates,error);
        require(ordinaryTransport,"two-ulp finite signed transport: "+error);
        Real allowed=0;require(estimateCpuFilmStep(h,p,forcing,d,requested,requested,allowed,error),"two-ulp physical step estimate: "+error);require(allowed==requested,"two-ulp pressure variation does not shorten physically allowed dt");
        CpuFilmCandidate c;require(advanceCpuFilmCandidate(h,p,requested,forcing,d,c,error),"two-ulp signed candidate: "+error);
        Real massRhs=0;for(const auto& rate:rates)massRhs+=rate.mass;
        const Real signedBoundaryMass=-(requested*massRhs);
        require(signedBoundaryMass<0,"two-ulp negative outward boundary mass retained by actual RHS");
        const Real independentFlux=p.liquid.rho*profiles[0].thickness*signedLengthRate;
        require(near(-massRhs,independentFlux),"two-ulp boundary flux matches independently sampled profile");
        Real deltaMass=0;for(int f=0;f<2;++f)deltaMass+=c.film[f].mass-h.film[f].mass;
        require(near(deltaMass,-signedBoundaryMass),"two-ulp actual mass boundary closure");
        for(int species=0;species<Ns;++species){Real delta=0,rhs=0;for(int f=0;f<2;++f){delta+=c.film[f].species[species]-h.film[f].species[species];rhs+=rates[f].species[species];}require(near(delta,requested*rhs),"two-ulp actual species boundary closure");}
        Real energyRhs=0;for(const auto& rate:rates)energyRhs+=rate.enthalpy;
        require(near(energy(c,p)-energy(h,p),requested*energyRhs),"two-ulp actual RHS and independent physical energy close");
        std::cout<<std::setprecision(17)<<"two-ulp signed edge half="<<halves<<" rate="<<signedLengthRate<<" requested="<<requested<<" allowed="<<allowed<<'\n';
    }
}

void independentKineticInventoryFromVelocities() {
    FilmQ q;FilmAux a;q.mass=2;
    a.baseVelocity=a.meanVelocity=a.topVelocity={1,2,-2};
    require(std::abs(verification::filmKineticEnergy(q,a)-9)<1e-15L,"constant profile physical kinetic inventory");
    q.mass=3;a.baseVelocity={};a.meanVelocity={1,0,0};a.topVelocity={2,0,0};
    require(std::abs(verification::filmKineticEnergy(q,a)-2)<1e-15L,"linear profile physical kinetic inventory");
    q.mass=15;a.meanVelocity={2.0/3,0,0};a.topVelocity={1,0,0};
    require(std::abs(verification::filmKineticEnergy(q,a)-4)<1e-14L,"quadratic profile physical kinetic inventory");
    q.mass=3;a.normal={0,0,1};a.baseVelocity={1,0,99};
    a.meanVelocity={1,0,2};a.topVelocity={1,0,3};
    require(std::abs(verification::filmKineticEnergy(q,a)-8)<1e-14L,"coupled normal trace supplies the physical bottom velocity");
}
void profileRejectsUnavailablePhysicalInputs() {
    auto p=physics();FilmAux a;a.area=1;a.thickness=.1;a.baseVelocity={1,0,0};
    p.liquidViscosity=0;
    require(!updateFilmProfileState(a,p,{0,0,1},{},{},false),"zero viscosity has no physical profile");
    p.liquidViscosity=1;a.thickness=-.1;
    require(!updateFilmProfileState(a,p,{0,0,1},{},{},false),"negative thickness has no physical profile");
}
void finiteProfileAtLargeAreaSmallThickness() {
    auto p=physics();p.liquid.rho=4;
    FilmAux aux;aux.area=1e308;aux.thickness=1e-308;
    aux.baseVelocity={1,0,0};aux.meanVelocity=aux.topVelocity=aux.baseVelocity;
    require(updateFilmProfileState(aux,p,{0,0,1},{},{},false),"finite thin-film velocity with representable physical mass");
    require(aux.meanVelocity.x==1&&aux.meanVelocity.y==0&&aux.meanVelocity.z==0,"thin-film mean velocity");
    require(aux.topVelocity.x==1&&aux.topVelocity.y==0&&aux.topVelocity.z==0,"thin-film top velocity");
}

}
int main(){independentKineticInventoryFromVelocities();profileRejectsUnavailablePhysicalInputs();finiteProfileAtLargeAreaSmallThickness();twoUlpPressureOpenEdge();transportBothTangents();pressureStorage();donorRollback();localStep();conservativeConduction();surfaceRhsOwnership();forcingAndKineticSeparation();signedOpenEdgeInflowAndOutflow();radiationAndSourceLimits();secondOrderAndPeriodicOffsets();exactDryoutRemainder();exactGasDryout();edgeSweepVolumeRate();bodySupportTopWorkOnce();rejectsMalformedGrossWithdrawals();std::cout<<"CPU film tests passed\n";}
