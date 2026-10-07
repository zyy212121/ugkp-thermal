#define main existing_geometry_fixture_main
#include "test_sweep_constraints.cpp"
#undef main
#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include <cmath>
#include <iomanip>
namespace {
PhysicsConfig curvedFilmPhysics() {
    PhysicsConfig p;p.enableFilm=true;p.spatialOrder=1;p.liquid.rho=1000;p.liquid.cp0=1000;
    p.liquid.Tmin=100;p.liquid.Tmax=1500;p.liquidViscosity=1;return p;
}
HostState curvedFilmState(const PhysicsConfig& p) {
    HostState h=curvedOffsetPair();std::string error;
    require(rebuildTrajectorySurface(h,h.gasMesh,h.solidMesh,1,h.surface,error),"curved surface: "+error);
    h.film.resize(h.surface.area.size());
    for(std::size_t f=0;f<h.film.size();++f) {
        auto& q=h.film[f];q.mass=1000*.02*h.surface.area[f];q.species[0]=q.mass;
        q.enthalpy=q.mass*liquidH(p,500,100000);
        require(recoverFilm(q,h.surface.area[f],100000,p,h.filmAux[f]),"curved film EOS");
    }
    return h;
}
std::vector<CpuFilmDrive> curvedDrive(const HostState& h) {
    std::vector<CpuFilmDrive> d(h.film.size());for(auto& x:d){x.pressure=100000;x.hasCoupledNormalTrace=true;}return d;
}
void stationaryCurvedWetSurface() {
    const auto p=curvedFilmPhysics();const auto h=curvedFilmState(p);std::string error;
    CpuFilmCandidate c;const bool ok=advanceCpuFilmCandidate(h,p,.01,std::vector<CpuFilmForcing>(h.film.size()),curvedDrive(h),c,error);
    require(ok,"stationary curved wet geometry must be consumable by CPU film: "+error);
    for(std::size_t f=0;f<h.film.size();++f) {
        require(c.film[f].mass==h.film[f].mass,"stationary curved film mass unchanged");
        require(c.film[f].enthalpy==h.film[f].enthalpy,"stationary curved film energy unchanged");
    }
}
void curvedConductionConservesPhysicalEnergy() {
    auto p=curvedFilmPhysics();p.liquid.conductivity=10;auto h=curvedFilmState(p);std::string error;
    h.film[0].enthalpy=h.film[0].mass*liquidH(p,400,100000);h.film[1].enthalpy=h.film[1].mass*liquidH(p,600,100000);
    CpuFilmCandidate c;const bool ok=advanceCpuFilmCandidate(h,p,.01,std::vector<CpuFilmForcing>(h.film.size()),curvedDrive(h),c,error);
    require(ok,"curved conduction: "+error);
    require(c.filmAux[0].temperature>400&&c.filmAux[1].temperature<600,"curved shared edge conducts hot to cold");
    require(absValue(c.report.internalEnergyAfter-c.report.internalEnergyBefore)<1e-8,"curved internal edge conserves physical energy");
    require(absValue(c.report.reducedEnergyResidual)<1e-8,"curved conduction energy residual");
}

void retainInternalEdges(SurfaceMesh& m) {
    const auto old=m;m.edgeOwner.clear();m.edgeNeighbour.clear();m.edgeLength.clear();m.edgeConormal.clear();
    m.edgeOwnerOffset.clear();m.edgeNeighbourOffset.clear();m.sweptEdgeArea.clear();
    for(std::size_t e=0;e<old.edgeOwner.size();++e)if(old.edgeNeighbour[e]>=0) {
        m.edgeOwner.push_back(old.edgeOwner[e]);m.edgeNeighbour.push_back(old.edgeNeighbour[e]);
        m.edgeLength.push_back(old.edgeLength[e]);m.edgeConormal.push_back(old.edgeConormal[e]);
        m.edgeOwnerOffset.push_back(old.edgeOwnerOffset[e]);m.edgeNeighbourOffset.push_back(old.edgeNeighbourOffset[e]);
        m.sweptEdgeArea.push_back(0);
    }
}
void curvedShearTransportUsesSharedMetric() {
    const auto p=curvedFilmPhysics();auto h=curvedFilmState(p);retainInternalEdges(h.surface);
    require(h.surface.edgeOwner.size()==1,"one actual curved internal edge");
    h.surface.gasFace.assign(2,-1);h.surface.prescribedTopTraction.resize(2);
    const auto co=h.surface.edgeConormal[0];const int owner=h.surface.edgeOwner[0],other=h.surface.edgeNeighbour[0];
    const Real speed[2]={.3,.1},temperature[2]={400,600},dt=.01;Real shearPower=0;
    for(int f=0;f<2;++f) {
        const Vec3 n=h.surface.normal[f],t=normalized(co-n*dot(co,n));const Real mass=h.film[f].mass;
        h.film[f].enthalpy=mass*liquidH(p,temperature[f],100000);
        h.surface.prescribedTopTraction[f]=t*(2*p.liquidViscosity*speed[f]/.02);
        h.filmAux[f].meanVelocity=t*speed[f];h.filmAux[f].topVelocity=t*(2*speed[f]);
        h.filmAux[f].kineticEnergy=(2.0/3)*mass*speed[f]*speed[f];
        shearPower+=h.surface.area[f]*4*p.liquidViscosity*speed[f]*speed[f]/.02;
    }
    const Real edgeSpeed=.5*dot(h.filmAux[owner].meanVelocity+h.filmAux[other].meanVelocity,co);
    const Real massFlux=1000*.02*h.surface.edgeLength[0]*edgeSpeed;
    const Real hFlux=massFlux*(p.liquid.cp0*temperature[owner]+100000/p.liquid.rho);
    // Integral of (1/2 rho |u|^2)(u dot conormal) for u(z)=2 U z/delta.
    const Vec3 no=h.surface.normal[owner],to=normalized(co-no*dot(co,no));
    const Real kineticFlux=1000*.02*h.surface.edgeLength[0]*std::pow(speed[owner],3)*dot(to,co);
    CpuFilmCandidate c;std::string error;const bool ok=advanceCpuFilmCandidate(h,p,dt,std::vector<CpuFilmForcing>(2),curvedDrive(h),c,error);
    require(ok,"nonzero curved shear transport: "+error);
    require(absValue(c.report.edgeFlux[0].mass-massFlux)<1e-12,"curved mass flux uses independently projected adjacent velocities");
    require(absValue(c.report.edgeFlux[0].species[0]-massFlux)<1e-12,"curved species donor flux");
    require(absValue(c.report.edgeFlux[0].enthalpy-hFlux)<1e-8,"curved physical enthalpy edge flux");
    require(absValue(c.film[owner].mass-h.film[owner].mass+dt*massFlux)<1e-12,"curved owner loses signed edge mass");
    require(absValue(c.film[other].mass-h.film[other].mass-dt*massFlux)<1e-12,"curved neighbour receives the same edge mass");
    require(absValue(c.report.rateBudget[owner].edgeKineticOutflow-kineticFlux)<1e-12,"curved kinetic flux uses the same shared metric");
    require(absValue(c.report.rateBudget[other].edgeKineticOutflow+kineticFlux)<1e-12,"curved kinetic edge transfer antisymmetric");
    require(absValue(c.report.budgetDelta.supportWork-dt*shearPower)<1e-10,"curved shear work and internal pressure work close");
    require(absValue(c.report.internalEnergyAfter-c.report.internalEnergyBefore-dt*shearPower)<1e-8,"curved film energy changes only by external shear work");
    require(absValue(c.report.reducedEnergyResidual)<1e-8,"curved transport and work budget residual");
}
void hingeProjectionHasSecondOrderAngularConsistency() {
    const auto p=curvedFilmPhysics();Real previous=0;
    for(Real scale:std::vector<Real>{1,.5,.25,.125}) {
        auto h=curvedFilmState(p);for(auto& x:h.solidMesh.points)if(x.z>.8)x.z=1+scale*(x.z-1);
        h.surface.gasFace.assign(2,-1);std::string error;
        require(rebuildGeometry(h.solidMesh,error),error);
        require(rebuildTrajectorySurface(h,h.gasMesh,h.solidMesh,1,h.surface,error),error);
        retainInternalEdges(h.surface);h.surface.prescribedTopTraction.resize(2);
        const Vec3 co=h.surface.edgeConormal[0];
        for(int f=0;f<2;++f) {
            const Vec3 n=h.surface.normal[f],t=normalized(co-n*dot(co,n));
            h.film[f].mass=1000*.02*h.surface.area[f];h.film[f].species[0]=h.film[f].mass;
            h.film[f].enthalpy=h.film[f].mass*liquidH(p,500,100000);
            require(recoverFilm(h.film[f],h.surface.area[f],100000,p,h.filmAux[f]),"angular EOS");
            h.surface.prescribedTopTraction[f]=t*(2*p.liquidViscosity*.2/.02);
        }
        CpuFilmCandidate c;const bool ok=advanceCpuFilmCandidate(h,p,.001,std::vector<CpuFilmForcing>(2),curvedDrive(h),c,error);
        require(ok,"angular refinement transport: "+error);
        const Real intrinsic=1000*.02*h.surface.edgeLength[0]*.2;
        const Real normalCosine=dot(h.surface.normal[0],h.surface.normal[1]);
        const Real expectedRatio=std::sqrt((1+normalCosine)/2);
        const Real ratio=c.report.edgeFlux[0].mass/intrinsic;
        require(absValue(ratio-expectedRatio)<1e-12,"shared-hinge projection matches independent half-angle geometry");
        const Real errorMagnitude=1-ratio;require(errorMagnitude>0,"nonzero curved projection error is disclosed");
        if(previous>0)require(previous/errorMagnitude>3.9&&previous/errorMagnitude<4.1,"hinge flux error vanishes quadratically with curvature angle");
        std::cout<<std::setprecision(12)<<"curved angle scale="<<scale<<", relative projected-flux error="<<errorMagnitude<<", refinement ratio="<<(previous>0?previous/errorMagnitude:0)<<"\n";
        previous=errorMagnitude;
    }
}

void invalidHingeConormalRollsBack() {
    const auto p=curvedFilmPhysics();auto h=curvedFilmState(p);std::string error;
    for(std::size_t e=0;e<h.surface.edgeOwner.size();++e)if(h.surface.edgeNeighbour[e]>=0) {
        const int a=h.surface.edgeOwner[e],b=h.surface.edgeNeighbour[e];h.surface.edgeConormal[e]=normalized(h.surface.normal[a]+h.surface.normal[b]);
    }
    CpuFilmCandidate c;c.film.resize(1);c.film[0].mass=987;
    require(!advanceCpuFilmCandidate(h,p,.01,std::vector<CpuFilmForcing>(h.film.size()),curvedDrive(h),c,error),"normal-directed hinge edge rejected");
    require(c.film.size()==1&&c.film[0].mass==987,"invalid hinge does not mutate output");
}
}
int main(){stationaryCurvedWetSurface();curvedConductionConservesPhysicalEnergy();curvedShearTransportUsesSharedMetric();hingeProjectionHasSecondOrderAngularConsistency();invalidHingeConormalRollsBack();std::cout<<"CPU stationary curved-film integration passed\n";}
