// Host integration of actual CPU proposals, gas-wall packets, film update,
// material packet ownership and measured three-dimensional moving geometry.
// This never substitutes for the unexecuted native OF/CUDA interval driver.
#define main existing_geometry_fixture_main
#include "test_sweep_constraints.cpp"
#undef main
#include "ablation/CpuSurfaceInterface.H"
#include "film/CpuFilmDriver.H"
#include "materials/MaterialTransport.H"
#include "gpu/GasWallMath.H"
#include <iomanip>
namespace {
PhysicsConfig lifecyclePhysics() {
    PhysicsConfig p;p.enableFilm=true;p.spatialOrder=1;p.liquidViscosity=1;
    p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1;p.meltTemperature=500;
    p.meshMotion.policy=MeshMotionPolicy::CoupledRecession;p.liquidReferencePressure=100000;
    for(auto& s:p.species){s.R=287;s.cp0=1000;s.cp1=.2;s.e0=2000000;s.Tmin=100;s.Tmax=3000;}
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=2;c.conductivity=10;c.Tmin=100;c.Tmax=3000;}
    p.liquid=p.condensed[0];p.liquid.e0=200100;p.gasConductivity=10;
    return p;
}
HostState lifecycleState(const PhysicsConfig& p,Real thickness,Real temperature=500) {
    auto h=coupledPair();for(auto& x:h.gasMesh.points)x.z+=thickness;std::string error;
    require(rebuildGeometry(h.gasMesh,error),error);
    require(rebuildTrajectorySurface(h,h.gasMesh,h.solidMesh,1,h.surface,error),error);
    h.solid.resize(2);h.film.resize(2);
    for(int f=0;f<2;++f) {
        h.solid[f].condensed[0]=1000*h.solidMesh.volumes[f];h.solid[f].energy=h.solid[f].condensed[0]*condensedE(p.condensed[0],500);
        auto& q=h.film[f];q.mass=1000*thickness*h.surface.area[f];q.species[0]=q.mass;
        q.enthalpy=q.mass*liquidH(p,temperature,100000);
        require(recoverFilm(q,h.surface.area[f],100000,p,h.filmAux[f]),"lifecycle EOS");
    }
    return h;
}
struct LifecycleStep {
    HostState state;Real gasMass=0,gasEnergy=0,phaseMass=0;bool phaseEvent=false;
    Real filmResidual=0,gcl=0,sweepResidual=0,storage=0;
};
LifecycleStep advanceLifecycle(const HostState& h,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,Real dt,bool exactEvent=false) {
    std::string error;CpuSurfaceResult proposal;
    require(evaluateCpuSurface(h,p,bulk,{},{},{},dt,1,proposal,error),"CPU lifecycle surface: "+error);
    auto aux=h.filmAux;std::vector<Real> target(2,0),gasMass(2,0);
    for(const auto& packet:proposal.phasePackets)target[packet.filmFace]-=packet.mass/p.condensed[p.material.phaseCondensed].rho;
    for(int f=0;f<2;++f) {
        for(int s=0;s<Ns;++s)gasMass[f]+=dt*h.surface.area[f]*proposal.wall[f].speciesRate[s];
        Real phase=0;for(const auto& packet:proposal.phasePackets)if(packet.filmFace==f)phase+=packet.mass;
        Real endpointMass=h.film[f].mass+phase-gasMass[f];
        if(exactEvent&&absValue(endpointMass)<1e-12*h.film[f].mass)endpointMass=0;
        require(filmThicknessFromMass(endpointMass,h.surface.area[f],p.liquid.rho,aux[f].thickness),"lifecycle endpoint thickness");
        aux[f].solidFront+=target[f]/h.surface.area[f];aux[f].gasFront+=dt*proposal.physics[f].topSpeed;
    }
    LifecycleStep out;out.state=h;auto& next=out.state;SweepConstraintReport motion;
    require(moveCoupledMeshesConstrained(h,aux,target,dt,next.gasMesh,next.solidMesh,next.surface,motion,error),"lifecycle actual geometry: "+error);
    std::vector<Real> gs,ss;
    require(makeStageGeometry(h.gasMesh,next.gasMesh.points,dt,next.gasMesh,gs,error),error);
    require(makeStageGeometry(h.solidMesh,next.solidMesh.points,dt,next.solidMesh,ss,error),error);
    require(updateMaterialSweepRemainder(h,target,ss,next.solidMesh,next.solidSweepRemainder,error),"phase lifecycle geometric carry: "+error);
    require(validateMaterialSweepRemainder(next,error),"phase lifecycle bounded geometric carry: "+error);
    out.gcl=std::max(maximumGclResidual(h.gasMesh,next.gasMesh,gs),maximumGclResidual(h.solidMesh,next.solidMesh,ss));
    std::vector<CpuFilmForcing> force(2);std::vector<CpuFilmDrive> drive(2);std::vector<ExchangePacket> packets=proposal.phasePackets;
    for(const auto& packet:proposal.phasePackets) {
        auto& f=force[packet.filmFace];f.phaseMass+=packet.mass;f.phaseEnergy+=packet.energy;
        for(int s=0;s<Ns;++s)f.phaseSpecies[s]+=packet.species[s];
        f.phaseKineticOutflow+=packet.liquidKineticAdvection;f.bottomWorkIncluded=true;out.phaseMass+=packet.mass;
    }
    for(int f=0;f<2;++f) {
        const int gf=h.surface.gasFace[f],sf=h.surface.solidFace[f];const auto& wall=proposal.wall[f];
        GasWallInput input;input.bulk=bulk[f];input.temperature=wall.temperature;input.gasDistance=h.surface.gasDistance[f];
        input.area=h.surface.area[f];input.gasArea=mag(h.gasMesh.areaVectors[gf]);input.dt=dt;
        input.normal=-h.gasMesh.areaVectors[gf]/input.gasArea;input.normalSpeed=-gs[gf]/(input.gasArea*dt);
        input.sweptVolume=-gs[gf];input.velocity=wall.velocity;input.primaryKind=wall.primaryKind;
        for(int s=0;s<Ns;++s){input.speciesRate[s]=wall.speciesRate[s];input.poreRate[s]=wall.poreRate[s];input.poreSweepRate[s]=wall.poreSweepRate[s];}
        for(int c=0;c<Nc;++c)input.condensedRate[c]=wall.condensedRate[c];
        SurfacePacketIdentity id;id.step=1;id.stage=1;id.face=h.gasMesh.faceIds[gf];id.geometry=h.gasMesh.geometryVersion;id.gasCell=f;id.solidCell=f;id.filmFace=f;
        GasWallResult gas;require(evaluateGasWall(input,id,p,gas),"actual host gas-wall packet generation");
        const auto delta=packetDelta(gas.primary);auto& a=force[f];a.mass=delta.film.mass;a.energy=-gas.primary.energy;
        for(int s=0;s<Ns;++s)a.species[s]=delta.film.species[s];
        a.interfaceKineticOutflow=gas.primary.liquidKineticAdvection;a.topWorkIncluded=true;a.exactPhaseEvent=exactEvent;
        out.gasMass+=gas.primary.mass;out.gasEnergy+=gas.primary.energy;
        packets.push_back(gas.primary);packets.push_back(gas.pore);
        auto& d=drive[f];d.pressure=bulk[f].pressure;d.endpointPressure=d.pressure;d.hasCoupledNormalTrace=true;
        d.shear=gas.traction-input.normal*dot(gas.traction,input.normal);
        d.bottomNormalVelocity=proposal.physics[f].liquidBottomNormal;
        d.topNormalVelocity=input.normalSpeed+gas.primary.mass/(dt*input.gasArea*p.liquid.rho);
        d.solidNormalVelocity=target[f]/(dt*h.surface.area[f]);d.interfaceNormalVelocity=input.normalSpeed;
        require(filmSweepsCompatible(h.film[f].mass,h.film[f].mass+a.mass+a.phaseMass,p.liquid.rho,gs[gf],ss[sf],0,p.tolerances),"lifecycle actual film-volume certificate");
        out.sweepResidual=std::max(out.sweepResidual,absValue(-gs[gf]-ss[sf]-(a.mass+a.phaseMass)/p.liquid.rho));
    }
    for(auto& packet:packets)if(packet.kind==ExchangeKind::SolidFilm) {
        const int f=packet.filmFace;bool finalized=false;const Vec3 bottom=h.surface.normal[f]*drive[f].bottomNormalVelocity;
        require(finalizeCpuFilmPhasePacket(h.film[f],h.filmAux[f].pressure,FilmQ{},force[f],dt,bottom,p,packet,finalized,error),"terminal phase completion: "+error);
        force[f].phaseMass=packet.mass;force[f].phaseEnergy=packet.energy;force[f].phaseKineticOutflow=packet.liquidKineticAdvection;
        for(int s=0;s<Ns;++s)force[f].phaseSpecies[s]=packet.species[s];
        force[f].exactPhaseEvent=force[f].exactPhaseEvent||finalized;
    }
    MaterialTransportResult transport;require(evaluateMaterialTransport(h.solidMesh,h.solid,p,ss,dt,.4,transport,error),"lifecycle material ALE: "+error);
    for(int c=0;c<2;++c)next.solid[c]=addSolid(h.solid[c],transport.rates[c],dt);
    Real numerical=0;std::vector<SolidQ> applied;require(applyMaterialPacketBatch(next.solid,packets,applied,numerical,error),"actual material packet batch: "+error);next.solid=applied;
    HostState filmState=h;filmState.surface=next.surface;CpuFilmCandidate film;
    Real admitted=0;
    require(estimateCpuFilmStep(filmState,p,force,drive,dt,dt,admitted,error),"production lifecycle film step bound: "+error);
    require(admitted>=dt*(1-32*std::numeric_limits<Real>::epsilon()),"lifecycle interval admitted by production film controller");
    const bool filmOk=advanceCpuFilmCandidate(filmState,p,dt,force,drive,film,error);
    require(filmOk,"CPU lifecycle film: "+error);
    out.phaseEvent=film.report.phaseEvent;out.filmResidual=film.report.reducedEnergyResidual;out.storage=film.report.budgetDelta.filmPressureVolume;
    next.film=film.film;next.filmAux=film.filmAux;next.time=h.time+dt;
    for(int f=0;f<2;++f){next.filmAux[f].solidFront=aux[f].solidFront;next.filmAux[f].gasFront=aux[f].gasFront;MaterialPrimitive m;
        require(recoverMaterial(next.solid[f],next.solidMesh.volumes[f],p,m),"lifecycle endpoint material");next.solid[f].porosity=m.porosity;}
    return out;
}
GasPrimitive primitive(const PhysicsConfig& p,Real temperature) {
    GasPrimitive g;g.temperature=temperature;g.pressure=100000;g.rho=g.pressure/(p.species[0].R*temperature);g.Y[0]=1;return g;
}
Real totalMass(const HostState& h){Real value=0;for(const auto& q:h.solid)for(Real m:q.condensed)value+=m;for(const auto& q:h.film)value+=q.mass;return value;}
Real totalEnergy(const HostState& h,const PhysicsConfig& p){Real value=0;for(const auto& q:h.solid)value+=q.energy;for(std::size_t f=0;f<h.film.size();++f)value+=filmInternalEnergy(h.film[f],h.filmAux[f].pressure,p);return value;}
void malformedSurfaceMapsAreRejectedBeforeIndexing() {
    const auto p=lifecyclePhysics();const auto original=lifecycleState(p,.02);
    const std::vector<GasPrimitive> bulk(2,primitive(p,500));
    for(int field=0;field<2;++field) {
        auto h=original;if(field==0)h.surface.solidCell.resize(1);else h.surface.baseVelocity.resize(1);
        CpuSurfaceResult sentinel;sentinel.wall.resize(1);sentinel.wall[0].temperature=987;std::string error;
        require(!evaluateCpuSurface(h,p,bulk,{},{},{},.1,1,sentinel,error),"truncated optional surface owner/velocity map rejected");
        require(sentinel.wall.size()==1&&sentinel.wall[0].temperature==987,"malformed map preserves candidate output");
        require(error.find("layout")!=std::string::npos,"malformed map has a layout diagnostic");
    }
}

void terminalRemainderSignsPressureAndRollback() {
    for(Real referencePressure:std::vector<Real>{0,20000000})for(Real remainder:std::vector<Real>{-7,7}) {
        auto p=lifecyclePhysics();p.liquidReferencePressure=referencePressure;
        FilmQ q;q.mass=1;q.species[0]=1;const Real pressure=123456;q.enthalpy=liquidH(p,500,pressure);
        FilmQ local;local.enthalpy=5;CpuFilmForcing primary;primary.mass=-1;primary.species[0]=-1;
        primary.interfaceKineticOutflow=2;primary.radiationEnergy=3;
        primary.energy=-filmInternalEnergy(q,pressure,p)+remainder-5-2-3;
        ExchangePacket phase;phase.kind=ExchangeKind::SolidFilm;phase.solidCell=0;phase.filmFace=0;
        phase.conductive=17;phase.pressureWork=-9;phase.viscousWork=2;phase.energy=10;
        bool event=false;std::string error;
        require(finalizeCpuFilmPhasePacket(q,pressure,local,primary,1,{},p,phase,event,error),"signed terminal remainder: "+error);
        require(event&&phase.mass==0,"gas-only exact terminal event has zero phase mass");
        require(absValue(phase.energy+remainder)<1e-8,"terminal residual energy transferred to correct solid/film side");
        require(absValue(phase.energy-phase.advective-phase.conductive-phase.pressureWork-phase.viscousWork)<1e-12,"terminal packet energy decomposition");
        // Even a tiny positive remaining mass is not a terminal event.
        primary.mass=-1+1e-10;primary.species[0]=primary.mass;const auto previous=phase;event=true;
        require(finalizeCpuFilmPhasePacket(q,pressure,local,primary,1,{},p,phase,event,error),"near-empty phase remains admissible");
        require(!event&&phase.mass==previous.mass&&phase.energy==previous.energy,"no tolerance-based premature dryout");
    }
    const auto p=lifecyclePhysics();FilmQ q;q.mass=1;q.species[0]=1;q.enthalpy=liquidH(p,500,100000);
    CpuFilmForcing primary;primary.mass=-.25;primary.species[0]=-.25;primary.energy=-123;
    ExchangePacket phase;phase.kind=ExchangeKind::SolidFilm;phase.solidCell=0;phase.filmFace=0;
    phase.mass=-2;phase.condensed[0]=-2;phase.species[0]=-2;phase.energy=-456;phase.advective=-456;
    bool event=false;std::string error;
    require(finalizeCpuFilmPhasePacket(q,100000,FilmQ{},primary,1,{2,0,0},p,phase,event,error),"freezing remainder retained: "+error);
    require(event&&phase.mass==-.75&&phase.species[0]==-.75&&phase.condensed[0]==-.75,"freezing transfers the exact available remainder");
    require(phase.liquidKineticAdvection==1.5,"freezing phase kinetic advection remains separate");
    require(absValue(phase.energy+filmInternalEnergy(q,100000,p)-123+1.5)<1e-8,"freezing physical energy plus kinetic channel closes");
    primary.species[0]=-.3;primary.species[1]=.05;const auto previous=phase;event=false;
    require(!finalizeCpuFilmPhasePacket(q,100000,FilmQ{},primary,1,{2,0,0},p,phase,event,error),"incompatible terminal phase composition rejected");
    require(phase.mass==previous.mass&&phase.energy==previous.energy&&!event,"failed terminal completion is transactional");
}

void nonlinearFilmCaloricAndPressure() {
    auto p=lifecyclePhysics();auto h=lifecycleState(p,.02,400);h.surface.gasFace.assign(2,-1);
    std::vector<CpuFilmDrive> drive(2);std::vector<CpuFilmForcing> force(2);
    const Real expectedT=700;
    for(int f=0;f<2;++f){drive[f].pressure=250000;drive[f].endpointPressure=600000;
        force[f].energy=h.film[f].mass*(1000*(expectedT-400)+(expectedT*expectedT-400*400));force[f].topWorkIncluded=force[f].bottomWorkIncluded=true;}
    CpuFilmCandidate c;std::string error;
    require(advanceCpuFilmCandidate(h,p,1,force,drive,c,error),"nonlinear caloric film candidate: "+error);
    for(const auto& a:c.filmAux)require(absValue(a.temperature-expectedT)<1e-9,"quadratic caloric heating plus independent pressure storage");
    require(absValue(c.report.budgetDelta.filmPressureVolume-20000)<1e-8,"variable pressure pV storage exactly once");
    require(absValue(c.report.reducedEnergyResidual)<1e-7,"nonlinear film physical energy residual");
}
void dryBirthFromGasPackets() {
    auto p=lifecyclePhysics();p.material.enableMelting=true;const auto h=lifecycleState(p,0);
    const auto gas=primitive(p,600);const Real dt=.1;
    auto next=advanceLifecycle(h,p,std::vector<GasPrimitive>(2,gas),dt);
    const Real latent=liquidH(p,500,100000)-condensedE(p.condensed[0],500)-100000/p.liquid.rho;
    const Real expectedPerFace=dt*(p.gasConductivity/h.surface.gasDistance[0])*(600-500)/latent;
    require(next.phaseEvent,"actual dry-to-wet birth event reported");
    for(int f=0;f<2;++f){require(absValue(next.state.film[f].mass-expectedPerFace)<1e-12,"birth mass follows independently computed latent balance");
        require(absValue(next.state.filmAux[f].temperature-500)<1e-8,"newborn film recovers melting temperature with nonlinear cp");}
    require(absValue(totalMass(next.state)-totalMass(h))<1e-10,"birth combined mass conservation");
    require(absValue(totalEnergy(next.state,p)-totalEnergy(h,p)+next.gasEnergy)<1e-6,"birth energy closes against immutable gas-generated packet");
    require(next.gcl<1e-12&&next.sweepResidual<1e-12,"birth actual 3-D geometry certificate");
}
void evaporatingFilmUsesActualGasPackets(bool terminal) {
    auto p=lifecyclePhysics();p.material.evaporation=EvaporationLaw::HertzKnudsen;
    p.material.evaporationAccommodation[0]=1;const Real temperature=500,targetJ=.01;
    p.material.saturationPressureScale[0]=100000+targetJ*std::sqrt(2*3.14159265358979323846*p.species[0].R*temperature);
    auto h=lifecycleState(p,.02,temperature);const auto base=h;const int steps=terminal?1:20;Real integratedGasMass=0,integratedGasEnergy=0;
    Real actualFlux[Ns]{};Real composition[Ns]{};composition[0]=1;
    require(hertzKnudsenFlux(h.film[0],composition,100000,temperature,p,actualFlux),"evaporation kinetic reference");
    // Independent kinetic oracle for the rounded, supplied saturation table.
    const Real j=(p.material.saturationPressureScale[0]-100000)
        /std::sqrt(2*3.14159265358979323846*p.species[0].R*temperature);
    require(absValue(actualFlux[0]-j)<1e-14&&absValue(j-targetJ)<1e-12,"Hertz-Knudsen flux matches the independently manufactured kinetic rate");
    Real dt=terminal?h.film[0].mass/(h.surface.area[0]*j):50;
    Real maximumTemperatureError=0,maximumMassError=0,maximumEnergyError=0;
    for(int step=0;step<steps;++step) {
        std::vector<GasPrimitive> bulk(2);
        for(int f=0;f<2;++f) {
            const Real gasDensity=100000/(p.species[0].R*temperature);
            const Real gasNormal=j*(1/gasDensity-1/p.liquid.rho);
            const Real latent=speciesH(p.species[0],temperature)-liquidH(p,temperature,100000)+.5*gasNormal*gasNormal;
            const Real conductance=p.gasConductivity/h.surface.gasDistance[f];
            bulk[f]=primitive(p,temperature+j*latent/conductance);
        }
        if(terminal)for(int align=0;align<8;++align) {
            CpuSurfaceResult proposed;std::string error;require(evaluateCpuSurface(h,p,bulk,{},{},{},dt,1,proposed,error),"terminal event rate: "+error);
            const Real rate=proposed.wall[0].speciesRate[0]*h.surface.area[0];
            const Real remainder=h.film[0].mass-dt*rate;
            if(absValue(remainder)<=8*std::numeric_limits<Real>::epsilon()*h.film[0].mass)break;
            dt=h.film[0].mass/rate;
        }
        const auto next=advanceLifecycle(h,p,bulk,dt,terminal);integratedGasMass+=next.gasMass;integratedGasEnergy+=next.gasEnergy;h=next.state;
        for(int f=0;f<2;++f) {
            const Real expectedMass=base.film[f].mass-j*h.surface.area[f]*(step+1)*dt;
            maximumMassError=std::max(maximumMassError,absValue(h.film[f].mass-expectedMass));
            if(!terminal)maximumTemperatureError=std::max(maximumTemperatureError,absValue(h.filmAux[f].temperature-temperature));
        }
        maximumEnergyError=std::max(maximumEnergyError,absValue(totalEnergy(h,p)-totalEnergy(base,p)+integratedGasEnergy));
        require(next.gcl<1e-12&&next.sweepResidual<1e-12,"evaporation actual geometry/sweep certificate");
        if(terminal)require(next.phaseEvent&&h.film[0].mass==0&&h.film[0].enthalpy==0,"gas-generated exact terminal dryout");
    }
    require(maximumTemperatureError<1e-7,"manufactured isothermal evaporation temperature");
    require(maximumMassError<1e-8,"evaporating film follows linear analytic inventory law");
    require(absValue(totalMass(h)-totalMass(base)+integratedGasMass)<1e-9,"evaporation combined mass plus gas packet budget");
    require(maximumEnergyError<1e-4,"evaporation physical energy plus gas packet budget");
    if(terminal) {
        CpuSurfaceResult nextInterval;std::string error;
        require(evaluateCpuSurface(h,p,std::vector<GasPrimitive>(2,primitive(p,600)),{},{},{},.01,2,nextInterval,error),"dry endpoint next interface: "+error);
        require(nextInterval.phasePackets.empty(),"dry endpoint has no spurious internal film packet");
        for(const auto& wall:nextInterval.wall)require(wall.primaryKind==ExchangeKind::GasSolid,"next interval switches to the actual dry owner");
    }
    std::cout<<std::setprecision(10)<<(terminal?"terminal dryout":"evaporation transient")<<": max mass error="<<maximumMassError<<" kg, temperature="<<maximumTemperatureError<<" K, energy="<<maximumEnergyError<<" J\n";
}
}
int main(int argc,char** argv){const std::string mode=argc>1?argv[1]:"all";
    if(mode=="all"||mode=="layout")malformedSurfaceMapsAreRejectedBeforeIndexing();
    if(mode=="all"||mode=="remainder")terminalRemainderSignsPressureAndRollback();
    if(mode=="all"||mode=="caloric")nonlinearFilmCaloricAndPressure();
    if(mode=="all"||mode=="birth")dryBirthFromGasPackets();
    if(mode=="all"||mode=="evaporation")evaporatingFilmUsesActualGasPackets(false);
    if(mode=="all"||mode=="dryout")evaporatingFilmUsesActualGasPackets(true);
    std::cout<<"CPU gas-packet/film phase lifecycle integration passed\n";
}
