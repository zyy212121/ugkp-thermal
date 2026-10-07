#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include "gas/AleFlux.H"
#include <algorithm>
#include <utility>

namespace chmt { namespace {
struct Evaluation {
    std::vector<FilmQ> q,rhs,edgeFlux,withdrawal;
    std::vector<FilmAux> aux;
    std::vector<Vec3> shear,gradient;
    std::vector<FilmRateBudget> budget;
    std::vector<Real> outgoingLength,conductance,sideMeshVolumeRate;
};
bool reject(std::string& error,const char* message,std::size_t index) {
    error=std::string("CPU film ")+message+" at "+std::to_string(index);return false;
}
template<class T> bool optionalSize(const std::vector<T>& value,std::size_t n) {
    return value.empty()||value.size()==n;
}
bool coupledFace(const SurfaceMesh& mesh,std::size_t f) {
    return !mesh.gasFace.empty()&&mesh.gasFace[f]>=0;
}
Real endpointPressure(const CpuFilmDrive& d) {
    // Only NaN is the documented omitted value; infinity is not omitted.
    return d.endpointPressure==d.endpointPressure?d.endpointPressure:d.pressure;
}
bool validate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    std::string& error) {
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    if(!finite(dt)||dt<=0||h.filmAux.size()!=n||m.area.size()!=n||m.normal.size()!=n
        ||forcing.size()!=n||drive.size()!=n) return reject(error,"invalid dimensions or interval",n);
    if(n==0) {
        if(ne) return reject(error,"edges without film faces",ne);
        return true;
    }
    if(!p.enableFilm||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged)
        return reject(error,"unsupported disabled or resolved-normal model",0);
    if(!validCondensed(p.liquid)||!finite(p.liquid.conductivity)||p.liquid.conductivity<0
        ||!finite(p.liquidViscosity)||p.liquidViscosity<=0||!finite(p.gravity)
        ||(p.spatialOrder!=1&&p.spatialOrder!=2)) return reject(error,"invalid constitutive configuration",0);
    if(p.enableRadiation&&(!finite(p.emissivity)||p.emissivity<0||p.emissivity>1
        ||!finite(p.ambientTemperature)||p.ambientTemperature<=0))
        return reject(error,"invalid radiation configuration",0);
    if(!optionalSize(m.gasFace,n)||!optionalSize(m.baseVelocity,n)
        ||!optionalSize(m.prescribedTopTraction,n)||!optionalSize(m.prescribedPressureGradient,n)
        ||m.edgeNeighbour.size()!=ne||m.edgeLength.size()!=ne||m.edgeConormal.size()!=ne
        ||m.edgeOwnerOffset.size()!=ne||m.edgeNeighbourOffset.size()!=ne
        ||!optionalSize(m.sweptEdgeArea,ne)) return reject(error,"invalid surface connectivity dimensions",n);
    for(std::size_t f=0;f<n;++f) {
        const auto& a=forcing[f];const auto& d=drive[f];
        if(!finite(m.area[f])||m.area[f]<=0||!finite(m.normal[f])
            ||!closeEnough(mag(m.normal[f]),1,1e-12,1e-12)
            ||!finite(d.pressure)||!finite(endpointPressure(d))||!finite(d.shear)
            ||!finite(h.filmAux[f].pressure)||!finite(h.filmAux[f].kineticEnergy)
            ||h.filmAux[f].kineticEnergy<0) return reject(error,"invalid geometry or pressure/traction trace",f);
        if(!finite(a.mass)||!finite(a.energy)||!finite(a.phaseMass)||!finite(a.phaseEnergy)
            ||!finite(a.phaseKineticOutflow)||!finite(a.interfaceKineticOutflow)
            ||!finite(a.radiationEnergy)||!finite(a.withdrawnMass)||a.withdrawnMass<0)
            return reject(error,"invalid integrated forcing",f);
        Real mass=0,phaseMass=0,gross=0;
        for(int s=0;s<Ns;++s) {
            if(!finite(a.species[s])||!finite(a.phaseSpecies[s])||!finite(a.withdrawnSpecies[s])||a.withdrawnSpecies[s]<0)
                return reject(error,"invalid integrated species forcing",f);
            mass+=a.species[s];phaseMass+=a.phaseSpecies[s];gross+=a.withdrawnSpecies[s];
        }
        if(!closeEnough(mass,a.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
            return reject(error,"forcing species/mass mismatch",f);
        if(!closeEnough(phaseMass,a.phaseMass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
            return reject(error,"phase forcing species/mass mismatch",f);
        if(!closeEnough(gross,a.withdrawnMass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
            return reject(error,"gross withdrawal species/mass mismatch",f);
        if(coupledFace(m,f)) {
            if((!m.prescribedTopTraction.empty()&&mag(m.prescribedTopTraction[f])!=0)
                ||(!m.prescribedPressureGradient.empty()&&mag(m.prescribedPressureGradient[f])!=0))
                return reject(error,"coupled face has prescribed standalone drive",f);
            if((h.film[f].mass>0||a.mass>0||a.phaseMass>0)&&!d.hasCoupledNormalTrace)
                return reject(error,"missing coupled normal trace",f);
            if(!finite(d.bottomNormalVelocity)||!finite(d.topNormalVelocity)
                ||!finite(d.interfaceNormalVelocity)||!finite(d.solidNormalVelocity))
                return reject(error,"invalid coupled normal trace",f);
        }
    }
    for(std::size_t e=0;e<ne;++e) {
        const int owner=m.edgeOwner[e],neighbour=m.edgeNeighbour[e];
        if(owner<0||static_cast<std::size_t>(owner)>=n||neighbour< -1
            ||(neighbour>=0&&static_cast<std::size_t>(neighbour)>=n)||owner==neighbour
            ||!finite(m.edgeLength[e])||m.edgeLength[e]<=0||!finite(m.edgeConormal[e])
            ||!closeEnough(mag(m.edgeConormal[e]),1,1e-12,1e-12)
            ||!finite(m.edgeOwnerOffset[e])||!finite(m.edgeNeighbourOffset[e])
            ||(!m.sweptEdgeArea.empty()&&!finite(m.sweptEdgeArea[e])))
            return reject(error,"invalid surface edge",e);
        // Geometry owns one shared hinge metric. On a curved surface its
        // conormal is tangent to the bisector plane, not either face alone.
        // Open edges retain their owner-face metric.
        const Vec3 hingeNormal=neighbour>=0?normalized(m.normal[owner]+m.normal[neighbour]):m.normal[owner];
        if(mag(hingeNormal)==0||!closeEnough(dot(m.edgeConormal[e],hingeNormal),0,1e-10,0))
            return reject(error,"edge conormal is not tangential",e);
        if(neighbour>=0&&dot(m.edgeOwnerOffset[e]-m.edgeNeighbourOffset[e],m.edgeConormal[e])<=0)
            return reject(error,"nonpositive surface edge distance",e);
    }
    return true;
}
Real surfaceValue(std::size_t f,int component,const Evaluation& e,const SurfaceMesh& m) {
    if(component<0)return e.aux[f].pressure;
    if(component==0)return e.q[f].mass/m.area[f];
    if(component==1)return e.q[f].enthalpy/m.area[f];
    return e.q[f].species[component-2]/m.area[f];
}
Vec3 surfaceGradient(std::size_t f,int component,const Evaluation& e,const SurfaceMesh& m) {
    Symmetric3 matrix;Vec3 rhs{};const Real value=surfaceValue(f,component,e,m);
    for(std::size_t edge=0;edge<m.edgeOwner.size();++edge) {
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        if(neighbour<0||(static_cast<std::size_t>(owner)!=f&&static_cast<std::size_t>(neighbour)!=f))continue;
        const bool isOwner=static_cast<std::size_t>(owner)==f;
        const int other=isOwner?neighbour:owner;
        Vec3 dx=m.edgeOwnerOffset[edge]-m.edgeNeighbourOffset[edge];if(!isOwner)dx=-dx;
        addLeastSquares(matrix,rhs,dx,surfaceValue(other,component,e,m)-value);
    }
    return solveLeastSquares(matrix,rhs);
}
FilmProfile faceProfile(std::size_t f,const Evaluation& e,const SurfaceMesh& m,const PhysicsConfig& p) {
    const auto& a=e.aux[f];const Vec3 normal=m.normal[f];
    const Vec3 gravity=p.gravity-normal*dot(p.gravity,normal);
    FilmProfile profile=filmProfile(a.thickness,p.liquidViscosity,a.baseVelocity,
        e.shear[f],e.gradient[f]-gravity*p.liquid.rho);
    return withNormalProfile(profile,normal,2*dot(a.meanVelocity,normal)-dot(a.topVelocity,normal),
        dot(a.topVelocity,normal));
}
bool reconstruct(std::size_t f,Vec3 offset,const Evaluation& e,const SurfaceMesh& m,
    const PhysicsConfig& p,FilmQ& result) {
    FilmQ base; base.mass=surfaceValue(f,0,e,m);base.enthalpy=surfaceValue(f,1,e,m);
    for(int s=0;s<Ns;++s)base.species[s]=surfaceValue(f,s+2,e,m);
    if(p.spatialOrder==1||base.mass==0){result=base;return true;}
    FilmQ increment;increment.mass=dot(surfaceGradient(f,0,e,m),offset);
    increment.enthalpy=dot(surfaceGradient(f,1,e,m),offset);
    for(int s=0;s<Ns;++s)increment.species[s]=dot(surfaceGradient(f,s+2,e,m),offset);
    Real scale=1;
    if(p.reconstruction==ReconstructionMode::LimitedLinear) {
        for(int component=0;component<Ns+2;++component) {
            const Real value=surfaceValue(f,component,e,m);Real lo=value,hi=value;
            for(std::size_t edge=0;edge<m.edgeOwner.size();++edge) {
                const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
                if(neighbour<0||(static_cast<std::size_t>(owner)!=f&&static_cast<std::size_t>(neighbour)!=f))continue;
                const int other=static_cast<std::size_t>(owner)==f?neighbour:owner;
                const Real v=surfaceValue(other,component,e,m);lo=minValue(lo,v);hi=maxValue(hi,v);
            }
            const Real change=component==0?increment.mass:(component==1?increment.enthalpy:increment.species[component-2]);
            scale=minValue(scale,barthJespersen(value,lo,hi,change));
        }
    }
    for(int trial=0;trial<48;++trial) {
        FilmQ q;q.mass=base.mass+scale*increment.mass;q.enthalpy=base.enthalpy+scale*increment.enthalpy;
        for(int s=0;s<Ns;++s)q.species[s]=base.species[s]+scale*increment.species[s];
        FilmAux aux;
        if(recoverFilm(q,1,e.aux[f].pressure,p,aux)){result=q;return true;}
        scale*=.5;
    }
    return false;
}
bool recoverAndProfile(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmDrive>& drive,Evaluation& e,std::string& error) {
    const auto& m=h.surface;const std::size_t n=h.film.size();
    e.q=h.film;e.aux=h.filmAux;e.shear.resize(n);e.gradient.resize(n);
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];const auto& d=drive[f];const bool coupled=coupledFace(m,f);
        a.baseVelocity=m.baseVelocity.empty()?Vec3{}:m.baseVelocity[f];a.normal=m.normal[f];
        FilmAux check=a;
        if(!recoverFilm(h.film[f],m.area[f],h.filmAux[f].pressure,p,check))
            return reject(error,"inadmissible accepted thermal inventory",f);
        // Evaluation pressure changes storage, never the physical base U or T.
        e.q[f].enthalpy+=pressureVolumeProduct(h.filmAux[f].pressure,h.film[f].mass/p.liquid.rho,
            d.pressure,h.film[f].mass/p.liquid.rho);
        if(coupled) {
            if(!setCoupledFilmNormalTrace(a,m.normal[f],h.film[f].mass>0,d.bottomNormalVelocity,d.topNormalVelocity))
                return reject(error,"invalid coupled profile trace",f);
            a.normalVelocity=d.interfaceNormalVelocity;a.solidNormalVelocity=d.solidNormalVelocity;
        }
        if(!recoverFilmThermalState(e.q[f],m.area[f],d.pressure,p,m.normal[f],coupled,a))
            return reject(error,"thermal recovery failed",f);
        const Vec3 supplied=coupled?d.shear:(m.prescribedTopTraction.empty()?Vec3{}:m.prescribedTopTraction[f]);
        e.shear[f]=supplied-m.normal[f]*dot(supplied,m.normal[f]);
    }
    for(std::size_t f=0;f<n;++f) {
        const Vec3 normal=m.normal[f];Vec3 gradient=surfaceGradient(f,-1,e,m);
        if(!m.prescribedPressureGradient.empty())gradient+=m.prescribedPressureGradient[f];
        e.gradient[f]=gradient-normal*dot(gradient,normal);
        const Vec3 G=e.gradient[f]-(p.gravity-normal*dot(p.gravity,normal))*p.liquid.rho;
        if(!updateFilmProfileState(e.aux[f],p,normal,e.shear[f],G,coupledFace(m,f)))
            return reject(error,"profile update failed",f);
    }
    return true;
}
void addRate(FilmQ& q,const FilmQ& flux,Real sign) {
    q.mass+=sign*flux.mass;q.enthalpy+=sign*flux.enthalpy;
    for(int s=0;s<Ns;++s)q.species[s]+=sign*flux.species[s];
}
bool evaluate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Evaluation& e,std::string& error) {
    if(!validate(h,p,dt,forcing,drive,error)||!recoverAndProfile(h,p,drive,e,error))return false;
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    e.rhs.resize(n);e.withdrawal.resize(n);e.budget.resize(n);e.edgeFlux.resize(ne);
    e.outgoingLength.assign(n,0);e.conductance.assign(n,0);e.sideMeshVolumeRate.assign(n,0);
    auto& sideVolumeRate=e.sideMeshVolumeRate;
    for(std::size_t edge=0;edge<ne;++edge) {
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        const Vec3 conormal=m.edgeConormal[edge];const Real length=m.edgeLength[edge];
        const Real meshRate=m.sweptEdgeArea.empty()?0:m.sweptEdgeArea[edge]/dt;
        const Vec3 velocity=neighbour>=0?(e.aux[owner].meanVelocity+e.aux[neighbour].meanVelocity)*.5:e.aux[owner].meanVelocity;
        const Real lengthRate=length*dot(velocity,conormal)-meshRate;
        if(!finite(lengthRate)||(neighbour<0&&lengthRate<0))return reject(error,"undeclared open-edge inflow",edge);
        const int donor=lengthRate>=0||neighbour<0?owner:neighbour;
        FilmQ reconstructed;
        if(!reconstruct(donor,donor==owner?m.edgeOwnerOffset[edge]:m.edgeNeighbourOffset[edge],e,m,p,reconstructed))
            return reject(error,"edge reconstruction failed",edge);
        FilmQ flux=filmAdvectiveFlux(reconstructed,lengthRate);
        flux.enthalpy+=e.aux[donor].pressure*reconstructed.mass/p.liquid.rho*meshRate;
        e.outgoingLength[donor]+=absValue(lengthRate);
        e.withdrawal[donor].mass+=absValue(flux.mass);
        for(int s=0;s<Ns;++s)e.withdrawal[donor].species[s]+=absValue(flux.species[s]);
        if(neighbour>=0&&h.film[owner].mass>0&&h.film[neighbour].mass>0) {
            const Real distance=dot(m.edgeOwnerOffset[edge]-m.edgeNeighbourOffset[edge],conormal);
            const Real conductance=p.liquid.conductivity*length*.5*(e.aux[owner].thickness+e.aux[neighbour].thickness)/distance;
            flux.enthalpy+=conductance*(e.aux[owner].temperature-e.aux[neighbour].temperature);
            e.conductance[owner]+=conductance;e.conductance[neighbour]+=conductance;
        }
        if(!finite(flux.mass)||!finite(flux.enthalpy))return reject(error,"nonfinite edge flux",edge);
        e.edgeFlux[edge]=flux;addRate(e.rhs[owner],flux,-1);
        e.budget[owner].edgeEnergyOutflow+=flux.enthalpy;
        const FilmProfile profile=faceProfile(donor,e,m,p);
        const Vec3 grid=conormal*(meshRate/length);
        const Real kinetic=p.liquid.rho*length*profileKineticFlux(profile,grid,conormal);
        e.budget[owner].edgeKineticOutflow+=kinetic;sideVolumeRate[owner]+=profile.delta*meshRate;
        if(neighbour>=0) {
            addRate(e.rhs[neighbour],flux,1);e.budget[neighbour].edgeEnergyOutflow-=flux.enthalpy;
            e.budget[neighbour].edgeKineticOutflow-=kinetic;sideVolumeRate[neighbour]-=profile.delta*meshRate;
        }
    }
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];auto& b=e.budget[f];const auto& source=forcing[f];
        const bool coupled=coupledFace(m,f);const FilmProfile profile=faceProfile(f,e,m,p);
        if(!coupled) {
            a.solidNormalVelocity=dot(a.baseVelocity,m.normal[f]);
            a.normalVelocity=a.solidNormalVelocity+((e.rhs[f].mass+(source.mass+source.phaseMass)/dt)/p.liquid.rho-sideVolumeRate[f])/a.area;
        }
        b.bodyPower=h.film[f].mass*dot(p.gravity,a.meanVelocity);
        if(!m.prescribedPressureGradient.empty())b.bodyPower-=h.film[f].mass/p.liquid.rho*dot(m.prescribedPressureGradient[f],a.meanVelocity);
        if(!source.topWorkIncluded)b.prescribedTopPower=a.area*(dot(e.shear[f],a.topVelocity)-a.pressure*a.normalVelocity);
        if(!source.bottomWorkIncluded)b.supportPower=a.area*(-dot(profile.bottomShear,a.baseVelocity)+a.pressure*a.solidNormalVelocity);
        const Real localRadiation=!coupled&&h.film[f].mass>0?a.area*grayRadiation(a.temperature,p):0;
        b.radiationPower=source.radiationEnergy/dt+localRadiation;
        b.dissipationPower=a.area*profile.dissipation;
        b.interfaceKineticOutflow=(source.interfaceKineticOutflow+source.phaseKineticOutflow)/dt;
        e.rhs[f].enthalpy+=b.bodyPower+b.supportPower+b.prescribedTopPower+localRadiation;
        if(!finite(e.rhs[f].enthalpy)||!finite(a.normalVelocity)||!finite(b.dissipationPower))
            return reject(error,"nonfinite surface source",f);
    }
    return true;
}
} // namespace

bool finalizeCpuFilmPhasePacket(const FilmQ& accepted,Real acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,Vec3 bottomVelocity,
    const PhysicsConfig& p,ExchangePacket& phase,bool& exactEvent,std::string& error) {
    if(!finite(dt)||dt<=0||!finite(acceptedPressure)||!finite(bottomVelocity)
        ||!finite(p.liquid.rho)||p.liquid.rho<=0||!validFilmInventory(accepted)
        ||phase.kind!=ExchangeKind::SolidFilm||!finite(phase.mass)) {
        error="invalid terminal film phase input";return false;
    }
    const Real available=(accepted.mass+dt*localRate.mass)+primary.mass;
    if(!finite(available)){error="nonfinite terminal film mass";return false;}
    const bool freezing=phase.mass<0&&available>=0&&-phase.mass>=available;
    const bool gasOrTransportEmpty=accepted.mass>0&&available==0&&phase.mass==0
        &&(primary.mass<0||localRate.mass<0);
    if(!freezing&&!gasOrTransportEmpty){exactEvent=false;error.clear();return true;}
    ExchangePacket candidate=phase;candidate.mass=-available;
    if(candidate.mass!=0&&(p.material.phaseCondensed<0||p.material.phaseCondensed>=Nc)) {
        error="terminal film phase has no condensed owner species";return false;
    }
    for(int c=0;c<Nc;++c)candidate.condensed[c]=c==p.material.phaseCondensed?candidate.mass:0;
    for(int s=0;s<Ns;++s) {
        const Real species=(accepted.species[s]+dt*localRate.species[s])+primary.species[s];
        if(!closeEnough(species,available*p.material.phaseFilmY[s],p.tolerances.absoluteMass,p.tolerances.relativeMass)) {
            error="terminal film phase composition mismatch";return false;
        }
        candidate.species[s]=-species;
    }
    candidate.liquidKineticAdvection=-.5*candidate.mass*dot(bottomVelocity,bottomVelocity);
    Real remaining=filmInternalEnergy(accepted,acceptedPressure,p);
    remaining+=dt*localRate.enthalpy;remaining+=primary.energy;
    remaining+=primary.interfaceKineticOutflow;remaining+=primary.radiationEnergy;
    remaining+=candidate.liquidKineticAdvection;
    candidate.energy=-remaining;
    candidate.advective=candidate.energy-candidate.conductive-candidate.pressureWork-candidate.viscousWork;
    if(!validatePacketMath(candidate,p.tolerances)){error="invalid terminal film phase packet";return false;}
    phase=candidate;exactEvent=true;error.clear();return true;
}

bool advanceCpuFilmCandidate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    CpuFilmCandidate& output,std::string& error) {
    Evaluation e;if(!evaluate(h,p,dt,forcing,drive,e,error))return false;
    CpuFilmCandidate next;next.film=h.film;next.filmAux=e.aux;
    auto& report=next.report;report.edgeFlux=e.edgeFlux;report.rateBudget=e.budget;
    auto& budget=report.budgetDelta;Real totalInput=0,totalKinetic=0,deltaK=0;
    for(std::size_t f=0;f<h.film.size();++f) {
        const auto& old=h.film[f];const auto& source=forcing[f];const auto& b=e.budget[f];
        const Real withdrawn=dt*e.withdrawal[f].mass+maxValue(source.withdrawnMass,maxValue(0,-source.mass)+maxValue(0,-source.phaseMass));
        if(!finite(withdrawn)||withdrawn-old.mass>16*std::numeric_limits<Real>::epsilon()*(absValue(withdrawn)+absValue(old.mass)))return reject(error,"aggregate mass donor exhausted",f);
        for(int s=0;s<Ns;++s) {
            const Real removed=dt*e.withdrawal[f].species[s]+maxValue(source.withdrawnSpecies[s],maxValue(0,-source.species[s])+maxValue(0,-source.phaseSpecies[s]));
            if(!finite(removed)||removed-old.species[s]>16*std::numeric_limits<Real>::epsilon()*(absValue(removed)+absValue(old.species[s])))return reject(error,"aggregate species donor exhausted",f);
        }
        auto& q=next.film[f];auto& a=next.filmAux[f];
        q.mass=(old.mass+dt*e.rhs[f].mass+source.mass)+source.phaseMass;
        for(int s=0;s<Ns;++s)q.species[s]=(old.species[s]+dt*e.rhs[f].species[s]+source.species[s])+source.phaseSpecies[s];
        const Real massScale=absValue(old.mass)+absValue(dt*e.rhs[f].mass)+absValue(source.mass)+absValue(source.phaseMass);
        const Real massRoundoff=16*std::numeric_limits<Real>::epsilon()*massScale;
        if((q.mass<0||source.exactPhaseEvent)&&absValue(q.mass)<=massRoundoff){report.numericalMassRoundoff-=q.mass;q.mass=0;}
        for(int species=0;species<Ns;++species){const Real scale=absValue(old.species[species])+absValue(dt*e.rhs[f].species[species])+absValue(source.species[species])+absValue(source.phaseSpecies[species]);
            const Real roundoff=16*std::numeric_limits<Real>::epsilon()*scale;
            if((q.species[species]<0||q.mass==0)&&absValue(q.species[species])<=roundoff)q.species[species]=0;
        }
        const Real oldPV=h.filmAux[f].pressure*old.mass/p.liquid.rho;
        const Real newPressure=endpointPressure(drive[f]);const Real newPV=newPressure*q.mass/p.liquid.rho;
        const Real before=old.enthalpy-oldPV;
        // Match assembleFilmCandidate: primary physical energy, primary K,
        // recorded radiation, then phase K and the exact phase remainder last.
        Real physical=before+dt*e.rhs[f].enthalpy;
        physical+=source.energy;physical+=source.interfaceKineticOutflow;
        physical+=source.radiationEnergy;physical+=source.phaseKineticOutflow;
        physical+=source.phaseEnergy;
        if(q.mass==0){const Real scale=absValue(before)+absValue(dt*e.rhs[f].enthalpy)+absValue(source.energy)+absValue(source.phaseEnergy)
            +absValue(source.interfaceKineticOutflow)+absValue(source.phaseKineticOutflow)+absValue(source.radiationEnergy);
            if(absValue(physical)<=32*std::numeric_limits<Real>::epsilon()*scale)physical=0;
        } // Any correction remains visible in the physical reduced-energy residual.
        q.enthalpy=physical+newPV;
        if(!validFilmInventory(q))return reject(error,"invalid candidate inventory or unaligned dryout event",f);
        if(!recoverFilmThermalState(q,h.surface.area[f],newPressure,p,h.surface.normal[f],coupledFace(h.surface,f),a))
            return reject(error,"candidate EOS or dryout event failed",f);
        a.solidFront=h.filmAux[f].solidFront+dt*e.aux[f].solidNormalVelocity;
        a.gasFront=h.filmAux[f].gasFront+dt*e.aux[f].normalVelocity;
        if(!finite(a.solidFront)||!finite(a.gasFront))return reject(error,"nonfinite front motion",f);
        report.phaseEvent=report.phaseEvent||((old.mass==0)!=(q.mass==0));
        report.internalEnergyBefore+=before;report.internalEnergyAfter+=q.enthalpy-newPV;
        budget.filmPressureVolume+=newPV-oldPV;
        budget.bodyWork+=dt*b.bodyPower;budget.supportWork+=dt*(b.supportPower+b.prescribedTopPower);
        budget.radiation+=dt*b.radiationPower;
        budget.boundaryEnergy+=dt*(b.edgeEnergyOutflow+b.edgeKineticOutflow);
        budget.exchangeMass[FilmParticipant]+=source.mass+source.phaseMass;
        budget.exchangeEnergy[FilmParticipant]+=source.energy+source.phaseEnergy;
        for(int s=0;s<Ns;++s)budget.exchangeSpecies[FilmParticipant][s]+=source.species[s]+source.phaseSpecies[s];
        budget.filmKineticAdvection+=source.interfaceKineticOutflow+source.phaseKineticOutflow;
        totalInput+=dt*(b.bodyPower+b.supportPower+b.prescribedTopPower+b.radiationPower)+source.energy+source.phaseEnergy+source.interfaceKineticOutflow+source.phaseKineticOutflow;
        totalKinetic+=dt*b.edgeKineticOutflow+source.interfaceKineticOutflow+source.phaseKineticOutflow;
    }
    // Recover endpoint profiles using endpoint pressure, composition and thickness.
    // This changes kinetic diagnostics only, never thermal inventory.
    HostState endpoint;endpoint.surface=h.surface;endpoint.film=next.film;endpoint.filmAux=next.filmAux;
    auto endpointDrive=drive;for(auto& d:endpointDrive)d.pressure=endpointPressure(d);
    Evaluation finalEvaluation;
    if(!recoverAndProfile(endpoint,p,endpointDrive,finalEvaluation,error))return false;
    next.filmAux=std::move(finalEvaluation.aux);
    Real edgeEnergy=0;
    for(std::size_t f=0;f<h.film.size();++f) {
        deltaK+=next.filmAux[f].kineticEnergy-h.filmAux[f].kineticEnergy;
        edgeEnergy+=dt*e.budget[f].edgeEnergyOutflow;
    }
    for(std::size_t edge=0;edge<h.surface.edgeOwner.size();++edge)if(h.surface.edgeNeighbour[edge]<0) {
        const auto& flux=e.edgeFlux[edge];budget.boundaryMass+=dt*flux.mass;
        for(int s=0;s<Ns;++s){budget.boundarySpecies[s]+=dt*flux.species[s];for(int element=0;element<p.nElements;++element)budget.boundaryElements[element]+=dt*flux.species[s]*p.species[s].element[element];}
    }
    report.reducedEnergyResidual=report.internalEnergyAfter-report.internalEnergyBefore+edgeEnergy-totalInput;
    report.kineticDefect=deltaK+totalKinetic;budget.filmKineticStorage=deltaK;
    budget.filmReducedResidual=report.reducedEnergyResidual;budget.filmKineticDefect=report.kineticDefect;
    budget.numericalEnergyResidual=report.reducedEnergyResidual;
    if(!finite(report.reducedEnergyResidual)||!finite(report.kineticDefect))return reject(error,"nonfinite energy audit",0);
    output=std::move(next);error.clear();return true;
}

bool evaluateCpuFilmTransport(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmDrive>& drive,std::vector<FilmQ>& rates,
    std::vector<FilmRateBudget>& budgets,std::string& error,
    std::vector<FilmAux>* refreshedAux,std::vector<Vec3>* pressureGradient,
    std::vector<Real>* sideMeshVolumeRate) {
    std::vector<CpuFilmForcing> owned(h.film.size());
    for(auto& force:owned)force.topWorkIncluded=force.bottomWorkIncluded=true;
    Evaluation e;if(!evaluate(h,p,dt,owned,drive,e,error))return false;
    rates=std::move(e.rhs);budgets=std::move(e.budget);
    if(refreshedAux)*refreshedAux=std::move(e.aux);
    if(pressureGradient)*pressureGradient=std::move(e.gradient);
    if(sideMeshVolumeRate)*sideMeshVolumeRate=std::move(e.sideMeshVolumeRate);
    error.clear();return true;
}

bool estimateCpuFilmStep(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Real forcingInterval,Real maxDt,Real& output,std::string& error) {
    if(!finite(maxDt)||maxDt<=0)return reject(error,"invalid local maximum interval",0);
    Evaluation e;if(!evaluate(h,p,forcingInterval,forcing,drive,e,error))return false;
    bool hasExactEvent=false;for(const auto& f:forcing)hasExactEvent=hasExactEvent||f.exactPhaseEvent;
    if(hasExactEvent){CpuFilmCandidate event;if(!advanceCpuFilmCandidate(h,p,forcingInterval,forcing,drive,event,error))return false;
        for(std::size_t f=0;f<forcing.size();++f)if(forcing[f].exactPhaseEvent){
            if(!(forcing[f].phaseMass<0||forcing[f].mass<0)||event.film[f].mass!=0||event.film[f].enthalpy!=0)return reject(error,"unproven exact phase event",f);
            for(int s=0;s<Ns;++s)if(event.film[f].species[s]!=0)return reject(error,"nonempty exact phase species",f);}}
    Real dt=minValue(maxDt,forcingInterval);const Real safety=.4;
    for(std::size_t f=0;f<h.film.size();++f) {
        const auto& q=h.film[f];const auto& a=e.aux[f];const auto& source=forcing[f];
        const Real withdrawal=e.withdrawal[f].mass+maxValue(source.withdrawnMass,maxValue(0,-source.mass)+maxValue(0,-source.phaseMass))/forcingInterval;
        if(withdrawal>0&&!source.exactPhaseEvent)dt=minValue(dt,safety*q.mass/withdrawal);
        for(int s=0;s<Ns;++s) {
            const Real removed=e.withdrawal[f].species[s]+maxValue(source.withdrawnSpecies[s],maxValue(0,-source.species[s])+maxValue(0,-source.phaseSpecies[s]))/forcingInterval;
            if(removed>0&&!source.exactPhaseEvent)dt=minValue(dt,safety*q.species[s]/removed);
        }
        if(e.outgoingLength[f]>0)dt=minValue(dt,safety*h.surface.area[f]/e.outgoingLength[f]);
        if(q.mass>0) {
            const Real cp=p.liquid.cp0+p.liquid.cp1*a.temperature;
            const Real capacity=q.mass*cp;
            if(e.conductance[f]>0)dt=minValue(dt,safety*capacity/e.conductance[f]);
            if(p.enableRadiation&&!coupledFace(h.surface,f)) {
                const Real radiationSlope=4*5.670374419e-8*p.emissivity*h.surface.area[f]*a.temperature*a.temperature*a.temperature;
                if(radiationSlope>0)dt=minValue(dt,safety*capacity/radiationSlope);
            }
            const Real energyRate=e.rhs[f].enthalpy+(source.energy+source.phaseEnergy+source.interfaceKineticOutflow+source.phaseKineticOutflow+source.radiationEnergy)/forcingInterval;
            const Real massRate=e.rhs[f].mass+(source.mass+source.phaseMass)/forcingInterval;
            const Real specificU=filmInternalEnergy(q,h.filmAux[f].pressure,p)/q.mass;
            const Real thermalRate=energyRate-specificU*massRate;
            if(thermalRate!=0&&!source.exactPhaseEvent) {
                const Real range=thermalRate>0?p.liquid.Tmax-a.temperature:a.temperature-p.liquid.Tmin;
                const Real permitted=minValue(.1*a.temperature,safety*range);
                dt=minValue(dt,capacity*permitted/absValue(thermalRate));
            }
        }
    }
    if(!finite(dt)||dt<=0)return reject(error,"local source/donor/thermal event requires alignment",0);
    output=dt;error.clear();return true;
}
} // namespace chmt
