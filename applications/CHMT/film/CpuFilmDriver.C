#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include "materials/StencilMath.H"
#include <algorithm>
#include <utility>

namespace chmt { namespace {
struct Evaluation {
    std::vector<std::size_t> incidentOffsets,incident;
    std::vector<FilmQ> q,rhs,withdrawal;
    std::vector<FilmAux> aux;
    std::vector<Vec3> shear,gradient;
    std::vector<Real> outgoingLength,conductance,sideMeshVolumeRate;
};
bool reject(std::string& error,const char* message,std::size_t index) {
    error=std::string("CPU film ")+message+" at "+std::to_string(index);return false;
}
bool numericReject(std::string& error,const char* message,std::size_t index,bool* numericalFailure){
    if(numericalFailure)*numericalFailure=true;
    return reject(error,message,index);
}
template<bool Terminal=false,class T> bool optionalSize(const std::vector<T>& value,std::size_t n) {
    return value.empty()||(Terminal?value.size()>=n:value.size()==n);
}
bool coupledFace(const SurfaceMesh& mesh,std::size_t f) {
    return !mesh.gasFace.empty()&&mesh.gasFace[f]>=0;
}
Real endpointPressure(const CpuFilmDrive& d) {
    // Only NaN is the documented omitted value; infinity is not omitted.
    return d.endpointPressure==d.endpointPressure?d.endpointPressure:d.pressure;
}
template<bool Terminal=false>
bool validate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    std::string& error,bool* numericalFailure) {
    if(numericalFailure)*numericalFailure=false;
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    if((Terminal?h.filmAux.size()<n:h.filmAux.size()!=n)||(Terminal?m.area.size()<n:m.area.size()!=n)||(Terminal?m.normal.size()<n:m.normal.size()!=n)
        ||(Terminal?forcing.size()<n:forcing.size()!=n)||(Terminal?drive.size()<n:drive.size()!=n)) {
        if(numericalFailure)*numericalFailure=h.filmAux.size()>=n&&m.area.size()>=n&&m.normal.size()>=n&&forcing.size()>=n&&drive.size()>=n;
        return reject(error,"invalid dimensions or interval",n);
    }
    if(!Terminal&&(!finite(dt)||dt<=0))return numericReject(error,"invalid dimensions or interval",n,numericalFailure);
    if(n==0) {
        if(ne) return reject(error,"edges without film faces",ne);
        return true;
    }
    if(!Terminal&&(!p.enableFilm||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged))
        return numericReject(error,"unsupported disabled or resolved-normal model",0,numericalFailure);
    if(!Terminal&&(!validCondensed(p.liquid)||!finite(p.liquid.conductivity)||p.liquid.conductivity<0
        ||!finite(p.liquidViscosity)||p.liquidViscosity<=0||!finite(p.gravity)
        ||(p.spatialOrder!=1&&p.spatialOrder!=2))) return numericReject(error,"invalid constitutive configuration",0,numericalFailure);
    if(!Terminal&&(p.enableRadiation&&(!finite(p.emissivity)||p.emissivity<0||p.emissivity>1
        ||!finite(p.ambientTemperature)||p.ambientTemperature<=0)))
        return numericReject(error,"invalid radiation configuration",0,numericalFailure);
    if(!optionalSize<Terminal>(m.gasFace,n)||!optionalSize<Terminal>(m.baseVelocity,n)
        ||!optionalSize<Terminal>(m.prescribedTopTraction,n)||!optionalSize<Terminal>(m.prescribedPressureGradient,n)
        ||(Terminal?m.edgeNeighbour.size()<ne:m.edgeNeighbour.size()!=ne)||(Terminal?m.edgeLength.size()<ne:m.edgeLength.size()!=ne)||(Terminal?m.edgeConormal.size()<ne:m.edgeConormal.size()!=ne)
        ||(Terminal?m.edgeOwnerOffset.size()<ne:m.edgeOwnerOffset.size()!=ne)||(Terminal?m.edgeNeighbourOffset.size()<ne:m.edgeNeighbourOffset.size()!=ne)
        ||!optionalSize<Terminal>(m.sweptEdgeArea,ne)) {
        if(numericalFailure)*numericalFailure=optionalSize<true>(m.gasFace,n)&&optionalSize<true>(m.baseVelocity,n)
            &&optionalSize<true>(m.prescribedTopTraction,n)&&optionalSize<true>(m.prescribedPressureGradient,n)
            &&m.edgeNeighbour.size()>=ne&&m.edgeLength.size()>=ne&&m.edgeConormal.size()>=ne
            &&m.edgeOwnerOffset.size()>=ne&&m.edgeNeighbourOffset.size()>=ne&&optionalSize<true>(m.sweptEdgeArea,ne);
        return reject(error,"invalid surface connectivity dimensions",n);
    }
    for(std::size_t f=0;f<n;++f) {
        const auto& a=forcing[f];const auto& d=drive[f];
        if(!Terminal&&(!finite(m.area[f])||m.area[f]<=0||!finite(m.normal[f])
            ||!closeEnough(mag(m.normal[f]),1,1e-12,1e-12)
            ||!finite(d.pressure)||!finite(endpointPressure(d))||!finite(d.shear)
            ||!finite(h.filmAux[f].pressure))) return numericReject(error,"invalid geometry or pressure/traction trace",f,numericalFailure);
        if(!Terminal&&(!finite(a.mass)||!finite(a.energy)||!finite(a.phaseMass)||!finite(a.phaseEnergy)
            ||!finite(a.phaseKineticOutflow)||!finite(a.interfaceKineticOutflow)
            ||!finite(a.radiationEnergy)||!finite(a.withdrawnMass)||a.withdrawnMass<0))
            return numericReject(error,"invalid integrated forcing",f,numericalFailure);
        Real mass=0,phaseMass=0,gross=0;
        for(int s=0;s<Ns;++s) {
            if(!Terminal&&(!finite(a.species[s])||!finite(a.phaseSpecies[s])||!finite(a.withdrawnSpecies[s])||a.withdrawnSpecies[s]<0))
                return numericReject(error,"invalid integrated species forcing",f,numericalFailure);
            mass+=a.species[s];phaseMass+=a.phaseSpecies[s];gross+=a.withdrawnSpecies[s];
        }
        if(!Terminal&&(!closeEnough(mass,a.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            return numericReject(error,"forcing species/mass mismatch",f,numericalFailure);
        if(!Terminal&&(!closeEnough(phaseMass,a.phaseMass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            return numericReject(error,"phase forcing species/mass mismatch",f,numericalFailure);
        if(!Terminal&&(!closeEnough(gross,a.withdrawnMass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            return numericReject(error,"gross withdrawal species/mass mismatch",f,numericalFailure);
        if(coupledFace(m,f)) {
            if(!Terminal&&((!m.prescribedTopTraction.empty()&&mag(m.prescribedTopTraction[f])!=0)
                ||(!m.prescribedPressureGradient.empty()&&mag(m.prescribedPressureGradient[f])!=0)))
                return numericReject(error,"coupled face has prescribed standalone drive",f,numericalFailure);
            if((h.film[f].mass>0||a.mass>0||a.phaseMass>0)&&!d.hasCoupledNormalTrace)
                return reject(error,"missing coupled normal trace",f);
            if(!Terminal&&(!finite(d.bottomNormalVelocity)||!finite(d.topNormalVelocity)
                ||!finite(d.interfaceNormalVelocity)||!finite(d.solidNormalVelocity)))
                return numericReject(error,"invalid coupled normal trace",f,numericalFailure);
        }
    }
    for(std::size_t e=0;e<ne;++e) {
        const int owner=m.edgeOwner[e],neighbour=m.edgeNeighbour[e];
        if(!Terminal&&(owner<0||static_cast<std::size_t>(owner)>=n||neighbour< -1
            ||(neighbour>=0&&static_cast<std::size_t>(neighbour)>=n)))return numericReject(error,"invalid surface edge",e,numericalFailure);
        if(!Terminal&&(owner==neighbour
            ||!finite(m.edgeLength[e])||m.edgeLength[e]<=0||!finite(m.edgeConormal[e])
            ||!closeEnough(mag(m.edgeConormal[e]),1,1e-12,1e-12)
            ||!finite(m.edgeOwnerOffset[e])||!finite(m.edgeNeighbourOffset[e])
            ||(!m.sweptEdgeArea.empty()&&!finite(m.sweptEdgeArea[e]))))
            return numericReject(error,"invalid surface edge",e,numericalFailure);
        // Geometry owns one shared hinge metric. On a curved surface its
        // conormal is tangent to the bisector plane, not either face alone.
        // Open edges retain their owner-face metric.
        const Vec3 hingeNormal=neighbour>=0?normalized(m.normal[owner]+m.normal[neighbour]):m.normal[owner];
        if(!Terminal&&(mag(hingeNormal)==0||!closeEnough(dot(m.edgeConormal[e],hingeNormal),0,1e-10,0)))
            return numericReject(error,"edge conormal is not tangential",e,numericalFailure);
        if(!Terminal&&(neighbour>=0&&dot(m.edgeOwnerOffset[e]-m.edgeNeighbourOffset[e],m.edgeConormal[e])<=0))
            return numericReject(error,"nonpositive surface edge distance",e,numericalFailure);
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
    if(!e.incident.empty())for(std::size_t entry=e.incidentOffsets[f];entry<e.incidentOffsets[f+1];++entry) {
        const std::size_t edge=e.incident[entry];
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        const bool isOwner=static_cast<std::size_t>(owner)==f;
        const int other=isOwner?neighbour:owner;
        Vec3 dx=m.edgeOwnerOffset[edge]-m.edgeNeighbourOffset[edge];if(!isOwner)dx=-dx;
        addLeastSquares(matrix,rhs,dx,surfaceValue(other,component,e,m)-value);
    }
    return solveLeastSquares(matrix,rhs);
}
template<bool Terminal=false>
FilmProfile faceProfile(std::size_t f,const Evaluation& e,const SurfaceMesh& m,const PhysicsConfig& p) {
    const auto& a=e.aux[f];const Vec3 normal=m.normal[f];
    const Vec3 gravity=p.gravity-normal*dot(p.gravity,normal);
    FilmProfile profile=filmProfile<Terminal>(a.thickness,p.liquidViscosity,a.baseVelocity,
        e.shear[f],e.gradient[f]-gravity*p.liquid.rho);
    return withNormalProfile(profile,normal,2*dot(a.meanVelocity,normal)-dot(a.topVelocity,normal),
        dot(a.topVelocity,normal));
}
template<bool Terminal=false>
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
            if(!e.incident.empty())for(std::size_t entry=e.incidentOffsets[f];entry<e.incidentOffsets[f+1];++entry) {
                const std::size_t edge=e.incident[entry];
                const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
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
        if(Terminal&&trial==47){result=q;return true;}
        scale*=.5;
    }
    return false;
}
template<bool Terminal=false>
bool recoverAndProfile(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmDrive>& drive,Evaluation& e,std::string& error,bool* numericalFailure=nullptr) {
    const auto& m=h.surface;const std::size_t n=h.film.size();
    e.q=h.film;e.aux=h.filmAux;e.shear.resize(n);e.gradient.resize(n);
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];const auto& d=drive[f];const bool coupled=coupledFace(m,f);
        a.baseVelocity=m.baseVelocity.empty()?Vec3{}:m.baseVelocity[f];a.normal=m.normal[f];
        FilmAux check=a;
        if(!recoverFilm<Terminal>(h.film[f],m.area[f],h.filmAux[f].pressure,p,check,nullptr,-1))
            return numericReject(error,"inadmissible accepted thermal inventory",f,numericalFailure);
        // Evaluation pressure changes storage, never the physical base U or T.
        e.q[f].enthalpy+=pressureVolumeProduct(h.filmAux[f].pressure,h.film[f].mass/p.liquid.rho,
            d.pressure,h.film[f].mass/p.liquid.rho);
        if(coupled) {
            if(!setCoupledFilmNormalTrace<Terminal>(a,m.normal[f],h.film[f].mass>0,d.bottomNormalVelocity,d.topNormalVelocity))
                return numericReject(error,"invalid coupled profile trace",f,numericalFailure);
            a.normalVelocity=d.interfaceNormalVelocity;a.solidNormalVelocity=d.solidNormalVelocity;
        }
        if(!recoverFilmThermalState<Terminal>(e.q[f],m.area[f],d.pressure,p,m.normal[f],coupled,a))
            return numericReject(error,"thermal recovery failed",f,numericalFailure);
        const Vec3 supplied=coupled?d.shear:(m.prescribedTopTraction.empty()?Vec3{}:m.prescribedTopTraction[f]);
        e.shear[f]=supplied-m.normal[f]*dot(supplied,m.normal[f]);
    }
    // Build here, not only in evaluate: endpoint recovery uses this path directly.
    // Ascending global edge IDs preserve each face's original reduction order.
    e.incidentOffsets.clear();e.incident.clear();
    for(std::size_t edge=0;edge<m.edgeOwner.size();++edge) {
        const int neighbour=m.edgeNeighbour[edge];
        if(neighbour>=0) {
            if(e.incidentOffsets.empty())e.incidentOffsets.assign(n+1,0);
            ++e.incidentOffsets[static_cast<std::size_t>(m.edgeOwner[edge])+1];
            ++e.incidentOffsets[static_cast<std::size_t>(neighbour)+1];
        }
    }
    if(!e.incidentOffsets.empty()) {
        for(std::size_t f=0;f<n;++f)e.incidentOffsets[f+1]+=e.incidentOffsets[f];
        e.incident.resize(e.incidentOffsets[n]);
        auto next=e.incidentOffsets;
        for(std::size_t edge=0;edge<m.edgeOwner.size();++edge) {
            const int neighbour=m.edgeNeighbour[edge];
            if(neighbour>=0) {
                e.incident[next[m.edgeOwner[edge]]++]=edge;
                e.incident[next[neighbour]++]=edge;
            }
        }
    }
    for(std::size_t f=0;f<n;++f) {
        const Vec3 normal=m.normal[f];Vec3 gradient=surfaceGradient(f,-1,e,m);
        if(!m.prescribedPressureGradient.empty())gradient+=m.prescribedPressureGradient[f];
        e.gradient[f]=gradient-normal*dot(gradient,normal);
        const Vec3 G=e.gradient[f]-(p.gravity-normal*dot(p.gravity,normal))*p.liquid.rho;
        if(!updateFilmProfileState<Terminal>(e.aux[f],p,normal,e.shear[f],G,coupledFace(m,f)))
            return numericReject(error,"profile update failed",f,numericalFailure);
    }
    return true;
}
void addRate(FilmQ& q,const FilmQ& flux,Real sign) {
    q.mass+=sign*flux.mass;q.enthalpy+=sign*flux.enthalpy;
    for(int s=0;s<Ns;++s)q.species[s]+=sign*flux.species[s];
}
template<bool Terminal=false>
bool evaluate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Evaluation& e,std::string& error,bool* numericalFailure=nullptr) {
    if(!validate<Terminal>(h,p,dt,forcing,drive,error,numericalFailure)||!recoverAndProfile<Terminal>(h,p,drive,e,error,numericalFailure))return false;
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    e.rhs.resize(n);e.withdrawal.resize(n);
    e.outgoingLength.assign(n,0);e.conductance.assign(n,0);e.sideMeshVolumeRate.assign(n,0);
    auto& sideVolumeRate=e.sideMeshVolumeRate;
    for(std::size_t edge=0;edge<ne;++edge) {
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        const Vec3 conormal=m.edgeConormal[edge];const Real length=m.edgeLength[edge];
        const Real meshRate=m.sweptEdgeArea.empty()?0:m.sweptEdgeArea[edge]/dt;
        const Vec3 velocity=neighbour>=0?(e.aux[owner].meanVelocity+e.aux[neighbour].meanVelocity)*.5:e.aux[owner].meanVelocity;
        const Real lengthRate=length*dot(velocity,conormal)-meshRate;
        // Open edges use the existing owner extrapolation for either flow direction.
        if(!Terminal&&!finite(lengthRate))return numericReject(error,"non-finite film edge transport rate",edge,numericalFailure);
        const int donor=lengthRate>=0||neighbour<0?owner:neighbour;
        FilmQ reconstructed;
        if(!reconstruct<Terminal>(donor,donor==owner?m.edgeOwnerOffset[edge]:m.edgeNeighbourOffset[edge],e,m,p,reconstructed))
            return numericReject(error,"edge reconstruction failed",edge,numericalFailure);
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
        if(!Terminal&&(!finite(flux.mass)||!finite(flux.enthalpy)))return numericReject(error,"nonfinite edge flux",edge,numericalFailure);
        addRate(e.rhs[owner],flux,-1);
        const FilmProfile profile=faceProfile<Terminal>(donor,e,m,p);
        sideVolumeRate[owner]+=profile.delta*meshRate;
        if(neighbour>=0) {
            addRate(e.rhs[neighbour],flux,1);
            sideVolumeRate[neighbour]-=profile.delta*meshRate;
        }
    }
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];Real bodyPower=0,supportPower=0,prescribedTopPower=0;const auto& source=forcing[f];
        const bool coupled=coupledFace(m,f);const FilmProfile profile=faceProfile<Terminal>(f,e,m,p);
        if(!coupled) {
            a.solidNormalVelocity=dot(a.baseVelocity,m.normal[f]);
            a.normalVelocity=a.solidNormalVelocity+((e.rhs[f].mass+(source.mass+source.phaseMass)/dt)/p.liquid.rho-sideVolumeRate[f])/a.area;
        }
        bodyPower=h.film[f].mass*dot(p.gravity,a.meanVelocity);
        if(!m.prescribedPressureGradient.empty())bodyPower-=h.film[f].mass/p.liquid.rho*dot(m.prescribedPressureGradient[f],a.meanVelocity);
        if(!source.topWorkIncluded)prescribedTopPower=a.area*(dot(e.shear[f],a.topVelocity)-a.pressure*a.normalVelocity);
        if(!source.bottomWorkIncluded)supportPower=a.area*(-dot(profile.bottomShear,a.baseVelocity)+a.pressure*a.solidNormalVelocity);
        const Real localRadiation=!coupled&&h.film[f].mass>0?a.area*grayRadiation(a.temperature,p):0;
        e.rhs[f].enthalpy+=bodyPower+supportPower+prescribedTopPower+localRadiation;
        if(!Terminal&&(!finite(e.rhs[f].enthalpy)||!finite(a.normalVelocity)))
            return numericReject(error,"nonfinite surface source",f,numericalFailure);
    }
    return true;
}
} // namespace

template<bool Terminal>
bool finalizeCpuFilmPhasePacketImpl(const FilmQ& accepted,Real acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,Vec3 bottomVelocity,
    const PhysicsConfig& p,ExchangePacket& phase,bool& exactEvent,std::string& error,bool* numericalFailure) {
    if(numericalFailure)*numericalFailure=true;
    if(!Terminal&&(!finite(dt)||dt<=0||!finite(acceptedPressure)||!finite(bottomVelocity)
        ||!finite(p.liquid.rho)||p.liquid.rho<=0||!validFilmInventory(accepted)
        ||phase.kind!=ExchangeKind::SolidFilm||!finite(phase.mass))) {
        error="invalid terminal film phase input";return false;
    }
    const Real available=(accepted.mass+dt*localRate.mass)+primary.mass;
    if(!Terminal&&(!finite(available))){error="nonfinite terminal film mass";return false;}
    const bool freezing=phase.mass<0&&available>=0&&-phase.mass>=available;
    const bool gasOrTransportEmpty=accepted.mass>0&&available==0&&phase.mass==0
        &&(primary.mass<0||localRate.mass<0);
    if(!freezing&&!gasOrTransportEmpty){exactEvent=false;error.clear();return true;}
    ExchangePacket candidate=phase;candidate.mass=-available;
    if(!Terminal&&(candidate.mass!=0&&(p.material.phaseCondensed<0||p.material.phaseCondensed>=Nc))) {
        error="terminal film phase has no condensed owner species";return false;
    }
    for(int c=0;c<Nc;++c)candidate.condensed[c]=c==p.material.phaseCondensed?candidate.mass:0;
    for(int s=0;s<Ns;++s) {
        const Real species=(accepted.species[s]+dt*localRate.species[s])+primary.species[s];
        if(!Terminal&&(!closeEnough(species,available*p.material.phaseFilmY[s],p.tolerances.absoluteMass,p.tolerances.relativeMass))) {
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
    if(!Terminal&&(!validatePacketMath(candidate,p.tolerances))){error="invalid terminal film phase packet";return false;}
    phase=candidate;exactEvent=true;error.clear();return true;
}

template<bool Terminal>
bool advanceCpuFilmCandidateImpl(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    CpuFilmCandidate& output,std::string& error,bool* numericalFailure) {
    Evaluation e;if(!evaluate<Terminal>(h,p,dt,forcing,drive,e,error,numericalFailure))return false;
    CpuFilmCandidate next;next.film=h.film;next.filmAux=e.aux;
    for(std::size_t f=0;f<h.film.size();++f) {
        const auto& old=h.film[f];const auto& source=forcing[f];
        const Real withdrawn=dt*e.withdrawal[f].mass+maxValue(source.withdrawnMass,maxValue(0,-source.mass)+maxValue(0,-source.phaseMass));
        if(!Terminal&&(!finite(withdrawn)||withdrawn-old.mass>16*std::numeric_limits<Real>::epsilon()*(absValue(withdrawn)+absValue(old.mass))))return numericReject(error,"aggregate mass donor exhausted",f,numericalFailure);
        for(int s=0;s<Ns;++s) {
            const Real removed=dt*e.withdrawal[f].species[s]+maxValue(source.withdrawnSpecies[s],maxValue(0,-source.species[s])+maxValue(0,-source.phaseSpecies[s]));
            if(!Terminal&&(!finite(removed)||removed-old.species[s]>16*std::numeric_limits<Real>::epsilon()*(absValue(removed)+absValue(old.species[s]))))return numericReject(error,"aggregate species donor exhausted",f,numericalFailure);
        }
        auto& q=next.film[f];auto& a=next.filmAux[f];
        q.mass=(old.mass+dt*e.rhs[f].mass+source.mass)+source.phaseMass;
        for(int s=0;s<Ns;++s)q.species[s]=(old.species[s]+dt*e.rhs[f].species[s]+source.species[s])+source.phaseSpecies[s];
        const Real massScale=absValue(old.mass)+absValue(dt*e.rhs[f].mass)+absValue(source.mass)+absValue(source.phaseMass);
        const Real massRoundoff=16*std::numeric_limits<Real>::epsilon()*massScale;
        if((q.mass<0||source.exactPhaseEvent)&&absValue(q.mass)<=massRoundoff){q.mass=0;}
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
        }
        q.enthalpy=physical+newPV;
        if(!Terminal&&(!validFilmInventory(q)))return numericReject(error,"invalid candidate inventory or unaligned dryout event",f,numericalFailure);
        if(!recoverFilmThermalState<Terminal>(q,h.surface.area[f],newPressure,p,h.surface.normal[f],coupledFace(h.surface,f),a))
            return numericReject(error,"candidate EOS or dryout event failed",f,numericalFailure);
        a.solidFront=h.filmAux[f].solidFront+dt*e.aux[f].solidNormalVelocity;
        a.gasFront=h.filmAux[f].gasFront+dt*e.aux[f].normalVelocity;
        if(!Terminal&&(!finite(a.solidFront)||!finite(a.gasFront)))return numericReject(error,"nonfinite front motion",f,numericalFailure);
    }
    // Recover endpoint profiles using endpoint pressure, composition and thickness.
    // The recovered auxiliary profile is the actual accepted endpoint state.
    HostState endpoint;endpoint.surface=h.surface;endpoint.film=next.film;endpoint.filmAux=next.filmAux;
    auto endpointDrive=drive;for(auto& d:endpointDrive)d.pressure=endpointPressure(d);
    Evaluation finalEvaluation;
    if(!recoverAndProfile<Terminal>(endpoint,p,endpointDrive,finalEvaluation,error,numericalFailure))return false;
    next.filmAux=std::move(finalEvaluation.aux);
    output=std::move(next);error.clear();return true;
}

template<bool Terminal>
bool evaluateCpuFilmTransportImpl(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmDrive>& drive,std::vector<FilmQ>& rates,
    std::string& error,
    std::vector<FilmAux>* refreshedAux,std::vector<Vec3>* pressureGradient,
    std::vector<Real>* sideMeshVolumeRate,bool* numericalFailure) {
    std::vector<CpuFilmForcing> owned(h.film.size());
    for(auto& force:owned)force.topWorkIncluded=force.bottomWorkIncluded=true;
    Evaluation e;if(!evaluate<Terminal>(h,p,dt,owned,drive,e,error,numericalFailure))return false;
    rates=std::move(e.rhs);
    if(refreshedAux)*refreshedAux=std::move(e.aux);
    if(pressureGradient)*pressureGradient=std::move(e.gradient);
    if(sideMeshVolumeRate)*sideMeshVolumeRate=std::move(e.sideMeshVolumeRate);
    error.clear();return true;
}

template<bool Terminal>
bool estimateCpuFilmStepImpl(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Real forcingInterval,Real maxDt,Real& output,std::string& error,bool* numericalFailure) {
    if(!Terminal&&(!finite(maxDt)||maxDt<=0))return numericReject(error,"invalid local maximum interval",0,numericalFailure);
    Evaluation e;if(!evaluate<Terminal>(h,p,forcingInterval,forcing,drive,e,error,numericalFailure))return false;
    bool hasExactEvent=false;for(const auto& f:forcing)hasExactEvent=hasExactEvent||f.exactPhaseEvent;
    if(hasExactEvent){CpuFilmCandidate event;if(!advanceCpuFilmCandidateImpl<Terminal>(h,p,forcingInterval,forcing,drive,event,error,numericalFailure))return false;
        for(std::size_t f=0;f<forcing.size();++f)if(!Terminal&&(forcing[f].exactPhaseEvent)){
            if(!Terminal&&(!(forcing[f].phaseMass<0||forcing[f].mass<0)||event.film[f].mass!=0||event.film[f].enthalpy!=0))return numericReject(error,"unproven exact phase event",f,numericalFailure);
            for(int s=0;s<Ns;++s)if(!Terminal&&(event.film[f].species[s]!=0))return numericReject(error,"nonempty exact phase species",f,numericalFailure);}}
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
    if(!Terminal&&(!finite(dt)||dt<=0))return numericReject(error,"local source/donor/thermal event requires alignment",0,numericalFailure);
    output=dt;error.clear();return true;
}

bool finalizeCpuFilmPhasePacket(const FilmQ& q,Real pressure,const FilmQ& rate,const CpuFilmForcing& f,Real dt,Vec3 velocity,const PhysicsConfig& p,ExchangePacket& phase,bool& exact,std::string& error,bool* numerical){return finalizeCpuFilmPhasePacketImpl<false>(q,pressure,rate,f,dt,velocity,p,phase,exact,error,numerical);}
bool advanceCpuFilmCandidate(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,CpuFilmCandidate& out,std::string& error,bool* numerical){return advanceCpuFilmCandidateImpl<false>(h,p,dt,f,d,out,error,numerical);}
bool evaluateCpuFilmTransport(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmDrive>& d,std::vector<FilmQ>& rates,std::string& error,std::vector<FilmAux>* aux,std::vector<Vec3>* gradient,std::vector<Real>* side,bool* numerical){return evaluateCpuFilmTransportImpl<false>(h,p,dt,d,rates,error,aux,gradient,side,numerical);}
bool estimateCpuFilmStep(const HostState& h,const PhysicsConfig& p,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,Real interval,Real maximum,Real& out,std::string& error,bool* numerical){return estimateCpuFilmStepImpl<false>(h,p,f,d,interval,maximum,out,error,numerical);}
namespace detail {
bool completeCpuFilmPhasePacket(const FilmQ& q,Real pressure,const FilmQ& rate,const CpuFilmForcing& f,Real dt,Vec3 velocity,const PhysicsConfig& p,ExchangePacket& phase,bool& exact,std::string& error,bool* numerical){return finalizeCpuFilmPhasePacketImpl<true>(q,pressure,rate,f,dt,velocity,p,phase,exact,error,numerical);}
bool completeCpuFilmCandidate(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,CpuFilmCandidate& out,std::string& error,bool* numerical){return advanceCpuFilmCandidateImpl<true>(h,p,dt,f,d,out,error,numerical);}
bool completeCpuFilmTransport(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmDrive>& d,std::vector<FilmQ>& rates,std::string& error,std::vector<FilmAux>* aux,std::vector<Vec3>* gradient,std::vector<Real>* side,bool* numerical){return evaluateCpuFilmTransportImpl<true>(h,p,dt,d,rates,error,aux,gradient,side,numerical);}
bool completeCpuFilmStep(const HostState& h,const PhysicsConfig& p,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,Real interval,Real maximum,Real& out,std::string& error,bool* numerical){return estimateCpuFilmStepImpl<true>(h,p,f,d,interval,maximum,out,error,numerical);}
}
} // namespace chmt
