#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include "materials/StencilMath.H"
#include "materials/EnergyStencilMath.H"
#include <algorithm>
#include <utility>

namespace chmt { namespace {
struct Evaluation {
    std::vector<std::size_t> incidentOffsets,incident;
    std::vector<FilmQ> q,rhs,withdrawal;
    std::vector<FilmAux> aux;
    std::vector<PhysicalVec3> shear,gradient;
    std::vector<PhysicalReal> outgoingLength;std::vector<Real> sideMeshVolumeRate;
    std::vector<EnergyReal> conductance;
};
bool reject(std::string& error,const char* message,std::size_t index) {
    error=std::string("CPU film ")+message+" at "+std::to_string(index);return false;
}
bool numericReject(std::string& error,const char* message,std::size_t index,bool* numericalFailure,PhysicalReal observed=0,PhysicalReal reference=0){
    if(numericalFailure)*numericalFailure=true;
    reject(error,message,index);error+=" observed="+std::to_string(observed)+" reference="+std::to_string(reference);return false;
}
template<class T> bool optionalSize(const std::vector<T>& value,std::size_t n) {
    return value.empty()||(value.size()==n);
}
bool coupledFace(const SurfaceMesh& mesh,std::size_t f) {
    return !mesh.gasFace.empty()&&mesh.gasFace[f]>=0;
}
PhysicalReal endpointPressure(const CpuFilmDrive& d) {
    // Only NaN is the documented omitted value; infinity is not omitted.
    return d.endpointPressure==d.endpointPressure?d.endpointPressure:d.pressure;
}
bool validate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    std::string& error,bool* numericalFailure) {
    if(numericalFailure)*numericalFailure=false;
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    if((h.filmAux.size()!=n)||(m.area.size()!=n)||(m.normal.size()!=n)
        ||(forcing.size()!=n)||(drive.size()!=n)) {
        if(numericalFailure)*numericalFailure=h.filmAux.size()>=n&&m.area.size()>=n&&m.normal.size()>=n&&forcing.size()>=n&&drive.size()>=n;
        return reject(error,"invalid dimensions or interval",n);
    }
    if((!finite(dt)||dt<=0))return numericReject(error,"invalid dimensions or interval",n,numericalFailure,dt,0);
    if(n==0) {
        if(ne) return reject(error,"edges without film faces",ne);
        return true;
    }
    if((!p.enableFilm||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged))
        return numericReject(error,"unsupported disabled or resolved-normal model",0,numericalFailure);
    if((!validCondensed(p.liquid)||!finite(p.liquid.conductivity)||p.liquid.conductivity<0
        ||!finite(p.liquidViscosity)||p.liquidViscosity<=0||!finite(p.gravity)
        ||(p.spatialOrder!=1&&p.spatialOrder!=2))) return numericReject(error,"invalid constitutive configuration",0,numericalFailure);
    if((p.enableRadiation&&(!finite(p.emissivity)
        ||!finite(p.ambientTemperature)||p.ambientTemperature<=0)))
        return numericReject(error,"invalid radiation configuration",0,numericalFailure);
    if(!optionalSize(m.gasFace,n)||!optionalSize(m.baseVelocity,n)
        ||!optionalSize(m.prescribedTopTraction,n)||!optionalSize(m.prescribedPressureGradient,n)
        ||(m.edgeNeighbour.size()!=ne)||(m.edgeLength.size()!=ne)||(m.edgeConormal.size()!=ne)
        ||(m.edgeOwnerOffset.size()!=ne)||(m.edgeNeighbourOffset.size()!=ne)
        ||!optionalSize(m.sweptEdgeArea,ne)) {
        if(numericalFailure)*numericalFailure=optionalSize(m.gasFace,n)&&optionalSize(m.baseVelocity,n)
            &&optionalSize(m.prescribedTopTraction,n)&&optionalSize(m.prescribedPressureGradient,n)
            &&m.edgeNeighbour.size()>=ne&&m.edgeLength.size()>=ne&&m.edgeConormal.size()>=ne
            &&m.edgeOwnerOffset.size()>=ne&&m.edgeNeighbourOffset.size()>=ne&&optionalSize(m.sweptEdgeArea,ne);
        return reject(error,"invalid surface connectivity dimensions",n);
    }
    for(std::size_t f=0;f<n;++f) {
        const auto& a=forcing[f];const auto& d=drive[f];
        if((!finite(m.area[f])||m.area[f]<=0||!finite(m.normal[f])
            ||mag(m.normal[f])==0
            ||!finite(d.pressure)||!finite(endpointPressure(d))||!finite(d.shear)
            ||!finite(h.filmAux[f].pressure))) return numericReject(error,"invalid geometry or pressure/traction trace",f,numericalFailure,m.area[f],d.pressure);
        if(!closeEnough(mag(m.normal[f]),1,PhysicalReal(1e-12),PhysicalReal(1e-12)))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,4210,mag(m.normal[f])-1,PhysicalReal(1e-12));
        if((!finite(a.mass)||!finite(a.energy)||!finite(a.phaseMass)||!finite(a.phaseEnergy)
            ||!finite(a.phaseKineticOutflow)||!finite(a.interfaceKineticOutflow)
            ||!finite(a.radiationEnergy)||!finite(a.withdrawnMass)||a.withdrawnMass<0))
            return numericReject(error,"invalid integrated forcing",f,numericalFailure,a.mass,a.energy);
        PhysicalReal mass=0,phaseMass=0,gross=0;
        for(int s=0;s<Ns;++s) {
            if((!finite(a.species[s])||!finite(a.phaseSpecies[s])||!finite(a.withdrawnSpecies[s])||a.withdrawnSpecies[s]<0))
                return numericReject(error,"invalid integrated species forcing",f,numericalFailure,a.species[s],s);
            mass+=a.species[s];phaseMass+=a.phaseSpecies[s];gross+=a.withdrawnSpecies[s];
        }
        if((!closeEnough(mass,a.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,4201,mass-a.mass,0);
        if((!closeEnough(phaseMass,a.phaseMass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,10051,phaseMass-a.phaseMass,0);
        if((!closeEnough(gross,a.withdrawnMass,p.tolerances.absoluteMass,p.tolerances.relativeMass)))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,10052,gross-a.withdrawnMass,0);
        if(coupledFace(m,f)) {
            if(((!m.prescribedTopTraction.empty()&&mag(m.prescribedTopTraction[f])!=0)
                ||(!m.prescribedPressureGradient.empty()&&mag(m.prescribedPressureGradient[f])!=0)))
                return numericReject(error,"coupled face has prescribed standalone drive",f,numericalFailure);
            if((h.film[f].mass>0||a.mass>0||a.phaseMass>0)&&!d.hasCoupledNormalTrace)
                return reject(error,"missing coupled normal trace",f);
            if((!finite(d.bottomNormalVelocity)||!finite(d.topNormalVelocity)
                ||!finite(d.interfaceNormalVelocity)||!finite(d.solidNormalVelocity)))
                return numericReject(error,"invalid coupled normal trace",f,numericalFailure);
        }
    }
    for(std::size_t e=0;e<ne;++e) {
        const int owner=m.edgeOwner[e],neighbour=m.edgeNeighbour[e];
        if((owner<0||static_cast<std::size_t>(owner)>=n||neighbour< -1
            ||(neighbour>=0&&static_cast<std::size_t>(neighbour)>=n)))return numericReject(error,"invalid surface edge",e,numericalFailure);
        if((owner==neighbour
            ||!finite(m.edgeLength[e])||m.edgeLength[e]<=0||!finite(m.edgeConormal[e])
            ||mag(m.edgeConormal[e])==0
            ||!finite(m.edgeOwnerOffset[e])||!finite(m.edgeNeighbourOffset[e])
            ||(!m.sweptEdgeArea.empty()&&!finite(m.sweptEdgeArea[e]))))
            return numericReject(error,"invalid surface edge",e,numericalFailure);
        // Geometry owns one shared hinge metric. On a curved surface its
        // conormal is tangent to the bisector plane, not either face alone.
        // Open edges retain their owner-face metric.
        const Vec3 hingeNormal=neighbour>=0?normalized(m.normal[owner]+m.normal[neighbour]):m.normal[owner];
        if(mag(hingeNormal)==0)
            return numericReject(error,"undefined hinge normal",e,numericalFailure,mag(hingeNormal),0);
        if(!closeEnough(mag(m.edgeConormal[e]),1,PhysicalReal(1e-12),PhysicalReal(1e-12)))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,4211,mag(m.edgeConormal[e])-1,PhysicalReal(1e-12));
        if(!closeEnough(dot(m.edgeConormal[e],hingeNormal),0,PhysicalReal(1e-10),0))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,4212,dot(m.edgeConormal[e],hingeNormal),PhysicalReal(1e-10));
        if((neighbour>=0&&dot(m.edgeOwnerOffset[e]-m.edgeNeighbourOffset[e],m.edgeConormal[e])<=0))
            return numericReject(error,"nonpositive surface edge distance",e,numericalFailure,dot(m.edgeOwnerOffset[e]-m.edgeNeighbourOffset[e],m.edgeConormal[e]),0);
    }
    return true;
}
PhysicalReal surfaceValue(std::size_t f,int component,const Evaluation& e,const SurfaceMesh& m) {
    if(component<0)return e.aux[f].pressure;
    if(component==0)return e.q[f].mass/PhysicalReal(m.area[f]);
    if(component==1)return e.q[f].enthalpy/PhysicalReal(m.area[f]);
    return e.q[f].species[component-2]/PhysicalReal(m.area[f]);
}
PhysicalVec3 surfaceGradient(std::size_t f,int component,const Evaluation& e,const SurfaceMesh& m) {
    namespace stencil=materialEnergyStencil;
    stencil::Symmetric3 matrix;stencil::Vec3 rhs{};
    const PhysicalReal value=surfaceValue(f,component,e,m);
    if(!e.incident.empty())for(std::size_t entry=e.incidentOffsets[f];entry<e.incidentOffsets[f+1];++entry) {
        const std::size_t edge=e.incident[entry];
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        const bool isOwner=static_cast<std::size_t>(owner)==f;
        const int other=isOwner?neighbour:owner;
        const auto a=toPhysical(m.edgeOwnerOffset[edge]),b=toPhysical(m.edgeNeighbourOffset[edge]);
        stencil::Vec3 dx{a.x-b.x,a.y-b.y,a.z-b.z};if(!isOwner)dx=dx*PhysicalReal(-1);
        stencil::addLeastSquares(matrix,rhs,dx,surfaceValue(other,component,e,m)-value);
    }
    const auto gradient=stencil::solveLeastSquares(matrix,rhs);
    return {gradient.x,gradient.y,gradient.z};
}
FilmProfile faceProfile(std::size_t f,const Evaluation& e,const SurfaceMesh& m,const PhysicsConfig& p) {
    const auto& a=e.aux[f];const PhysicalVec3 normal=toPhysical(m.normal[f]);
    const PhysicalVec3 gravity=p.gravity-normal*dot(p.gravity,normal);
    FilmProfile profile=filmProfile(a.thickness,p.liquidViscosity,a.baseVelocity,
        e.shear[f],e.gradient[f]-gravity*p.liquid.rho);
    return withNormalProfile(profile,normal,2*dot(a.meanVelocity,normal)-dot(a.topVelocity,normal),
        dot(a.topVelocity,normal));
}
bool reconstruct(std::size_t f,PhysicalVec3 offset,const Evaluation& e,const SurfaceMesh& m,
    const PhysicsConfig& p,FilmQ& result) {
    FilmQ base; base.mass=surfaceValue(f,0,e,m);base.enthalpy=surfaceValue(f,1,e,m);
    for(int s=0;s<Ns;++s)base.species[s]=surfaceValue(f,s+2,e,m);
    if(p.spatialOrder==1||base.mass==0){result=base;return true;}
    FilmQ increment;increment.mass=dot(surfaceGradient(f,0,e,m),offset);
    increment.enthalpy=dot(surfaceGradient(f,1,e,m),offset);
    for(int s=0;s<Ns;++s)increment.species[s]=dot(surfaceGradient(f,s+2,e,m),offset);
    if(!finite(increment.mass)||!finite(increment.enthalpy))return false;
    for(int s=0;s<Ns;++s)if(!finite(increment.species[s]))return false;
    PhysicalReal scale=1;
    if(p.reconstruction==ReconstructionMode::LimitedLinear) {
        for(int component=0;component<Ns+2;++component) {
            const PhysicalReal value=surfaceValue(f,component,e,m);PhysicalReal lo=value,hi=value;
            if(!e.incident.empty())for(std::size_t entry=e.incidentOffsets[f];entry<e.incidentOffsets[f+1];++entry) {
                const std::size_t edge=e.incident[entry];
                const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
                const int other=static_cast<std::size_t>(owner)==f?neighbour:owner;
                const PhysicalReal v=surfaceValue(other,component,e,m);lo=energyArithmetic::minimum(lo,v);hi=energyArithmetic::maximum(hi,v);
            }
            const PhysicalReal change=component==0?increment.mass:(component==1?increment.enthalpy:increment.species[component-2]);
            scale=energyArithmetic::minimum(scale,materialEnergyStencil::barthJespersen(value,lo,hi,change));
        }
    }
    FilmQ q;q.mass=base.mass+scale*increment.mass;q.enthalpy=base.enthalpy+EnergyReal(scale)*increment.enthalpy;
    for(int s=0;s<Ns;++s)q.species[s]=base.species[s]+scale*increment.species[s];
    if(q.mass!=0)for(int species=0;species<Ns;++species){
        PhysicalReal Y=q.species[species]/q.mass;const PhysicalReal originalY=Y;
        if(!clipPhysicsMassFraction(p,Y,4203,species))return false;
        if(Y!=originalY)q.species[species]=q.mass*Y;
    }
    FilmAux aux;
    if(!recoverFilm(q,1,e.aux[f].pressure,p,aux))return false;
    result=q;return true;
}
bool recoverAndProfile(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmDrive>& drive,Evaluation& e,std::string& error,bool* numericalFailure=nullptr) {
    const auto& m=h.surface;const std::size_t n=h.film.size();
    e.q=h.film;e.aux=h.filmAux;e.shear.resize(n);e.gradient.resize(n);
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];const auto& d=drive[f];const bool coupled=coupledFace(m,f);
        a.baseVelocity=m.baseVelocity.empty()?PhysicalVec3{}:physicalVectorInput(m.baseVelocity[f]);a.normal=toPhysical(m.normal[f]);
        FilmAux check=a;
        if(!recoverFilm(h.film[f],m.area[f],h.filmAux[f].pressure,p,check,nullptr,-1))
            return numericReject(error,"inadmissible accepted thermal inventory",f,numericalFailure);
        // Evaluation pressure changes storage, never the physical base U or T.
        e.q[f].enthalpy+=pressureVolumeProduct(h.filmAux[f].pressure,EnergyReal(h.film[f].mass)/EnergyReal(p.liquid.rho),
            d.pressure,EnergyReal(h.film[f].mass)/EnergyReal(p.liquid.rho));
        if(coupled) {
            if(!setCoupledFilmNormalTrace(a,toPhysical(m.normal[f]),h.film[f].mass>0,d.bottomNormalVelocity,d.topNormalVelocity,nullptr,static_cast<int>(f),p.diagnostics))
                return numericReject(error,"invalid coupled profile trace",f,numericalFailure);
            a.normalVelocity=d.interfaceNormalVelocity;a.solidNormalVelocity=d.solidNormalVelocity;
        }
        if(!recoverFilmThermalState(e.q[f],m.area[f],d.pressure,p,toPhysical(m.normal[f]),coupled,a))
            return numericReject(error,"thermal recovery failed",f,numericalFailure);
        const PhysicalVec3 supplied=coupled?d.shear:(m.prescribedTopTraction.empty()?PhysicalVec3{}:physicalVectorInput(m.prescribedTopTraction[f]));
        e.shear[f]=supplied-toPhysical(m.normal[f])*dot(supplied,toPhysical(m.normal[f]));
    }


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
        const PhysicalVec3 normal=toPhysical(m.normal[f]);PhysicalVec3 gradient=surfaceGradient(f,-1,e,m);
        if(!finite(gradient))return numericReject(error,"nonfinite film pressure gradient",f,numericalFailure);
        if(!m.prescribedPressureGradient.empty())gradient+=physicalVectorInput(m.prescribedPressureGradient[f]);
        e.gradient[f]=gradient-normal*dot(gradient,normal);
        const PhysicalVec3 G=e.gradient[f]-(p.gravity-normal*dot(p.gravity,normal))*p.liquid.rho;
        if(!updateFilmProfileState(e.aux[f],p,normal,e.shear[f],G,coupledFace(m,f)))
            return numericReject(error,"profile update failed",f,numericalFailure);
    }
    return true;
}
void addRate(FilmQ& q,const FilmQ& flux,PhysicalReal sign) {
    q.mass+=sign*flux.mass;q.enthalpy+=EnergyReal(sign)*flux.enthalpy;
    for(int s=0;s<Ns;++s)q.species[s]+=sign*flux.species[s];
}
bool evaluate(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Evaluation& e,std::string& error,bool* numericalFailure=nullptr) {
    if(!validate(h,p,dt,forcing,drive,error,numericalFailure)||!recoverAndProfile(h,p,drive,e,error,numericalFailure))return false;
    const auto& m=h.surface;const std::size_t n=h.film.size(),ne=m.edgeOwner.size();
    e.rhs.resize(n);e.withdrawal.resize(n);
    e.outgoingLength.assign(n,0);e.conductance.assign(n,0);e.sideMeshVolumeRate.assign(n,0);
    auto& sideVolumeRate=e.sideMeshVolumeRate;
    for(std::size_t edge=0;edge<ne;++edge) {
        const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];
        const PhysicalVec3 conormal=toPhysical(m.edgeConormal[edge]);const PhysicalReal length=m.edgeLength[edge];
        const Real meshRate=m.sweptEdgeArea.empty()?0:m.sweptEdgeArea[edge]/dt;
        const PhysicalVec3 velocity=neighbour>=0?(e.aux[owner].meanVelocity+e.aux[neighbour].meanVelocity)*PhysicalReal(.5):e.aux[owner].meanVelocity;
        const PhysicalReal lengthRate=length*dot(velocity,conormal)-PhysicalReal(meshRate);

        if(!finite(lengthRate))return numericReject(error,"non-finite film edge transport rate",edge,numericalFailure,lengthRate,0);
        const int donor=lengthRate>=0||neighbour<0?owner:neighbour;
        FilmQ reconstructed;
        if(!reconstruct(donor,donor==owner?toPhysical(m.edgeOwnerOffset[edge]):toPhysical(m.edgeNeighbourOffset[edge]),e,m,p,reconstructed))
            return numericReject(error,"edge reconstruction failed",edge,numericalFailure);
        FilmQ flux=filmAdvectiveFlux(reconstructed,lengthRate);
        flux.enthalpy+=EnergyReal(e.aux[donor].pressure)*EnergyReal(reconstructed.mass)/EnergyReal(p.liquid.rho)*EnergyReal(meshRate);
        e.outgoingLength[donor]+=energyArithmetic::absolute(lengthRate);
        e.withdrawal[donor].mass+=energyArithmetic::absolute(flux.mass);
        for(int s=0;s<Ns;++s)e.withdrawal[donor].species[s]+=energyArithmetic::absolute(flux.species[s]);
        if(neighbour>=0&&h.film[owner].mass>0&&h.film[neighbour].mass>0) {
            const auto left=materialEnergyStencil::localVector(m.edgeOwnerOffset[edge]),right=materialEnergyStencil::localVector(m.edgeNeighbourOffset[edge]);
            const EnergyReal distance=materialEnergyStencil::dot({left.x-right.x,left.y-right.y,left.z-right.z},materialEnergyStencil::Vec3{conormal.x,conormal.y,conormal.z});
            const EnergyReal conductance=EnergyReal(p.liquid.conductivity)*EnergyReal(length)*EnergyReal(.5)*(EnergyReal(e.aux[owner].thickness)+EnergyReal(e.aux[neighbour].thickness))/distance;
            flux.enthalpy+=conductance*(e.aux[owner].temperature-e.aux[neighbour].temperature);
            e.conductance[owner]+=conductance;e.conductance[neighbour]+=conductance;
        }
        if((!finite(flux.mass)||!finite(flux.enthalpy)))return numericReject(error,"nonfinite edge flux",edge,numericalFailure,flux.mass,flux.enthalpy);
        addRate(e.rhs[owner],flux,-1);
        const FilmProfile profile=faceProfile(donor,e,m,p);
        sideVolumeRate[owner]+=e.aux[donor].thickness*meshRate;
        if(neighbour>=0) {
            addRate(e.rhs[neighbour],flux,1);
            sideVolumeRate[neighbour]-=e.aux[donor].thickness*meshRate;
        }
    }
    for(std::size_t f=0;f<n;++f) {
        auto& a=e.aux[f];EnergyReal bodyPower=0,supportPower=0,prescribedTopPower=0;const auto& source=forcing[f];
        const bool coupled=coupledFace(m,f);const FilmProfile profile=faceProfile(f,e,m,p);
        if(!coupled) {
            a.solidNormalVelocity=dot(a.baseVelocity,toPhysical(m.normal[f]));
            a.normalVelocity=a.solidNormalVelocity+((e.rhs[f].mass+(source.mass+source.phaseMass)/PhysicalReal(dt))/p.liquid.rho-PhysicalReal(sideVolumeRate[f]))/PhysicalReal(a.area);
        }
        bodyPower=EnergyReal(h.film[f].mass)*surfaceEnergyDot(p.gravity,a.meanVelocity);
        if(!m.prescribedPressureGradient.empty())bodyPower-=EnergyReal(h.film[f].mass)/EnergyReal(p.liquid.rho)*surfaceEnergyDot(physicalVectorInput(m.prescribedPressureGradient[f]),a.meanVelocity);
        if(!source.topWorkIncluded)prescribedTopPower=EnergyReal(a.area)*(surfaceEnergyDot(e.shear[f],a.topVelocity)-EnergyReal(a.pressure)*EnergyReal(a.normalVelocity));
        if(!source.bottomWorkIncluded)supportPower=EnergyReal(a.area)*(-surfaceEnergyDot(profile.bottomShear,a.baseVelocity)+EnergyReal(a.pressure)*EnergyReal(a.solidNormalVelocity));
        const EnergyReal localRadiation=!coupled&&h.film[f].mass>0?EnergyReal(a.area)*grayRadiation(a.temperature,p):0;
        e.rhs[f].enthalpy+=bodyPower+supportPower+prescribedTopPower+localRadiation;
        if((!finite(e.rhs[f].enthalpy)||!finite(a.normalVelocity)))
            return numericReject(error,"nonfinite surface source",f,numericalFailure);
    }
    return true;
}
} // namespace

bool finalizeCpuFilmPhasePacketImpl(const FilmQ& accepted,PhysicalReal acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,PhysicalVec3 bottomVelocity,
    const PhysicsConfig& p,ExchangePacket& phase,bool& exactEvent,std::string& error,bool* numericalFailure) {
    if(numericalFailure)*numericalFailure=true;
    if((!finite(dt)||dt<=0||!finite(acceptedPressure)||!finite(bottomVelocity)
        ||!finite(p.liquid.rho)||p.liquid.rho<=0||!validFilmInventory(accepted,false,p.diagnostics)
        ||phase.kind!=ExchangeKind::SolidFilm||!finite(phase.mass))) {
        error="invalid film phase input";return false;
    }
    const PhysicalReal available=(accepted.mass+PhysicalReal(dt)*localRate.mass)+primary.mass;
    if((!finite(available))){error="nonfinite film mass";return false;}
    const bool freezing=phase.mass<0&&available>=0&&-phase.mass>=available;
    const bool gasOrTransportEmpty=accepted.mass>0&&available==0&&phase.mass==0
        &&(primary.mass<0||localRate.mass<0);
    if(!freezing&&!gasOrTransportEmpty){exactEvent=false;error.clear();return true;}
    ExchangePacket candidate=phase;candidate.mass=-available;
    if((candidate.mass!=0&&(p.material.phaseCondensed<0||p.material.phaseCondensed>=Nc))) {
        error="film phase has no condensed owner species";return false;
    }
    for(int c=0;c<Nc;++c)candidate.condensed[c]=c==p.material.phaseCondensed?candidate.mass:0;
    for(int s=0;s<Ns;++s) {
        const PhysicalReal species=(accepted.species[s]+PhysicalReal(dt)*localRate.species[s])+primary.species[s];
        if((!closeEnough(species,available*physicalFraction(p,p.material.phaseFilmY[s],s),p.tolerances.absoluteMass,p.tolerances.relativeMass))) {
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,4202,species-available*physicalFraction(p,p.material.phaseFilmY[s],s),0,0,s);
        }
        candidate.species[s]=-species;
    }
    candidate.liquidKineticAdvection=-EnergyReal(.5)*EnergyReal(candidate.mass)*surfaceEnergyDot(bottomVelocity,bottomVelocity);
    EnergyReal remaining=filmInternalEnergy(accepted,acceptedPressure,p);
    remaining+=EnergyReal(dt)*localRate.enthalpy;remaining+=primary.energy;
    remaining+=primary.interfaceKineticOutflow;remaining+=primary.radiationEnergy;
    remaining+=candidate.liquidKineticAdvection;
    candidate.energy=-remaining;
    candidate.advective=candidate.energy-candidate.conductive-candidate.pressureWork-candidate.viscousWork;
    Tolerances packetTolerances=p.tolerances;packetTolerances.diagnostics=p.diagnostics;
    if((!validatePacketMath(candidate,packetTolerances))){error="invalid film phase packet";return false;}
    phase=candidate;exactEvent=true;error.clear();return true;
}

bool advanceCpuFilmCandidateImpl(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    CpuFilmCandidate& output,std::string& error,bool* numericalFailure) {
    Evaluation e;if(!evaluate(h,p,dt,forcing,drive,e,error,numericalFailure))return false;
    CpuFilmCandidate next;next.film=h.film;next.filmAux=e.aux;
    for(std::size_t f=0;f<h.film.size();++f) {
        const auto& old=h.film[f];const auto& source=forcing[f];
        const PhysicalReal withdrawn=PhysicalReal(dt)*e.withdrawal[f].mass+energyArithmetic::maximum(source.withdrawnMass,energyArithmetic::maximum(0,-source.mass)+energyArithmetic::maximum(0,-source.phaseMass));
        if((!finite(withdrawn)||withdrawn-old.mass>16*std::numeric_limits<PhysicalReal>::epsilon()*(energyArithmetic::absolute(withdrawn)+energyArithmetic::absolute(old.mass))))return numericReject(error,"aggregate mass donor exhausted",f,numericalFailure,withdrawn,old.mass);
        for(int s=0;s<Ns;++s) {
            const PhysicalReal removed=PhysicalReal(dt)*e.withdrawal[f].species[s]+energyArithmetic::maximum(source.withdrawnSpecies[s],energyArithmetic::maximum(0,-source.species[s])+energyArithmetic::maximum(0,-source.phaseSpecies[s]));
            if((!finite(removed)||removed-old.species[s]>16*std::numeric_limits<PhysicalReal>::epsilon()*(energyArithmetic::absolute(removed)+energyArithmetic::absolute(old.species[s]))))return numericReject(error,"aggregate species donor exhausted",f,numericalFailure,removed,old.species[s]);
        }
        auto& q=next.film[f];auto& a=next.filmAux[f];
        q.mass=(old.mass+PhysicalReal(dt)*e.rhs[f].mass+source.mass)+source.phaseMass;
        for(int s=0;s<Ns;++s)q.species[s]=(old.species[s]+PhysicalReal(dt)*e.rhs[f].species[s]+source.species[s])+source.phaseSpecies[s];
        const PhysicalReal massScale=energyArithmetic::absolute(old.mass)+energyArithmetic::absolute(PhysicalReal(dt)*e.rhs[f].mass)+energyArithmetic::absolute(source.mass)+energyArithmetic::absolute(source.phaseMass);
        const PhysicalReal massRoundoff=16*std::numeric_limits<PhysicalReal>::epsilon()*massScale;
        if((q.mass<0||source.exactPhaseEvent)&&energyArithmetic::absolute(q.mass)<=massRoundoff){q.mass=0;}
        for(int species=0;species<Ns;++species){const PhysicalReal scale=energyArithmetic::absolute(old.species[species])+energyArithmetic::absolute(PhysicalReal(dt)*e.rhs[f].species[species])+energyArithmetic::absolute(source.species[species])+energyArithmetic::absolute(source.phaseSpecies[species]);
            const PhysicalReal roundoff=16*std::numeric_limits<PhysicalReal>::epsilon()*scale;
            if((q.species[species]<0||q.mass==0)&&energyArithmetic::absolute(q.species[species])<=roundoff)q.species[species]=0;
        }
        const EnergyReal oldPV=EnergyReal(h.filmAux[f].pressure)*EnergyReal(old.mass)/EnergyReal(p.liquid.rho);
        const PhysicalReal newPressure=endpointPressure(drive[f]);const EnergyReal newPV=EnergyReal(newPressure)*EnergyReal(q.mass)/EnergyReal(p.liquid.rho);
        const EnergyReal before=old.enthalpy-oldPV;
        // Match assembleFilmCandidate: primary physical energy, primary K,
        // recorded radiation, then phase K and the exact phase remainder last.
        EnergyReal physical=before+EnergyReal(dt)*e.rhs[f].enthalpy;
        physical+=source.energy;physical+=source.interfaceKineticOutflow;
        physical+=source.radiationEnergy;physical+=source.phaseKineticOutflow;
        physical+=source.phaseEnergy;
        if(q.mass==0){const EnergyReal scale=energyArithmetic::absolute(before)+energyArithmetic::absolute(EnergyReal(dt)*e.rhs[f].enthalpy)+energyArithmetic::absolute(source.energy)+energyArithmetic::absolute(source.phaseEnergy)
            +energyArithmetic::absolute(source.interfaceKineticOutflow)+energyArithmetic::absolute(source.phaseKineticOutflow)+energyArithmetic::absolute(source.radiationEnergy);
            if(energyArithmetic::absolute(physical)<=EnergyReal(32)*std::numeric_limits<EnergyReal>::epsilon()*scale)physical=0;
        }
        q.enthalpy=physical+newPV;
        if(q.mass!=0)for(int species=0;species<Ns;++species){
            PhysicalReal Y=q.species[species]/q.mass;const PhysicalReal originalY=Y;
            if(!clipPhysicsMassFraction(p,Y,4204,species))return numericReject(error,"nonfinite film mass fraction",f,numericalFailure,Y,species);
            if(Y!=originalY)q.species[species]=q.mass*Y;
        }
        if((!validFilmInventory(q,false,p.diagnostics)))return numericReject(error,"invalid candidate inventory or unaligned dryout event",f,numericalFailure,q.mass,q.enthalpy);
        if(!recoverFilmThermalState(q,h.surface.area[f],newPressure,p,toPhysical(h.surface.normal[f]),coupledFace(h.surface,f),a))
            return numericReject(error,"candidate EOS or dryout event failed",f,numericalFailure,q.mass,q.enthalpy);
        a.solidFront=h.filmAux[f].solidFront+dt*Real(e.aux[f].solidNormalVelocity);
        a.gasFront=h.filmAux[f].gasFront+dt*Real(e.aux[f].normalVelocity);
        if((!finite(a.solidFront)||!finite(a.gasFront)))return numericReject(error,"nonfinite front motion",f,numericalFailure);
    }
    // Recover endpoint profiles using endpoint pressure, composition and thickness.

    HostState endpoint;endpoint.surface=h.surface;endpoint.film=next.film;endpoint.filmAux=next.filmAux;
    auto endpointDrive=drive;for(auto& d:endpointDrive)d.pressure=endpointPressure(d);
    Evaluation finalEvaluation;
    if(!recoverAndProfile(endpoint,p,endpointDrive,finalEvaluation,error,numericalFailure))return false;
    next.filmAux=std::move(finalEvaluation.aux);
    output=std::move(next);error.clear();return true;
}

bool evaluateCpuFilmTransportImpl(const HostState& h,const PhysicsConfig& p,Real dt,
    const std::vector<CpuFilmDrive>& drive,std::vector<FilmQ>& rates,
    std::string& error,
    std::vector<FilmAux>* refreshedAux,std::vector<PhysicalVec3>* pressureGradient,
    std::vector<Real>* sideMeshVolumeRate,bool* numericalFailure) {
    std::vector<CpuFilmForcing> owned(h.film.size());
    for(auto& force:owned)force.topWorkIncluded=force.bottomWorkIncluded=true;
    Evaluation e;if(!evaluate(h,p,dt,owned,drive,e,error,numericalFailure))return false;
    rates=std::move(e.rhs);
    if(refreshedAux)*refreshedAux=std::move(e.aux);
    if(pressureGradient)*pressureGradient=std::move(e.gradient);
    if(sideMeshVolumeRate)*sideMeshVolumeRate=std::move(e.sideMeshVolumeRate);
    error.clear();return true;
}

bool estimateCpuFilmStepImpl(const HostState& h,const PhysicsConfig& p,
    const std::vector<CpuFilmForcing>& forcing,const std::vector<CpuFilmDrive>& drive,
    Real forcingInterval,Real maxDt,Real& output,std::string& error,bool* numericalFailure) {
    if((!finite(maxDt)||maxDt<=0))return numericReject(error,"invalid local maximum interval",0,numericalFailure,maxDt,0);
    Evaluation e;if(!evaluate(h,p,forcingInterval,forcing,drive,e,error,numericalFailure))return false;
    Real dt=std::min<Real>(maxDt,forcingInterval);const PhysicalReal safety=PhysicalReal(.4);
    for(std::size_t f=0;f<h.film.size();++f) {
        const auto& q=h.film[f];const auto& a=e.aux[f];const auto& source=forcing[f];
        const PhysicalReal withdrawal=e.withdrawal[f].mass+energyArithmetic::maximum(source.withdrawnMass,energyArithmetic::maximum(0,-source.mass)+energyArithmetic::maximum(0,-source.phaseMass))/PhysicalReal(forcingInterval);
        if(withdrawal>0&&!source.exactPhaseEvent)dt=std::min<Real>(dt,safety*q.mass/withdrawal);
        for(int s=0;s<Ns;++s) {
            const PhysicalReal removed=e.withdrawal[f].species[s]+energyArithmetic::maximum(source.withdrawnSpecies[s],energyArithmetic::maximum(0,-source.species[s])+energyArithmetic::maximum(0,-source.phaseSpecies[s]))/PhysicalReal(forcingInterval);
            if(removed>0&&!source.exactPhaseEvent)dt=std::min<Real>(dt,safety*q.species[s]/removed);
        }
        if(e.outgoingLength[f]>0)dt=std::min<Real>(dt,safety*PhysicalReal(h.surface.area[f])/e.outgoingLength[f]);
        if(q.mass>0) {
            const EnergyReal cp=EnergyReal(p.liquid.cp0)+EnergyReal(p.liquid.cp1)*a.temperature;
            const EnergyReal capacity=EnergyReal(q.mass)*cp;
            if(e.conductance[f]>0)dt=std::min<Real>(dt,EnergyReal(safety)*capacity/e.conductance[f]);
            if(p.enableRadiation&&!coupledFace(h.surface,f)) {
                const EnergyReal radiationSlope=EnergyReal(4)*EnergyReal(PhysicalReal(5.670374419e-8))*EnergyReal(physicalFraction(p,p.emissivity))*EnergyReal(h.surface.area[f])*a.temperature*a.temperature*a.temperature;
                if(radiationSlope>0)dt=std::min<Real>(dt,EnergyReal(safety)*capacity/radiationSlope);
            }
            const EnergyReal energyRate=e.rhs[f].enthalpy+(source.energy+source.phaseEnergy+source.interfaceKineticOutflow+source.phaseKineticOutflow+source.radiationEnergy)/EnergyReal(forcingInterval);
            const PhysicalReal massRate=e.rhs[f].mass+(source.mass+source.phaseMass)/PhysicalReal(forcingInterval);
            const EnergyReal specificU=filmInternalEnergy(q,h.filmAux[f].pressure,p)/EnergyReal(q.mass);
            const EnergyReal thermalRate=energyRate-specificU*EnergyReal(massRate);
            if(thermalRate!=0&&!source.exactPhaseEvent) {
                const EnergyReal permitted=EnergyReal(PhysicalReal(.1))*a.temperature;
                dt=std::min<Real>(dt,capacity*permitted/energyArithmetic::absolute(thermalRate));
            }
        }
    }
    if((!finite(dt)||dt<0))return numericReject(error,"invalid local source/donor/thermal estimate",0,numericalFailure,dt,maxDt);
    output=dt;error.clear();return true;
}

bool finalizeCpuFilmPhasePacket(const FilmQ& q,PhysicalReal pressure,const FilmQ& rate,const CpuFilmForcing& f,Real dt,PhysicalVec3 velocity,const PhysicsConfig& p,ExchangePacket& phase,bool& exact,std::string& error,bool* numerical){return finalizeCpuFilmPhasePacketImpl(q,pressure,rate,f,dt,velocity,p,phase,exact,error,numerical);}
bool advanceCpuFilmCandidate(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,CpuFilmCandidate& out,std::string& error,bool* numerical){return advanceCpuFilmCandidateImpl(h,p,dt,f,d,out,error,numerical);}
bool evaluateCpuFilmTransport(const HostState& h,const PhysicsConfig& p,Real dt,const std::vector<CpuFilmDrive>& d,std::vector<FilmQ>& rates,std::string& error,std::vector<FilmAux>* aux,std::vector<PhysicalVec3>* gradient,std::vector<Real>* side,bool* numerical){return evaluateCpuFilmTransportImpl(h,p,dt,d,rates,error,aux,gradient,side,numerical);}
bool estimateCpuFilmStep(const HostState& h,const PhysicsConfig& p,const std::vector<CpuFilmForcing>& f,const std::vector<CpuFilmDrive>& d,Real interval,Real maximum,Real& out,std::string& error,bool* numerical){return estimateCpuFilmStepImpl(h,p,f,d,interval,maximum,out,error,numerical);}
} // namespace chmt
