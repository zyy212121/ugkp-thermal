#ifndef CHMT_FILM_FILMMATH_H
#define CHMT_FILM_FILMMATH_H
#include "materials/Reaction.H"
#include "coupling/ExchangeLedger.H"

namespace chmt {
// Explicit value boundaries, never implicit physical/geometry operators.
CHMT_HD inline PhysicalVec3 physicalVectorInput(Vec3 value){return toPhysical(value);}
CHMT_HD inline PhysicalVec3 physicalVectorInput(PhysicalVec3 value){return value;}

// Physical energy products round their operands before arithmetic. PhysicalVec3 remains
// the independent host geometry/kinematics transport type.
CHMT_HD inline EnergyReal surfaceEnergyDot(PhysicalVec3 a,PhysicalVec3 b) {
    return EnergyReal(a.x)*EnergyReal(b.x)+EnergyReal(a.y)*EnergyReal(b.y)+EnergyReal(a.z)*EnergyReal(b.z);
}

struct FilmProfile {
    PhysicalVec3 meanVelocity{}, topVelocity{}, bottomShear{}, baseVelocity{}, linear{}, quadratic{};
    PhysicalReal delta=0;
};
CHMT_HD inline PhysicalVec3 profileVelocity(const FilmProfile& p, PhysicalReal z) {
    return p.baseVelocity+p.linear*z+p.quadratic*(z*z);
}
template<bool Terminal=false>
CHMT_HD inline FilmProfile filmProfile(PhysicalReal delta, PhysicalReal mu, PhysicalVec3 ub, PhysicalVec3 tau, PhysicalVec3 G) {
    FilmProfile p;
    p.delta=delta; p.baseVelocity=ub;
    if ((!finite(delta) || delta<0 || !finite(mu) || mu<=0
        || !finite(ub) || !finite(tau) || !finite(G))) {
        p.meanVelocity=p.topVelocity=p.bottomShear=p.baseVelocity=p.linear=p.quadratic=
            {undefinedValue(),undefinedValue(),undefinedValue()};
        return p;
    }
    p.linear=(tau-G*delta)/mu;
    p.quadratic=G/(2*mu);
    p.meanVelocity=ub+tau*(delta/(2*mu))-G*(delta*delta/(3*mu));
    p.topVelocity=profileVelocity(p,delta);
    p.bottomShear=tau-G*delta;
    return p;
}
CHMT_HD inline FilmProfile withNormalProfile(FilmProfile p, PhysicalVec3 normal, PhysicalReal bottom, PhysicalReal top) {
    if (p.delta<=0) {
        const PhysicalVec3 tangent=p.baseVelocity-normal*dot(p.baseVelocity,normal);
        p.baseVelocity=tangent+normal*bottom;
        p.meanVelocity=tangent+normal*(PhysicalReal(0.5)*(bottom+top));
        p.topVelocity=tangent+normal*top;
        return p;
    }
    p.baseVelocity+=normal*(bottom-dot(p.baseVelocity,normal));
    p.linear+=normal*((top-bottom)/p.delta-dot(p.linear,normal));
    p.quadratic-=normal*dot(p.quadratic,normal);
    p.meanVelocity=p.baseVelocity+p.linear*(p.delta/2)+p.quadratic*(p.delta*p.delta/3);
    p.topVelocity=profileVelocity(p,p.delta);
    return p;
}
// The interface owner supplies a reference-frame normal trace independently
// of film mass. A dry owner explicitly clears it; no old wet trace is inferred.
template<bool Terminal=false>
CHMT_HD inline bool setCoupledFilmNormalTrace(FilmAux& aux, PhysicalVec3 normal, bool wet,
    PhysicalReal bottom, PhysicalReal top, DeviceStatus* status=nullptr, int index=-1,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if ((!finite(normal) || mag(normal)==0 || !finite(aux.baseVelocity))) {
        return failStatus(status,ErrorCode::InvalidInput,index,0,ErrorLocation::Face);
    }
    if(!closeEnough(mag(normal),1,PhysicalReal(1e-12),PhysicalReal(1e-12))){using namespace ugkp::diagnostics;
        recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,4217,42,0,EntityKind::Face,index,-1,mag(normal)-1,PhysicalReal(1e-12),0,Quantity::Accuracy));}
    FilmAux next=aux;
    if (wet) {
        if ((!finite(bottom) || !finite(top) || !finite(aux.meanVelocity) || !finite(aux.topVelocity))) {
            return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
        }
        next.meanVelocity+=normal*(PhysicalReal(0.5)*(bottom+top)-dot(aux.meanVelocity,normal));
        next.topVelocity+=normal*(top-dot(aux.topVelocity,normal));
        if ((!finite(next.meanVelocity) || !finite(next.topVelocity))) {
            return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
        }
    } else {
        next.meanVelocity=next.baseVelocity;
        next.topVelocity=next.baseVelocity;
    }
    aux=next; return true;
}
// Thermal recovery does not own a coupled interface's normal kinematics. The
// plain recoverFilm empty-film convention remains unchanged for standalone EOS
// callers. Only a finite, explicitly coupled trace is restored here.
template<bool Terminal=false>
CHMT_HD inline bool recoverFilmThermalState(const FilmQ& q, PhysicalReal area, PhysicalReal pressure,
    const PhysicsConfig& p, PhysicalVec3 normal, bool coupled, FilmAux& aux,
    DeviceStatus* status=nullptr, int index=-1) {
    if ((!finite(normal) || mag(normal)==0 || !finite(aux.baseVelocity))) {
        return failStatus(status,ErrorCode::InvalidInput,index,0,ErrorLocation::Face);
    }
    if(!closeEnough(mag(normal),1,PhysicalReal(1e-12),PhysicalReal(1e-12)))reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,4218,mag(normal)-1,PhysicalReal(1e-12));
    PhysicalReal meanNormal=0, topNormal=0;
    if (coupled) {
        if ((!finite(aux.meanVelocity) || !finite(aux.topVelocity))) {
            return failStatus(status,ErrorCode::Nonfinite,index,0,ErrorLocation::Face);
        }
        meanNormal=dot(aux.meanVelocity,normal);
        topNormal=dot(aux.topVelocity,normal);
        if ((!finite(meanNormal) || !finite(topNormal))) {
            return failStatus(status,ErrorCode::Nonfinite,index,topNormal,ErrorLocation::Face);
        }
    }
    FilmAux next=aux;
    next.normal=normal;
    if (!recoverFilm<Terminal>(q,area,pressure,p,next,status,index)) return false;
    if (coupled) {
        next.meanVelocity+=normal*(meanNormal-dot(next.meanVelocity,normal));
        next.topVelocity+=normal*(topNormal-dot(next.topVelocity,normal));
        if ((!finite(next.meanVelocity) || !finite(next.topVelocity))) {
            return failStatus(status,ErrorCode::Nonfinite,index,topNormal,ErrorLocation::Face);
        }
    }
    aux=next; return true;
}
// Shared by the actual CUDA profile kernel and the host sequence regression.
template<bool Terminal=false>
CHMT_HD inline bool updateFilmProfileState(FilmAux& aux, const PhysicsConfig& p, PhysicalVec3 normal,
    PhysicalVec3 shear, PhysicalVec3 pressureBodyGradient, bool coupled, DeviceStatus* status=nullptr, int index=-1) {
    if ((!finite(normal) || mag(normal)==0)) {
        return failStatus(status,ErrorCode::InvalidInput,index,aux.area,ErrorLocation::Face);
    }
    if(!closeEnough(mag(normal),1,PhysicalReal(1e-12),PhysicalReal(1e-12)))reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,4219,mag(normal)-1,PhysicalReal(1e-12));
    PhysicalReal bottom=dot(aux.baseVelocity,normal), top=bottom;
    if (coupled) {
        if ((!finite(aux.meanVelocity) || !finite(aux.topVelocity))) {
            return failStatus(status,ErrorCode::Nonfinite,index,0,ErrorLocation::Face);
        }
        top=dot(aux.topVelocity,normal);
        bottom=2*dot(aux.meanVelocity,normal)-top;
    }
    if ((!finite(bottom) || !finite(top))) {
        return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
    }
    FilmProfile profile=filmProfile<Terminal>(aux.thickness,p.liquidViscosity,aux.baseVelocity,shear,pressureBodyGradient);
    profile=withNormalProfile(profile,normal,bottom,top);
    if ((!finite(profile.meanVelocity) || !finite(profile.topVelocity))) {
        return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
    }
    FilmAux next=aux;
    next.meanVelocity=profile.meanVelocity;
    next.topVelocity=profile.topVelocity;
    aux=next; return true;
}
CHMT_HD inline EnergyReal profileKineticFlux(const FilmProfile& p, PhysicalVec3 gridVelocity, PhysicalVec3 conormal) {
    // The same four-point Gauss rule, evaluated entirely at energy precision.
    const EnergyReal x[4]={EnergyReal(-PhysicalReal(0.8611363115940525752)),EnergyReal(-PhysicalReal(0.3399810435848562648)),
        EnergyReal(PhysicalReal(0.3399810435848562648)),EnergyReal(PhysicalReal(0.8611363115940525752))};
    const EnergyReal w[4]={EnergyReal(PhysicalReal(0.3478548451374538574)),EnergyReal(PhysicalReal(0.6521451548625461426)),
        EnergyReal(PhysicalReal(0.6521451548625461426)),EnergyReal(PhysicalReal(0.3478548451374538574))};
    EnergyReal result=0;
    for (int i=0; i<4; ++i) {
        const EnergyReal z=EnergyReal(.5)*EnergyReal(p.delta)*(EnergyReal(1)+x[i]),z2=z*z;
        const PhysicalVec3 u{EnergyReal(p.baseVelocity.x)+EnergyReal(p.linear.x)*z+EnergyReal(p.quadratic.x)*z2,
            EnergyReal(p.baseVelocity.y)+EnergyReal(p.linear.y)*z+EnergyReal(p.quadratic.y)*z2,
            EnergyReal(p.baseVelocity.z)+EnergyReal(p.linear.z)*z+EnergyReal(p.quadratic.z)*z2};
        const PhysicalVec3 relative{EnergyReal(u.x)-EnergyReal(gridVelocity.x),EnergyReal(u.y)-EnergyReal(gridVelocity.y),EnergyReal(u.z)-EnergyReal(gridVelocity.z)};
        result+=w[i]*EnergyReal(.5)*surfaceEnergyDot(u,u)*surfaceEnergyDot(relative,conormal)*EnergyReal(.5)*EnergyReal(p.delta);
    }
    return result;
}
CHMT_HD inline EnergyReal pressureVolumeProduct(EnergyReal p0, EnergyReal V0, EnergyReal p1, EnergyReal V1) {
    return EnergyReal(.5)*(p0+p1)*(V1-V0)+EnergyReal(.5)*(V0+V1)*(p1-p0);
}
CHMT_HD inline EnergyReal physicalInterfaceEnergy(EnergyReal J, EnergyReal speciesEnthalpyFlux,
    PhysicalVec3 velocity, EnergyReal pressure, EnergyReal normalSpeed, EnergyReal viscousPower, EnergyReal conductiveFlux) {
    return speciesEnthalpyFlux+EnergyReal(.5)*J*surfaceEnergyDot(velocity,velocity)
        +pressure*normalSpeed-viscousPower+conductiveFlux;
}
CHMT_HD inline EnergyReal liquidInterfaceHeat(EnergyReal J, EnergyReal h, PhysicalVec3 velocity, EnergyReal pressure,
    EnergyReal normalSpeed, EnergyReal viscousPower, EnergyReal gasEnergyFlux) {
    return physicalInterfaceEnergy(J,J*h,velocity,pressure,normalSpeed,viscousPower,0)-gasEnergyFlux;
}
CHMT_HD inline EnergyReal solidPhaseEnergy(EnergyReal J, EnergyReal e, EnergyReal pressure, EnergyReal normalSpeed,
    EnergyReal rho, EnergyReal conductiveFlux) {
    return J*(e+pressure/rho)+pressure*normalSpeed+conductiveFlux;
}
CHMT_HD inline FilmQ filmAdvectiveFlux(const FilmQ& density, PhysicalReal sweptLengthRate) {
    FilmQ flux;
    flux.mass=density.mass*sweptLengthRate;
    flux.enthalpy=density.enthalpy*EnergyReal(sweptLengthRate);
    for (int s=0; s<Ns; ++s) flux.species[s]=density.species[s]*sweptLengthRate;
    return flux;
}
CHMT_HD inline EnergyReal grayRadiation(EnergyReal temperature, const PhysicsConfig& p) {
    if (!p.enableRadiation) return 0;
    const EnergyReal t2=temperature*temperature, ambient=EnergyReal(p.ambientTemperature), a2=ambient*ambient;
    return EnergyReal(PhysicalReal(5.670374419e-8))*EnergyReal(physicalFraction(p,p.emissivity))*(a2*a2-t2*t2);
}
CHMT_HD inline bool freezeFilmRemainder(const FilmQ& q, PhysicalReal pressure,
    const PhysicsConfig& p, ExchangePacket& out) {
    const int c=p.material.phaseCondensed;
    if (c<0 || c>=Nc || !validFilmInventory(q) || q.mass<=0) return false;
    for (int s=0; s<Ns; ++s) {
        const PhysicalReal expected=q.mass*physicalFraction(p,p.material.phaseFilmY[s],s);
        if(!finite(expected))return false;
        if (!closeEnough(q.species[s],expected,
            p.tolerances.absoluteMass,p.tolerances.relativeMass))
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,4213,q.species[s]-q.mass*physicalFraction(p,p.material.phaseFilmY[s],s),0,0,s);
    }
    ExchangePacket packet;
    packet.kind=ExchangeKind::SolidFilm;
    packet.mass=-q.mass;
    packet.condensed[c]=-q.mass;
    for (int s=0; s<Ns; ++s) packet.species[s]=-q.species[s];
    packet.energy=-filmInternalEnergy(q,pressure,p);
    packet.advective=packet.energy;
    out=packet; return true;
}
} // namespace chmt
#endif
