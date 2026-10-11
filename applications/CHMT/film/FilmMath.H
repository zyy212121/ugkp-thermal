#ifndef CHMT_FILM_FILMMATH_H
#define CHMT_FILM_FILMMATH_H
#include "materials/Reaction.H"
#include "coupling/ExchangeLedger.H"

namespace chmt {
struct FilmProfile {
    Vec3 meanVelocity{}, topVelocity{}, bottomShear{}, baseVelocity{}, linear{}, quadratic{};
    Real delta=0, dissipation=0, kineticPerArea=0;
    // kineticPerArea is the UNIT-DENSITY integral 1/2 int |u|^2 dz [m^3/s^2].
    // Multiply by actual liquid rho for J/m^2; the frozen profile API has no rho.
};
CHMT_HD inline Vec3 profileVelocity(const FilmProfile& p, Real z) {
    return p.baseVelocity+p.linear*z+p.quadratic*(z*z);
}
CHMT_HD inline FilmProfile filmProfile(Real delta, Real mu, Vec3 ub, Vec3 tau, Vec3 G) {
    FilmProfile p;
    p.delta=delta; p.baseVelocity=ub;
    if (!finite(delta) || delta<0 || !finite(mu) || mu<=0
        || !finite(ub) || !finite(tau) || !finite(G)) {
        p.dissipation=p.kineticPerArea=undefinedValue(); return p;
    }
    p.linear=(tau-G*delta)/mu;
    p.quadratic=G/(2*mu);
    p.meanVelocity=ub+tau*(delta/(2*mu))-G*(delta*delta/(3*mu));
    p.topVelocity=profileVelocity(p,delta);
    p.bottomShear=tau-G*delta;
    const Vec3 centred=tau-G*(delta/2);
    p.dissipation=delta/mu*dot(centred,centred)+delta*delta*delta/(12*mu)*dot(G,G);
    const Real d2=delta*delta, d3=d2*delta, d4=d3*delta, d5=d4*delta;
    p.kineticPerArea=0.5*(dot(ub,ub)*delta+dot(ub,p.linear)*d2
        +(dot(p.linear,p.linear)+2*dot(ub,p.quadratic))*d3/3
        +dot(p.linear,p.quadratic)*d4/2+dot(p.quadratic,p.quadratic)*d5/5);
    return p;
}
CHMT_HD inline FilmProfile withNormalProfile(FilmProfile p, Vec3 normal, Real bottom, Real top) {
    if (p.delta<=0) {
        const Vec3 tangent=p.baseVelocity-normal*dot(p.baseVelocity,normal);
        p.baseVelocity=tangent+normal*bottom;
        p.meanVelocity=tangent+normal*(0.5*(bottom+top));
        p.topVelocity=tangent+normal*top;
        return p;
    }
    p.baseVelocity+=normal*(bottom-dot(p.baseVelocity,normal));
    p.linear+=normal*((top-bottom)/p.delta-dot(p.linear,normal));
    p.quadratic-=normal*dot(p.quadratic,normal);
    p.meanVelocity=p.baseVelocity+p.linear*(p.delta/2)+p.quadratic*(p.delta*p.delta/3);
    p.topVelocity=profileVelocity(p,p.delta);
    const Real d=p.delta;
    p.kineticPerArea=0.5*(dot(p.baseVelocity,p.baseVelocity)*d
        +dot(p.baseVelocity,p.linear)*d*d
        +(dot(p.linear,p.linear)+2*dot(p.baseVelocity,p.quadratic))*d*d*d/3
        +dot(p.linear,p.quadratic)*d*d*d*d/2+dot(p.quadratic,p.quadratic)*d*d*d*d*d/5);
    return p;
}
// The interface owner supplies a reference-frame normal trace independently
// of film mass. A dry owner explicitly clears it; no old wet trace is inferred.
CHMT_HD inline bool setCoupledFilmNormalTrace(FilmAux& aux, Vec3 normal, bool wet,
    Real bottom, Real top, DeviceStatus* status=nullptr, int index=-1) {
    if (!finite(normal) || !closeEnough(mag(normal),1,1e-12,1e-12) || !finite(aux.baseVelocity)) {
        return failStatus(status,ErrorCode::InvalidInput,index,0,ErrorLocation::Face);
    }
    FilmAux next=aux;
    if (wet) {
        if (!finite(bottom) || !finite(top) || !finite(aux.meanVelocity) || !finite(aux.topVelocity)) {
            return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
        }
        next.meanVelocity+=normal*(0.5*(bottom+top)-dot(aux.meanVelocity,normal));
        next.topVelocity+=normal*(top-dot(aux.topVelocity,normal));
        if (!finite(next.meanVelocity) || !finite(next.topVelocity)) {
            return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
        }
    } else {
        next.meanVelocity=next.baseVelocity;
        next.topVelocity=next.baseVelocity;
        next.kineticEnergy=0;
    }
    aux=next; return true;
}
// Thermal recovery does not own a coupled interface's normal kinematics. The
// plain recoverFilm empty-film convention remains unchanged for standalone EOS
// callers. Only a finite, explicitly coupled trace is restored here.
CHMT_HD inline bool recoverFilmThermalState(const FilmQ& q, Real area, Real pressure,
    const PhysicsConfig& p, Vec3 normal, bool coupled, FilmAux& aux,
    DeviceStatus* status=nullptr, int index=-1) {
    if (!finite(normal) || !closeEnough(mag(normal),1,1e-12,1e-12) || !finite(aux.baseVelocity)) {
        return failStatus(status,ErrorCode::InvalidInput,index,0,ErrorLocation::Face);
    }
    Real meanNormal=0, topNormal=0;
    if (coupled) {
        if (!finite(aux.meanVelocity) || !finite(aux.topVelocity)) {
            return failStatus(status,ErrorCode::Nonfinite,index,0,ErrorLocation::Face);
        }
        meanNormal=dot(aux.meanVelocity,normal);
        topNormal=dot(aux.topVelocity,normal);
        if (!finite(meanNormal) || !finite(topNormal)) {
            return failStatus(status,ErrorCode::Nonfinite,index,topNormal,ErrorLocation::Face);
        }
    }
    FilmAux next=aux;
    next.normal=normal;
    if (!recoverFilm(q,area,pressure,p,next,status,index)) return false;
    if (coupled) {
        next.meanVelocity+=normal*(meanNormal-dot(next.meanVelocity,normal));
        next.topVelocity+=normal*(topNormal-dot(next.topVelocity,normal));
        if (!finite(next.meanVelocity) || !finite(next.topVelocity)) {
            return failStatus(status,ErrorCode::Nonfinite,index,topNormal,ErrorLocation::Face);
        }
    }
    aux=next; return true;
}
// Shared by the actual CUDA profile kernel and the host sequence regression.
CHMT_HD inline bool updateFilmProfileState(FilmAux& aux, const PhysicsConfig& p, Vec3 normal,
    Vec3 shear, Vec3 pressureBodyGradient, bool coupled, DeviceStatus* status=nullptr, int index=-1) {
    if (!finite(normal) || !closeEnough(mag(normal),1,1e-12,1e-12)
        || !finite(aux.area) || aux.area<=0 || !finite(p.liquid.rho) || p.liquid.rho<=0) {
        return failStatus(status,ErrorCode::InvalidInput,index,aux.area,ErrorLocation::Face);
    }
    Real bottom=dot(aux.baseVelocity,normal), top=bottom;
    if (coupled) {
        if (!finite(aux.meanVelocity) || !finite(aux.topVelocity)) {
            return failStatus(status,ErrorCode::Nonfinite,index,0,ErrorLocation::Face);
        }
        top=dot(aux.topVelocity,normal);
        bottom=2*dot(aux.meanVelocity,normal)-top;
    }
    if (!finite(bottom) || !finite(top)) {
        return failStatus(status,ErrorCode::Nonfinite,index,top,ErrorLocation::Face);
    }
    FilmProfile profile=filmProfile(aux.thickness,p.liquidViscosity,aux.baseVelocity,shear,pressureBodyGradient);
    profile=withNormalProfile(profile,normal,bottom,top);
    const Real kinetic=p.liquid.rho*aux.area*profile.kineticPerArea;
    if (!finite(profile.meanVelocity) || !finite(profile.topVelocity) || !finite(profile.dissipation)
        || !finite(kinetic) || kinetic<0) {
        return failStatus(status,ErrorCode::Nonfinite,index,kinetic,ErrorLocation::Face);
    }
    FilmAux next=aux;
    next.meanVelocity=profile.meanVelocity;
    next.topVelocity=profile.topVelocity;
    next.kineticEnergy=kinetic;
    aux=next; return true;
}
CHMT_HD inline Real profileKineticFlux(const FilmProfile& p, Vec3 gridVelocity, Vec3 conormal) {
    // Four-point Gauss quadrature exactly integrates the degree-six polynomial.
    const Real x[4]={-0.8611363115940525752,-0.3399810435848562648,
        0.3399810435848562648,0.8611363115940525752};
    const Real w[4]={0.3478548451374538574,0.6521451548625461426,
        0.6521451548625461426,0.3478548451374538574};
    Real result=0;
    for (int i=0; i<4; ++i) {
        const Vec3 u=profileVelocity(p,0.5*p.delta*(1+x[i]));
        result+=w[i]*0.5*dot(u,u)*dot(u-gridVelocity,conormal)*0.5*p.delta;
    }
    return result;
}
CHMT_HD inline Real pressureVolumeProduct(Real p0, Real V0, Real p1, Real V1) {
    return 0.5*(p0+p1)*(V1-V0)+0.5*(V0+V1)*(p1-p0);
}
CHMT_HD inline Real physicalInterfaceEnergy(Real J, Real speciesEnthalpyFlux,
    Vec3 velocity, Real pressure, Real normalSpeed, Real viscousPower, Real conductiveFlux) {
    return speciesEnthalpyFlux+0.5*J*dot(velocity,velocity)
        +pressure*normalSpeed-viscousPower+conductiveFlux;
}
CHMT_HD inline Real liquidInterfaceHeat(Real J, Real h, Vec3 velocity, Real pressure,
    Real normalSpeed, Real viscousPower, Real gasEnergyFlux) {
    return physicalInterfaceEnergy(J,J*h,velocity,pressure,normalSpeed,viscousPower,0)-gasEnergyFlux;
}
CHMT_HD inline Real solidPhaseEnergy(Real J, Real e, Real pressure, Real normalSpeed,
    Real rho, Real conductiveFlux) {
    return J*(e+pressure/rho)+pressure*normalSpeed+conductiveFlux;
}
CHMT_HD inline FilmQ filmAdvectiveFlux(const FilmQ& density, Real sweptLengthRate) {
    FilmQ flux;
    flux.mass=density.mass*sweptLengthRate;
    flux.enthalpy=density.enthalpy*sweptLengthRate;
    for (int s=0; s<Ns; ++s) flux.species[s]=density.species[s]*sweptLengthRate;
    return flux;
}
CHMT_HD inline Real grayRadiation(Real temperature, const PhysicsConfig& p) {
    if (!p.enableRadiation) return 0;
    const Real t2=temperature*temperature, a2=p.ambientTemperature*p.ambientTemperature;
    return 5.670374419e-8*p.emissivity*(a2*a2-t2*t2);
}
CHMT_HD inline bool freezeFilmRemainder(const FilmQ& q, Real pressure,
    const PhysicsConfig& p, ExchangePacket& out) {
    const int c=p.material.phaseCondensed;
    if (c<0 || c>=Nc || !validFilmInventory(q) || q.mass<=0) return false;
    for (int s=0; s<Ns; ++s) {
        if (!closeEnough(q.species[s],q.mass*p.material.phaseFilmY[s],
            p.tolerances.absoluteMass,p.tolerances.relativeMass)) return false;
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
