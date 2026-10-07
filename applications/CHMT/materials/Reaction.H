#ifndef CHMT_MATERIALS_REACTION_H
#define CHMT_MATERIALS_REACTION_H
#include "materials/Thermodynamics.H"

namespace chmt {
struct MaterialPrimitive {
    Real temperature=0, pressure=0, porosity=0, poreDensity=0, conductivity=0;
};

CHMT_HD inline SolidQ addSolid(const SolidQ& a, const SolidQ& b, Real scale=1) {
    SolidQ out=a;
    for (int c=0; c<Nc; ++c) out.condensed[c]+=scale*b.condensed[c];
    for (int s=0; s<Ns; ++s) out.pore[s]+=scale*b.pore[s];
    for (int r=0; r<Nr; ++r) out.progress[r]+=scale*b.progress[r];
    out.energy+=scale*b.energy;
    return out;
}

CHMT_HD inline bool recoverMaterial(const SolidQ& q, Real volume, const PhysicsConfig& p,
    MaterialPrimitive& out, DeviceStatus* status=nullptr, int index=-1) {
    if (!finite(volume) || volume<=0) {
        return failStatus(status,ErrorCode::InvalidInput,index,volume,ErrorLocation::Cell);
    }
    Real occupied=0, poreMass=0, poreR=0, conductivity=0;
    for (int c=0; c<Nc; ++c) {
        if (!finite(q.condensed[c]) || q.condensed[c]<0) {
            return failStatus(status,ErrorCode::Inventory,index,q.condensed[c],ErrorLocation::Cell);
        }
        if (q.condensed[c]==0) continue;
        if (!validCondensed(p.condensed[c])) {
            return failStatus(status,ErrorCode::PropertyRange,index,c,ErrorLocation::Cell);
        }
        const Real constituentVolume=q.condensed[c]/p.condensed[c].rho;
        occupied+=constituentVolume;
        conductivity+=constituentVolume*p.condensed[c].conductivity/volume;
    }
    Real phi=1-occupied/volume;
    if (absValue(occupied-volume)<=32*std::numeric_limits<Real>::epsilon()*maxValue(occupied,volume)) {
        phi=0; // Derived geometry roundoff only; no mass or energy is altered.
    }
    if (!(occupied>0) || !finite(phi) || phi<p.material.minimumPorosity
        || phi>=p.material.maximumPorosity || phi>=1) {
        return failStatus(status,ErrorCode::Inventory,index,phi,ErrorLocation::Cell);
    }
    SolidQ thermal=q;
    thermal.porosity=phi;
    MaterialPrimitive next;
    if (!recoverSolid(thermal,p,next.temperature,status,index)) return false;
    for (int s=0; s<Ns; ++s) {
        poreMass+=q.pore[s];
        poreR+=q.pore[s]*p.species[s].R;
    }
    if ((poreMass>0 && phi<=0) || !finite(poreMass) || !finite(poreR)) {
        return failStatus(status,ErrorCode::Inventory,index,poreMass,ErrorLocation::Cell);
    }
    next.porosity=phi;
    next.poreDensity=phi>0 ? poreMass/(phi*volume) : 0;
    next.pressure=phi>0 ? poreR*next.temperature/(phi*volume) : 0;
    next.conductivity=conductivity+(poreMass>0 ? phi*p.gasConductivity : 0);
    if (!finite(next.pressure) || !finite(next.conductivity) || next.conductivity<0) {
        return failStatus(status,ErrorCode::Nonfinite,index,next.pressure,ErrorLocation::Cell);
    }
    out=next;
    return true;
}

// All reactions evaluate the same stage state. Bound the NET assembled species
// loss with one common extent factor. Gross-consumption limiting would freeze
// a midpoint-generated intermediate absent from the accepted base, even when
// simultaneous production leaves its final inventory positive. Shared-donor
// losses still add before applying the one conservative mechanism-wide factor.
CHMT_HD inline bool boundedReactionRates(const SolidQ& evaluation, const SolidQ& capacity,
    Real volume, const PhysicsConfig& p, Real dt, Real* rates,
    DeviceStatus* status=nullptr, int index=-1) {
    if (!finite(dt) || dt<=0 || !rates) return false;
    for (int r=0; r<Nr; ++r) rates[r]=0;
    if (!p.enableReactions) return true;
    Real T=0;
    if (!recoverSolid(evaluation,p,T,status,index)) return false;
    SolidQ consumption;
    for (int r=0; r<p.nReactions; ++r) {
        rates[r]=reactionRate(p.reactions[r],evaluation.condensed,volume,T);
        if (!finite(rates[r]) || rates[r]<0) return false;
        for (int c=0; c<Nc; ++c) {
            consumption.condensed[c]-=rates[r]*p.reactions[r].condensedNu[c];
        }
        for (int species=0; species<Ns; ++species) {
            consumption.pore[species]-=rates[r]*p.reactions[r].gasNu[species];
        }
    }
    Real factor=1;
    for (int c=0; c<Nc; ++c) if (consumption.condensed[c]>0) {
        factor=minValue(factor,capacity.condensed[c]/(dt*consumption.condensed[c]));
    }
    for (int species=0; species<Ns; ++species) if (consumption.pore[species]>0) {
        factor=minValue(factor,capacity.pore[species]/(dt*consumption.pore[species]));
    }
    if (!finite(factor) || factor<0) return false;
    for (int r=0; r<p.nReactions; ++r) rates[r]*=factor;
    return true;
}
CHMT_HD inline bool advanceLocalReactions(const SolidQ& base, Real volume,
    const PhysicsConfig& p, Real dt, SolidQ& out, Real* rates,
    DeviceStatus* status=nullptr, int index=-1) {
    Real localRates[Nr]{};
    if (!boundedReactionRates(base,base,volume,p,dt,localRates,status,index)) return false;
    SolidQ next=base;
    for (int r=0; r<p.nReactions; ++r) {
        const Real extent=dt*localRates[r];
        for (int c=0; c<Nc; ++c) next.condensed[c]+=extent*p.reactions[r].condensedNu[c];
        for (int species=0; species<Ns; ++species) next.pore[species]+=extent*p.reactions[r].gasNu[species];
        next.progress[r]+=extent;
    }
    for (int r=0; r<Nr; ++r) if (rates) rates[r]=localRates[r];
    MaterialPrimitive state;
    if (!recoverMaterial(next,volume,p,state,status,index)) return false;
    next.porosity=state.porosity;
    out=next; return true;
}

// Stationary skeleton ALE flux uses INTERNAL energy, not pore enthalpy. Darcy's
// physical pore flow below separately uses enthalpy. Interface faces replace
// this boundary remap by the same accepted moving-interface packet.
CHMT_HD inline SolidQ materialAleFlux(const SolidQ& q, Real volume, Real meshVolumeRate) {
    SolidQ flux;
    const Real factor=-meshVolumeRate/volume;
    for (int c=0; c<Nc; ++c) flux.condensed[c]=factor*q.condensed[c];
    for (int s=0; s<Ns; ++s) flux.pore[s]=factor*q.pore[s];
    for (int r=0; r<Nr; ++r) flux.progress[r]=factor*q.progress[r];
    flux.energy=factor*q.energy;
    return flux;
}

CHMT_HD inline Real seriesConductance(Real k0, Real k1, Real d0, Real d1) {
    if (k0<=0 || k1<=0) return 0;
    return 1/(d0/k0+d1/k1);
}

CHMT_HD inline bool darcyConductionFluxWithGradient(const SolidQ& left, const SolidQ& right,
    const MaterialPrimitive& a, const MaterialPrimitive& b, Real volumeA, Real volumeB,
    Real area, Real distance, Vec3 normal, Real normalPressureGradient,
    const PhysicsConfig& p, SolidQ& out) {
    if (!(volumeA>0 && volumeB>0 && area>0 && distance>0)
        || !finite(normal) || !finite(normalPressureGradient)) return false;
    SolidQ flux;
    Real volumeRate=0;
    if (p.permeability>0) {
        if (!(p.poreViscosity>0)) return false;
        const Real rho=0.5*(a.poreDensity+b.poreDensity);
        volumeRate=-area*p.permeability/p.poreViscosity
            *(normalPressureGradient-rho*dot(p.gravity,normal));
    }
    if (!finite(volumeRate)) return false;
    const SolidQ& donor=volumeRate>=0 ? left : right;
    const MaterialPrimitive& state=volumeRate>=0 ? a : b;
    const Real volume=volumeRate>=0 ? volumeA : volumeB;
    if (volumeRate!=0 && state.porosity<=0) return false;
    for (int s=0; s<Ns; ++s) {
        flux.pore[s]=state.porosity>0 ? volumeRate*donor.pore[s]/(state.porosity*volume) : 0;
        flux.energy+=flux.pore[s]*speciesH(p.species[s],state.temperature);
    }
    flux.energy+=area*seriesConductance(a.conductivity,b.conductivity,distance/2,distance/2)
        *(a.temperature-b.temperature);
    if (!finite(flux.energy)) return false;
    out=flux;
    return true;
}

// Orthogonal/two-point convenience entry. General mesh operators must use
// darcyConductionFluxWithGradient with a corrected original-cell derivative.
CHMT_HD inline bool darcyConductionFlux(const SolidQ& left, const SolidQ& right,
    const MaterialPrimitive& a, const MaterialPrimitive& b, Real volumeA, Real volumeB,
    Real area, Real distance, Vec3 normal, const PhysicsConfig& p, SolidQ& out) {
    if (!(distance>0)) return false;
    return darcyConductionFluxWithGradient(left,right,a,b,volumeA,volumeB,area,
        distance,normal,(b.pressure-a.pressure)/distance,p,out);
}

inline bool validateMaterialPolicy(const PhysicsConfig& p, std::string& error) {
    const MaterialPolicyConfig& m=p.material;
    if (!finite(m.minimumPorosity) || !finite(m.maximumPorosity) || m.minimumPorosity<0
        || m.minimumPorosity>=m.maximumPorosity || m.maximumPorosity>1) {
        error="material porosity bounds must satisfy 0 <= minimum < maximum <= 1";
        return false;
    }
    if (!finite(m.gasContactResistance) || !finite(m.solidContactResistance)
        || m.gasContactResistance<0 || m.solidContactResistance<0) {
        error="material contact resistances must be finite nonnegative m^2 K/W";
        return false;
    }
    if (m.enableMelting && !p.enableFilm) {
        error="moving solid/film phase policy requires film enabled"; return false;
    }
    if (m.enableMelting || p.filmThermalMode==FilmThermalMode::ResolvedNormal) {
        if (m.phaseCondensed<0 || m.phaseCondensed>=Nc || !validCondensed(p.liquid)
            || !validCondensed(p.condensed[m.phaseCondensed])) {
            error="phase policy requires an explicit constituent and liquid table";
            return false;
        }
        Real sum=0, elements[Ne]{};
        for (int s=0; s<Ns; ++s) {
            if (!finite(m.phaseFilmY[s]) || m.phaseFilmY[s]<0) {
                error="phase composition is invalid"; return false;
            }
            sum+=m.phaseFilmY[s];
            for (int e=0; e<Ne; ++e) elements[e]+=m.phaseFilmY[s]*p.species[s].element[e];
        }
        if (!closeEnough(sum,1,1e-14,p.tolerances.relativeMass)) {
            error="phase composition does not sum to one"; return false;
        }
        for (int e=0; e<Ne; ++e) {
            if (!closeEnough(elements[e],p.condensed[m.phaseCondensed].element[e],1e-13,1e-12)) {
                error="phase map changes elemental inventory"; return false;
            }
        }
    }
    if (m.nSurfaceReactions<0 || m.nSurfaceReactions>Nr
        || (m.enableSurfaceReactions && m.nSurfaceReactions==0)) {
        error="surface reaction count is invalid"; return false;
    }
    for (int r=0; r<m.nSurfaceReactions; ++r) {
        if (!validateReaction(m.surfaceReactions[r],p,r,error)) return false;
        for (int s=0; s<Ns; ++s) {
            const Real order=m.surfaceGasOrder[r][s];
            if (!finite(order) || order<0 || (order>0 && m.surfaceReactions[r].gasNu[s]>=0)) {
                error="surface gas reaction order is invalid"; return false;
            }
        }
    }
    if (m.evaporation!=EvaporationLaw::Disabled && m.evaporation!=EvaporationLaw::HertzKnudsen) {
        error="unsupported evaporation law"; return false;
    }
    bool active=false;
    for (int s=0; s<Ns; ++s) {
        const Real a=m.evaporationAccommodation[s];
        if (!finite(a) || a<0 || a>1 || !finite(m.saturationPressureScale[s])
            || !finite(m.saturationTemperature[s]) || m.saturationPressureScale[s]<0
            || m.saturationTemperature[s]<0 || (a>0 && m.saturationPressureScale[s]<=0)) {
            error="invalid Hertz-Knudsen/saturation inputs"; return false;
        }
        active=active || a>0;
    }
    if (m.evaporation!=EvaporationLaw::Disabled && (!active || !p.enableFilm)) {
        error="evaporation requires explicitly active species and film"; return false;
    }
    error.clear(); return true;
}
} // namespace chmt
#endif
