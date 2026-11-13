#ifndef CHMT_MATERIALS_REACTION_H
#define CHMT_MATERIALS_REACTION_H
#include "materials/Thermodynamics.H"

namespace chmt {
struct MaterialPrimitive {
    EnergyReal temperature=0, pressure=0, conductivity=0;
    PhysicalReal porosity=0, poreDensity=0;
};

CHMT_HD inline SolidQ addSolid(const SolidQ& a, const SolidQ& b, PhysicalReal scale=1) {
    SolidQ out=a;
    for (int c=0; c<Nc; ++c) out.condensed[c]+=scale*b.condensed[c];
    for (int s=0; s<Ns; ++s) out.pore[s]+=scale*b.pore[s];
    for (int r=0; r<Nr; ++r) out.progress[r]+=scale*b.progress[r];
    out.energy+=EnergyReal(scale)*b.energy;
    return out;
}

template<bool Terminal=false>
CHMT_HD inline bool recoverMaterial(const SolidQ& q, PhysicalReal volume, const PhysicsConfig& p,
    MaterialPrimitive& out, DeviceStatus* status=nullptr, int index=-1,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if ((!finite(volume) || volume<=0)) {
        return failStatus(status,ErrorCode::InvalidInput,index,volume,ErrorLocation::Cell);
    }
    PhysicalReal occupied=0, poreMass=0;
    EnergyReal poreR=0, conductivity=0;
    for (int c=0; c<Nc; ++c) {
        if ((!finite(q.condensed[c]) || q.condensed[c]<0)) {
            return failStatus(status,ErrorCode::Inventory,index,q.condensed[c],ErrorLocation::Cell);
        }
        if (q.condensed[c]==0) continue;
        if (!validCondensed(p.condensed[c])) {
            return failStatus(status,ErrorCode::PropertyRange,index,c,ErrorLocation::Cell);
        }
        const PhysicalReal constituentVolume=q.condensed[c]/p.condensed[c].rho;
        occupied+=constituentVolume;
        conductivity+=EnergyReal(constituentVolume)*EnergyReal(p.condensed[c].conductivity)/EnergyReal(volume);
    }
    PhysicalReal phi=1-occupied/volume;
    if (!(occupied>0)||!clipPhysicsMassFraction(p,phi,3822,index))
        return failStatus(status,ErrorCode::Inventory,index,phi,ErrorLocation::Cell);
    if(phi<p.material.minimumPorosity||phi>=p.material.maximumPorosity)
        reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3823,phi,phi<p.material.minimumPorosity?p.material.minimumPorosity:p.material.maximumPorosity);
    SolidQ thermal=q;
    thermal.porosity=phi;
    MaterialPrimitive next;
    if (!recoverSolid<Terminal>(thermal,p,next.temperature,status,index,numericalFailure)) return false;
    for (int s=0; s<Ns; ++s) {
        poreMass+=q.pore[s];
        poreR+=EnergyReal(q.pore[s])*EnergyReal(p.species[s].R);
    }
    if (((poreMass>0 && phi<=0) || !finite(poreMass) || !finite(poreR))) {
        return failStatus(status,ErrorCode::Inventory,index,poreMass,ErrorLocation::Cell);
    }
    next.porosity=phi;
    next.poreDensity=phi>0 ? poreMass/(phi*volume) : 0;
    next.pressure=phi>0 ? poreR*next.temperature/(EnergyReal(phi)*EnergyReal(volume)) : EnergyReal(0);
    next.conductivity=conductivity+(poreMass>0 ? EnergyReal(phi)*EnergyReal(p.gasConductivity) : EnergyReal(0));
    if ((!finite(next.pressure) || !finite(next.conductivity) || next.conductivity<0)) {
        return failStatus(status,ErrorCode::Nonfinite,index,next.pressure,ErrorLocation::Cell);
    }
    out=next;
    return true;
}

// The same material caloric law and conductivity used by the material owner
// supply the finite-contact wall response; particle material defaults are not
// properties of a CHMT wall.
CHMT_HD inline EnergyReal materialThermalEffusivity(const SolidQ& q,PhysicalReal volume,const PhysicsConfig& p){
    MaterialPrimitive w;if(!recoverMaterial(q,volume,p,w))return undefinedValue();
    EnergyReal capacity=0;
    for(int c=0;c<Nc;++c)capacity+=EnergyReal(q.condensed[c])*(EnergyReal(p.condensed[c].cp0)+EnergyReal(p.condensed[c].cp1)*w.temperature);
    const GasThermoAdapter<Ns> thermo(p.species);
    for(int j=0;j<Ns;++j)if(q.pore[j]!=0)capacity+=EnergyReal(q.pore[j])*ugkwp::speciesCv(j,w.temperature,thermo.view(p.diagnostics));
    return std::sqrt(w.conductivity*capacity/EnergyReal(volume));
}

// All reactions evaluate the same stage state. Bound the NET assembled species
// loss with one common extent factor. Gross-consumption limiting would freeze
// a midpoint-generated intermediate absent from the accepted base, even when
// simultaneous production leaves its final inventory positive. Shared-donor
// losses still add before applying the one conservative mechanism-wide factor.
template<bool Terminal=false>
CHMT_HD inline bool boundedReactionRates(const SolidQ& evaluation, const SolidQ& capacity,
    PhysicalReal volume, const PhysicsConfig& p, PhysicalReal dt, PhysicalReal* rates,
    DeviceStatus* status=nullptr, int index=-1,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if(!rates){if(numericalFailure)*numericalFailure=false;return failStatus(status,ErrorCode::InvalidInput,index,0,ErrorLocation::Cell);}
    if((!finite(dt)||dt<=0))return failStatus(status,ErrorCode::InvalidInput,index,dt,ErrorLocation::Cell);
    for (int r=0; r<Nr; ++r) rates[r]=0;
    if (!p.enableReactions) return true;
    PhysicalReal T=0;
    if (!recoverSolid<Terminal>(evaluation,p,T,status,index,numericalFailure)) return false;
    SolidQ consumption;
    for (int r=0; r<p.nReactions; ++r) {
        rates[r]=reactionRate<Terminal>(p.reactions[r],evaluation.condensed,volume,T);
        if ((!finite(rates[r]) || rates[r]<0)) return failStatus(status,ErrorCode::Nonfinite,index,rates[r],ErrorLocation::Cell);
        for (int c=0; c<Nc; ++c) {
            consumption.condensed[c]-=rates[r]*p.reactions[r].condensedNu[c];
        }
        for (int species=0; species<Ns; ++species) {
            consumption.pore[species]-=rates[r]*p.reactions[r].gasNu[species];
        }
    }
    PhysicalReal factor=1;
    for (int c=0; c<Nc; ++c) if (consumption.condensed[c]>0) {
        factor=minValue(factor,capacity.condensed[c]/(dt*consumption.condensed[c]));
    }
    for (int species=0; species<Ns; ++species) if (consumption.pore[species]>0) {
        factor=minValue(factor,capacity.pore[species]/(dt*consumption.pore[species]));
    }
    if ((!finite(factor) || factor<0)) return failStatus(status,ErrorCode::Nonfinite,index,factor,ErrorLocation::Cell);
    for (int r=0; r<p.nReactions; ++r) rates[r]*=factor;
    return true;
}
template<bool Terminal=false>
CHMT_HD inline bool advanceLocalReactions(const SolidQ& base, PhysicalReal volume,
    const PhysicsConfig& p, PhysicalReal dt, SolidQ& out, PhysicalReal* rates,
    DeviceStatus* status=nullptr, int index=-1,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    PhysicalReal localRates[Nr]{};
    if (!boundedReactionRates<Terminal>(base,base,volume,p,dt,localRates,status,index,numericalFailure)) return false;
    SolidQ next=base;
    for (int r=0; r<p.nReactions; ++r) {
        const PhysicalReal extent=dt*localRates[r];
        for (int c=0; c<Nc; ++c) next.condensed[c]+=extent*p.reactions[r].condensedNu[c];
        for (int species=0; species<Ns; ++species) next.pore[species]+=extent*p.reactions[r].gasNu[species];
        next.progress[r]+=extent;
    }
    for (int r=0; r<Nr; ++r) if (rates) rates[r]=localRates[r];
    MaterialPrimitive state;
    if (!recoverMaterial<Terminal>(next,volume,p,state,status,index,numericalFailure)) return false;
    next.porosity=state.porosity;
    out=next; return true;
}

// Stationary skeleton ALE flux uses INTERNAL energy, not pore enthalpy. Darcy's
// physical pore flow below separately uses enthalpy. Interface faces replace
// this boundary remap by the same accepted moving-interface packet.
CHMT_HD inline SolidQ materialAleFlux(const SolidQ& q, PhysicalReal volume, PhysicalReal meshVolumeRate) {
    SolidQ flux;
    const PhysicalReal factor=-meshVolumeRate/volume;
    for (int c=0; c<Nc; ++c) flux.condensed[c]=factor*q.condensed[c];
    for (int s=0; s<Ns; ++s) flux.pore[s]=factor*q.pore[s];
    for (int r=0; r<Nr; ++r) flux.progress[r]=factor*q.progress[r];
    flux.energy=(-EnergyReal(meshVolumeRate)/EnergyReal(volume))*q.energy;
    return flux;
}

CHMT_HD inline EnergyReal seriesConductance(EnergyReal k0, EnergyReal k1, EnergyReal d0, EnergyReal d1) {
    if (k0<=0 || k1<=0) return 0;
    return 1/(d0/k0+d1/k1);
}

template<bool Terminal=false>
CHMT_HD inline bool darcyConductionFluxWithGradient(const SolidQ& left, const SolidQ& right,
    const MaterialPrimitive& a, const MaterialPrimitive& b, PhysicalReal volumeA, PhysicalReal volumeB,
    PhysicalReal area, PhysicalReal distance, PhysicalVec3 normal, PhysicalReal normalPressureGradient,
    const PhysicsConfig& p, SolidQ& out) {
    if ((!(volumeA>0 && volumeB>0 && area>0 && distance>0)
        || !finite(normal) || !finite(normalPressureGradient))) return false;
    SolidQ flux;
    PhysicalReal volumeRate=0;
    if (p.permeability>0) {
        if (!(p.poreViscosity>0)) return false;
        const PhysicalReal rho=0.5f*(a.poreDensity+b.poreDensity);
        volumeRate=-area*p.permeability/p.poreViscosity
            *(normalPressureGradient-rho*dot(p.gravity,normal));
    }
    if (!finite(volumeRate)) return false;
    const SolidQ& donor=volumeRate>=0 ? left : right;
    const MaterialPrimitive& state=volumeRate>=0 ? a : b;
    const PhysicalReal volume=volumeRate>=0 ? volumeA : volumeB;
    if (volumeRate!=0 && state.porosity<=0) return false;
    for (int s=0; s<Ns; ++s) {
        flux.pore[s]=state.porosity>0 ? volumeRate*donor.pore[s]/(state.porosity*volume) : 0;
        // Dry/no-Darcy faces transport no gas enthalpy. Their inactive gas
        // tables may be absent; never turn zero transport into 0*NaN.
        if(flux.pore[s]!=0){
            EnergyReal enthalpy=speciesH(p.species[s],state.temperature);
            if(!finite(enthalpy))return false;
            flux.energy+=EnergyReal(flux.pore[s])*enthalpy;
        }
    }
    flux.energy+=EnergyReal(area)*seriesConductance(EnergyReal(a.conductivity),EnergyReal(b.conductivity),EnergyReal(distance)/2,EnergyReal(distance)/2)
        *(EnergyReal(a.temperature)-EnergyReal(b.temperature));
    if (!finite(flux.energy)) return false;
    out=flux;
    return true;
}

// Orthogonal/two-point convenience entry. General mesh operators must use
// darcyConductionFluxWithGradient with a corrected original-cell derivative.
CHMT_HD inline bool darcyConductionFlux(const SolidQ& left, const SolidQ& right,
    const MaterialPrimitive& a, const MaterialPrimitive& b, PhysicalReal volumeA, PhysicalReal volumeB,
    PhysicalReal area, PhysicalReal distance, PhysicalVec3 normal, const PhysicsConfig& p, SolidQ& out) {
    if (!(distance>0)) return false;
    return darcyConductionFluxWithGradient(left,right,a,b,volumeA,volumeB,area,
        distance,normal,(b.pressure-a.pressure)/distance,p,out);
}

inline bool validateMaterialPolicy(const PhysicsConfig& p, std::string& error) {
    const MaterialPolicyConfig& m=p.material;
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
        PhysicalReal sum=0, elements[Ne]{};
        for (int s=0; s<Ns; ++s) {
            if (!finite(m.phaseFilmY[s])) {
                error="phase composition is invalid"; return false;
            }
            const PhysicalReal fraction=physicalFraction(p,m.phaseFilmY[s],s);sum+=fraction;
            for (int e=0; e<Ne; ++e) elements[e]+=fraction*p.species[s].element[e];
        }
        if(!finite(sum)){error="nonfinite phase composition sum="+std::to_string(sum);return false;}
        if (!closeEnough(sum,1,1e-14f,p.tolerances.relativeMass)) {
            reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3836,sum-1,1e-14f);
        }
        for (int e=0; e<Ne; ++e) {
            if(!finite(elements[e])||!finite(p.condensed[m.phaseCondensed].element[e])){error="nonfinite phase element "+std::to_string(e)+" value="+std::to_string(elements[e]);return false;}
            if (!closeEnough(elements[e],p.condensed[m.phaseCondensed].element[e],1e-13f,1e-12f)) {
                reportPhysicsWarning(p,ugkp::diagnostics::Code::ConservationResidual,3837,elements[e]-p.condensed[m.phaseCondensed].element[e],1e-13f,0,e);
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
            const PhysicalReal order=m.surfaceGasOrder[r][s];
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
        const PhysicalReal a=m.evaporationAccommodation[s];
        if (!finite(a) || !finite(m.saturationPressureScale[s])
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
