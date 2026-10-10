#ifndef CHMT_COUPLING_COORDINATOR_H
#define CHMT_COUPLING_COORDINATOR_H
#include "core/HostState.H"
#include "coupling/ExchangeLedger.H"
#include "materials/Reaction.H"
#include <utility>
namespace chmt {
// Constant-composition, local-temperature compressible pore mass storage
// dm/dp=phi*V/(Rmix*T). Empty pores use the largest configured R, so a
// neighboring donor still gives a finite conservative pressure-relaxation bound.
CHMT_HD inline Real porePressureStorage(const SolidQ& q,Real poreVolume,Real temperature,const PhysicsConfig& p){
    Real mass=0,weightedR=0,maximumR=0;for(int s=0;s<Ns;++s){mass+=q.pore[s];weightedR+=q.pore[s]*p.species[s].R;maximumR=maxValue(maximumR,p.species[s].R);}
    const Real R=mass>0?weightedR/mass:maximumR;
    return poreVolume>0&&temperature>0&&R>0?poreVolume/(R*temperature):0;
}
CHMT_HD inline Real darcyPressureRelaxation(Real storage,Real permeability,Real viscosity,Real faceDensity,Real area,Real distance){
    if(permeability==0||faceDensity==0)return 0;
    if(storage<=0||viscosity<=0||area<=0||distance<=0)return std::numeric_limits<Real>::max();
    return permeability*faceDensity*area/(viscosity*distance*storage);
}
// Coupled gray radiation belongs to the material-side primary receiver.
// A dry GasSolid face may retain an empty FilmQ for later melting; that does
// not make the film the current radiation owner. Birth switches by packet kind.
CHMT_HD inline bool filmReceivesSurfaceRadiation(bool coupled,const ExchangePacket* primary){
    return !coupled||(primary&&primary->kind==ExchangeKind::GasFilm);
}
CHMT_HD inline bool validBudget(const Budget& b){
    const Real values[]={b.boundaryMass,b.boundaryEnergy,b.supportWork,b.bodyWork,b.radiation,b.gclResidual,
        b.filmPressureVolume,b.filmKineticAdvection,b.filmKineticStorage,b.filmReducedResidual,b.filmKineticDefect,
        b.thinLayerGeometryDefect,b.thinLayerStressDefect,b.numericalEnergyResidual,b.turbulenceOmegaConstraint,
        b.turbulenceInventory,b.turbulenceProduction,b.turbulenceDissipation,b.turbulenceBoundaryFlux};
    for(Real value:values)if(!finite(value))return false;
    if(!finite(b.boundaryMomentum)||!finite(b.supportImpulse))return false;
    for(int s=0;s<Ns;++s)if(!finite(b.boundarySpecies[s]))return false;
    for(int e=0;e<Ne;++e)if(!finite(b.boundaryElements[e]))return false;
    for(int k=0;k<NParticipants;++k){if(!finite(b.exchangeMass[k])||!finite(b.exchangeEnergy[k])||!finite(b.exchangeMomentum[k]))return false;
        for(int s=0;s<Ns;++s)if(!finite(b.exchangeSpecies[k][s]))return false;}
    return true;
}
CHMT_HD inline SolidQ assembleSolidCandidate(const SolidQ& base,const SolidQ& rhs,Real dt,
    const ExchangePacket* packets,int count,int cell) {
    SolidQ next=addSolid(base,rhs,dt);
    for(int i=0;i<count;++i)if((requiredConsumers(packets[i].kind)&ConsumeSolid)&&packets[i].solidCell==cell)
        next=addSolid(next,packetDelta(packets[i]).solid);
    return next;
}
CHMT_HD inline FilmQ assembleFilmCandidate(const FilmQ& base,const FilmQ& rhs,Real dt,
    const ExchangePacket* packets,int count,int face,const FilmPressureVolumeUpdate& storage,Real radiation) {
    FilmQ next=base;next.mass+=dt*rhs.mass;
    next.enthalpy=(base.enthalpy-storage.oldPV)+dt*rhs.enthalpy;
    for(int s=0;s<Ns;++s)next.species[s]+=dt*rhs.species[s];
    for(int i=0;i<count;++i)if(packets[i].kind!=ExchangeKind::SolidFilm&&(requiredConsumers(packets[i].kind)&ConsumeFilm)&&packets[i].filmFace==face){
        const auto d=packetDelta(packets[i]).film;next.mass+=d.mass;
        next.enthalpy-=packets[i].energy;next.enthalpy+=packets[i].liquidKineticAdvection;
        for(int s=0;s<Ns;++s)next.species[s]+=d.species[s];
    }
    next.enthalpy+=dt*radiation;
    for(int i=0;i<count;++i)if(packets[i].kind==ExchangeKind::SolidFilm&&packets[i].filmFace==face){
        const auto d=packetDelta(packets[i]).film;next.mass+=d.mass;
        // Exact dry-event pairing uses the same arithmetic grouping as its
        // packet owner: phase.energy=-(remainingU+outwardK). No clipping.
        next.enthalpy+=packets[i].liquidKineticAdvection;next.enthalpy+=packets[i].energy;
        for(int s=0;s<Ns;++s)next.species[s]+=d.species[s];
    }
    next.enthalpy+=storage.newPV;return next;
}
// Participants have ALREADY been assembled from their common base. This
// function only records the exact common transfer; it never reapplies a source.
CHMT_HD inline bool accountAssembledPacket(ExchangePacket& packet,Budget& budget,
    const Tolerances& tol,DeviceStatus* status=nullptr) {
    if(!validatePacketMath(packet,tol,status))return false;
    if(packet.consumerMask)return packetFailure(status,packet,ErrorCode::Duplicate,packet.consumerMask);
    Budget next=budget;const auto delta=packetDelta(packet);const unsigned mask=requiredConsumers(packet.kind);
    for(int k=0;k<NParticipants;++k){unsigned bit=1u<<k;if(!(mask&bit))continue;
        Real dm=0,de=0;Vec3 momentum{};Real species[Ns]{};
        if(bit==ConsumeGas){dm=delta.gas.mass;de=delta.gas.energy;momentum=delta.gas.momentum;
            for(int s=0;s<Ns;++s)species[s]=delta.gas.species[s];}
        if(bit==ConsumeSolid){const Real sign=packet.kind==ExchangeKind::ParticleWall?1:-1;
            dm=sign*packet.mass;de=delta.solid.energy;momentum=packet.momentum*sign;
            for(int s=0;s<Ns;++s){species[s]=sign*packet.species[s];}
            next.supportImpulse+=momentum;}
        if(bit==ConsumeFilm){const Real sign=packet.kind==ExchangeKind::SolidFilm?1:-1;
            dm=delta.film.mass;de=sign*packet.energy;momentum=packet.momentum*sign;
            for(int s=0;s<Ns;++s)species[s]=delta.film.species[s];
            next.filmKineticAdvection+=packet.liquidKineticAdvection;}
        if(bit==ConsumeParticle){dm=delta.particleMass;de=delta.particleEnergy;momentum=delta.particleMomentum;
            for(int s=0;s<Ns;++s)species[s]=-packet.species[s];}
        next.exchangeMass[k]+=dm;next.exchangeEnergy[k]+=de;next.exchangeMomentum[k]+=momentum;
        if(!finite(next.exchangeMass[k])||!finite(next.exchangeEnergy[k])||!finite(next.exchangeMomentum[k]))
            return packetFailure(status,packet,ErrorCode::Nonfinite,de);
        for(int s=0;s<Ns;++s){next.exchangeSpecies[k][s]+=species[s];if(!finite(next.exchangeSpecies[k][s]))return false;}
    }
    if(!finite(next.supportImpulse)||!finite(next.filmKineticAdvection))return false;
    if(next.consumedPackets==std::numeric_limits<std::uint64_t>::max()||!validBudget(next))return packetFailure(status,packet,ErrorCode::Nonfinite,0);
    ++next.consumedPackets;packet.consumerMask=mask;budget=next;return true;
}
inline bool publishAcceptedCandidate(HostState& accepted,HostState candidate,int stage,
    Real dt,bool synchronized,std::string& error) {
    if(stage!=1||!synchronized||!finite(dt)||dt<=0){error="only a synchronized final stage may commit";return false;}
    for(const auto& q:candidate.gas)if(!validGasInventory(q)){error="invalid candidate gas inventory";return false;}
    for(const auto& q:candidate.solid)if(!validSolidInventory(q)){error="invalid candidate material inventory";return false;}
    for(const auto& q:candidate.film)if(!validFilmInventory(q)){error="invalid candidate film inventory";return false;}
    for(const auto& p:candidate.ledger)if(p.consumerMask!=requiredConsumers(p.kind)){error="unconsumed candidate packet";return false;}
    for(const auto& s:candidate.filmStorage)if(!s.consumed){error="unconsumed candidate film storage";return false;}
    if(accepted.acceptedSteps==std::numeric_limits<std::uint64_t>::max()||accepted.commitSequence==std::numeric_limits<std::uint64_t>::max()){error="accepted counter overflow";return false;}
    candidate.time=accepted.time+dt;candidate.acceptedSteps=accepted.acceptedSteps+1;
    candidate.commitSequence=accepted.commitSequence+1;candidate.lastAcceptedDt=dt;
    if(!finite(candidate.time)||candidate.time<=accepted.time){error="nonfinite/nonincreasing candidate time";return false;}
    accepted=std::move(candidate);error.clear();return true;
}
} // namespace chmt
#endif
