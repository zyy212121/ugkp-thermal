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
CHMT_HD inline bool accountAssembledPacket(ExchangePacket& packet,
    const Tolerances& tol,DeviceStatus* status=nullptr) {
    if(!validatePacketMath(packet,tol,status))return false;
    if(packet.consumerMask)return packetFailure(status,packet,ErrorCode::Duplicate,packet.consumerMask,tol.diagnostics);
    packet.consumerMask=requiredConsumers(packet.kind);return true;
}

} // namespace chmt
#endif
