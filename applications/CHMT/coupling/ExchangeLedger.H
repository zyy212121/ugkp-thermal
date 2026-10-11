#ifndef CHMT_COUPLING_EXCHANGELEDGER_H
#define CHMT_COUPLING_EXCHANGELEDGER_H
#include "core/Configuration.H"
#include <string>
namespace chmt {
struct PacketDelta { GasQ gas{};SolidQ solid{};FilmQ film{};Real particleMass=0,particleEnergy=0;Vec3 particleMomentum{}; };
CHMT_HD inline unsigned requiredConsumers(ExchangeKind kind) {
    switch(kind){
    case ExchangeKind::GasSolid:case ExchangeKind::PoreGas:return ConsumeGas|ConsumeSolid;
    case ExchangeKind::GasFilm:return ConsumeGas|ConsumeFilm;
    case ExchangeKind::SolidFilm:return ConsumeSolid|ConsumeFilm;
    case ExchangeKind::ParticleGas:return ConsumeParticle|ConsumeGas;
    case ExchangeKind::ParticleWall:return ConsumeParticle|ConsumeSolid;
    }return 0;
}
CHMT_HD inline PacketDelta packetDelta(const ExchangePacket& p) {
    PacketDelta d{};
    if(requiredConsumers(p.kind)&ConsumeGas){
        d.gas.mass=p.mass;d.gas.momentum=p.momentum;d.gas.energy=p.energy;
        for(int s=0;s<Ns;++s)d.gas.species[s]=p.species[s];
    }
    if(requiredConsumers(p.kind)&ConsumeSolid){
        const Real sign=p.kind==ExchangeKind::ParticleWall?1:-1;
        for(int c=0;c<Nc;++c)d.solid.condensed[c]=sign*p.condensed[c];
        for(int s=0;s<Ns;++s)d.solid.pore[s]=sign*p.pore[s];
        d.solid.energy=sign*p.energy;
    }
    if(requiredConsumers(p.kind)&ConsumeFilm){
        const Real sign=p.kind==ExchangeKind::SolidFilm?1:-1;
        d.film.mass=sign*p.mass;for(int s=0;s<Ns;++s)d.film.species[s]=sign*p.species[s];
        d.film.enthalpy=sign*p.energy+p.liquidKineticAdvection;
    }
    if(requiredConsumers(p.kind)&ConsumeParticle){
        d.particleMass=-p.mass;d.particleEnergy=-p.energy;d.particleMomentum=-p.momentum;
    }
    return d;
}
// Physical packets never contain delta(pV). The film owner forms exactly one
// storage update from accepted and candidate p,V for each face/stage.
CHMT_HD inline Real filmPressureVolumeDelta(const FilmPressureVolumeUpdate& u) { return u.newPV-u.oldPV; }
CHMT_HD bool validatePacketMath(const ExchangePacket&,const Tolerances&,DeviceStatus* status=nullptr);
// Raw staged delta helper: caller MUST mark the successful consumer bit on the
// uniquely owned packet before another invocation. It is not a stand-alone
// consumer. Prefer applyPacketTransaction or consumePacket wrappers.
CHMT_HD bool applyPacketDelta(const ExchangePacket&,unsigned consumerBit,GasQ* gas,SolidQ* solid,FilmQ* film,Budget&,DeviceStatus* status=nullptr,bool pendingFilmStorage=false);
bool validatePacket(const ExchangePacket&,const PhysicsConfig&,std::string& error);
bool consumePacket(ExchangePacket&,unsigned consumerBit,GasQ* gas,SolidQ* solid,FilmQ* film,Budget&,std::string& error);
CHMT_HD bool applyPacketTransaction(ExchangePacket&,GasQ*,SolidQ*,FilmQ*,FilmPressureVolumeUpdate*,Budget&,const Tolerances&,DeviceStatus* status=nullptr);
bool consumePacketTransaction(ExchangePacket&,GasQ*,SolidQ*,FilmQ*,FilmPressureVolumeUpdate*,Budget&,std::string& error);
bool consumeFilmPressureVolume(FilmPressureVolumeUpdate&,FilmQ&,Budget&,std::string& error);
// Generic particle transfer uses the same packetDelta, but ParticleQ's mass,
// velocity and thermal-energy convention requires its owner to atomically
// consume both particle and receiver. The host API above rejects ConsumeParticle.
} // namespace chmt
#include "ExchangeLedger.C"
#endif
