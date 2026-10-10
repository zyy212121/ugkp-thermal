#ifndef CHMT_COUPLING_EXCHANGELEDGER_IMPLEMENTATION
#define CHMT_COUPLING_EXCHANGELEDGER_IMPLEMENTATION
#include "ExchangeLedger.H"
namespace chmt {
CHMT_HD inline bool packetFailure(DeviceStatus* status,const ExchangePacket& p,ErrorCode code,Real value) {
    // A persistent face is uint64; index carries local film/gas/solid location,
    // while the host diagnostic below retains the full persistent identifier.
    int index=p.filmFace>=0?p.filmFace:(p.gasCell>=0?p.gasCell:p.solidCell);
    return failStatus(status,code,index,value,ErrorLocation::Face);
}
CHMT_HD inline bool packetClose(Real value,Real target,Real absolute,Real relative,Real scale) {
    if(!finite(value)||!finite(target)||!finite(absolute)||!finite(relative)||!finite(scale)||absolute<0||relative<0||scale<0)return false;
    const Real residual=value-target;
    const Real tolerance=absolute+relative*maxValue(scale,maxValue(absValue(value),absValue(target)));
    return finite(residual)&&finite(tolerance)&&absValue(residual)<=tolerance;
}
CHMT_HD inline bool validatePacketMath(const ExchangePacket& p,const Tolerances& tol,DeviceStatus* status) {
    const unsigned required=requiredConsumers(p.kind);
    if(required==0||p.stage<0||p.stage>1||(p.consumerMask&~required)!=0)
        return packetFailure(status,p,ErrorCode::Packet,p.consumerMask);
    if(((required&ConsumeGas)&&p.gasCell<0)||((required&ConsumeSolid)&&p.solidCell<0)
        ||((required&ConsumeFilm)&&p.filmFace<0)||((required&ConsumeParticle)&&p.particleIndex<0))
        return packetFailure(status,p,ErrorCode::Packet,-1);
    if(!finite(p.mass)||!finite(p.momentum)||!finite(p.energy)||!finite(p.conductive)||!finite(p.advective)
        ||!finite(p.pressureWork)||!finite(p.viscousWork)||!finite(p.radiation)||!finite(p.liquidKineticAdvection))
        return packetFailure(status,p,ErrorCode::Nonfinite,p.energy);
    Real sum=0,scale=0,donor=0,donorScale=0;
    for(int s=0;s<Ns;++s){
        if(!finite(p.species[s])||!finite(p.pore[s])||!finite(p.poreSweep[s]))return packetFailure(status,p,ErrorCode::Nonfinite,s);
        if(p.kind!=ExchangeKind::PoreGas&&p.poreSweep[s]!=0)return packetFailure(status,p,ErrorCode::Packet,p.poreSweep[s]);
        sum+=p.species[s];scale+=absValue(p.species[s]);donor+=p.pore[s];donorScale+=absValue(p.pore[s]);
    }
    for(int c=0;c<Nc;++c){if(!finite(p.condensed[c]))return packetFailure(status,p,ErrorCode::Nonfinite,c);donor+=p.condensed[c];donorScale+=absValue(p.condensed[c]);}
    if(!finite(sum)||!finite(scale)||!finite(donor)||!finite(donorScale))return packetFailure(status,p,ErrorCode::Nonfinite,sum);
    // Deposition has no gas receiver and therefore no fictitious gas-species
    // equivalent. Its actual condensed/pore receiver still closes below.
    if(p.kind==ExchangeKind::ParticleWall){
        if(scale!=0)return packetFailure(status,p,ErrorCode::Composition,scale);
    }else if(!packetClose(sum,p.mass,tol.absoluteMass,tol.relativeMass,scale))return packetFailure(status,p,ErrorCode::Composition,sum-p.mass);
    if((required&ConsumeSolid)&&!packetClose(donor,p.mass,tol.absoluteMass,tol.relativeMass,donorScale))
        return packetFailure(status,p,ErrorCode::Composition,donor-p.mass);
    if(!(required&ConsumeSolid)&&donorScale!=0)return packetFailure(status,p,ErrorCode::Packet,donorScale);
    if(p.kind==ExchangeKind::PoreGas){
        for(int c=0;c<Nc;++c)if(p.condensed[c]!=0)return packetFailure(status,p,ErrorCode::Packet,p.condensed[c]);
        for(int s=0;s<Ns;++s)if(!packetClose(p.pore[s],p.species[s],tol.absoluteMass,tol.relativeMass,absValue(p.species[s])))
            return packetFailure(status,p,ErrorCode::Composition,p.pore[s]-p.species[s]);
    }
    if(!(required&ConsumeFilm)&&p.liquidKineticAdvection!=0)return packetFailure(status,p,ErrorCode::Packet,p.liquidKineticAdvection);
    const Real parts=p.conductive+p.advective+p.pressureWork+p.viscousWork+p.radiation;
    const Real partScale=absValue(p.conductive)+absValue(p.advective)+absValue(p.pressureWork)+absValue(p.viscousWork)+absValue(p.radiation);
    if(!finite(parts)||!finite(partScale))return packetFailure(status,p,ErrorCode::Nonfinite,parts);
    if(!packetClose(parts,p.energy,tol.absoluteEnergy,tol.relativeEnergy,partScale))return packetFailure(status,p,ErrorCode::Packet,parts-p.energy);
    return true;
}
CHMT_HD inline bool validGasInventory(const GasQ& q) {
    if(!finite(q.mass)||q.mass<=0||!finite(q.momentum)||!finite(q.energy))return false;
    Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0)return false;sum+=q.species[s];}
    return packetClose(sum,q.mass,1e-14,1e-10,sum);
}
CHMT_HD inline bool validSolidInventory(const SolidQ& q) {
    if(!finite(q.energy)||!finite(q.porosity)||q.porosity<0||q.porosity>=1)return false;
    Real mass=0;for(int c=0;c<Nc;++c){if(!finite(q.condensed[c])||q.condensed[c]<0)return false;mass+=q.condensed[c];}
    for(int s=0;s<Ns;++s){if(!finite(q.pore[s])||q.pore[s]<0)return false;mass+=q.pore[s];}
    for(int r=0;r<Nr;++r)if(!finite(q.progress[r])||q.progress[r]<0)return false;
    return finite(mass)&&(mass!=0||q.energy==0);
}
CHMT_HD inline bool validFilmInventory(const FilmQ& q,bool pendingStorage=false) {
    if(!finite(q.mass)||q.mass<0||!finite(q.enthalpy))return false;
    Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0)return false;sum+=q.species[s];}
    return packetClose(sum,q.mass,1e-14,1e-10,sum)&&(pendingStorage||q.mass!=0||q.enthalpy==0);
}
CHMT_HD inline bool applyPacketDelta(const ExchangePacket& p,unsigned bit,GasQ* gas,SolidQ* solid,FilmQ* film,Budget& budget,DeviceStatus* status,bool pendingFilmStorage) {
    if(bit==0||(bit&(bit-1))!=0||(bit&requiredConsumers(p.kind))==0)return packetFailure(status,p,ErrorCode::Packet,bit);
    if(p.consumerMask&bit)return packetFailure(status,p,ErrorCode::Duplicate,bit);
    const PacketDelta d=packetDelta(p);int participant=-1;Real dm=0,dE=0;Vec3 dMomentum{};
    Real ds[Ns]{};
    GasQ g{};SolidQ s{};FilmQ f{};
    if(bit==ConsumeGas){
        if(!gas)return packetFailure(status,p,ErrorCode::InvalidInput,bit);
        g=*gas+d.gas;if(!validGasInventory(g))return packetFailure(status,p,ErrorCode::Inventory,g.mass);
        participant=GasParticipant;dm=d.gas.mass;dE=d.gas.energy;dMomentum=d.gas.momentum;
        for(int k=0;k<Ns;++k)ds[k]=d.gas.species[k];
    }else if(bit==ConsumeSolid){
        if(!solid)return packetFailure(status,p,ErrorCode::InvalidInput,bit);
        s=*solid;for(int c=0;c<Nc;++c)s.condensed[c]+=d.solid.condensed[c];
        for(int k=0;k<Ns;++k)s.pore[k]+=d.solid.pore[k];
        s.energy+=d.solid.energy;
        if(!validSolidInventory(s))return packetFailure(status,p,ErrorCode::Inventory,s.energy);
        participant=SolidParticipant;dm=p.kind==ExchangeKind::ParticleWall?p.mass:-p.mass;dE=d.solid.energy;
        dMomentum=p.kind==ExchangeKind::ParticleWall?p.momentum:-p.momentum;
        for(int k=0;k<Ns;++k)ds[k]=(p.kind==ExchangeKind::ParticleWall?1:-1)*p.species[k];
    }else if(bit==ConsumeFilm){
        if(!film)return packetFailure(status,p,ErrorCode::InvalidInput,bit);
        f=*film;f.mass+=d.film.mass;f.enthalpy+=d.film.enthalpy;
        for(int k=0;k<Ns;++k)f.species[k]+=d.film.species[k];
        if(!validFilmInventory(f,pendingFilmStorage))return packetFailure(status,p,ErrorCode::Inventory,f.mass);
        participant=FilmParticipant;dm=d.film.mass;dE=p.kind==ExchangeKind::SolidFilm?p.energy:-p.energy;
        dMomentum=(p.kind==ExchangeKind::SolidFilm?1:-1)*p.momentum;
        for(int k=0;k<Ns;++k)ds[k]=d.film.species[k];
    }else return packetFailure(status,p,ErrorCode::Unsupported,bit);
    Budget b=budget;b.exchangeMass[participant]+=dm;b.exchangeEnergy[participant]+=dE;b.exchangeMomentum[participant]+=dMomentum;
    for(int k=0;k<Ns;++k)b.exchangeSpecies[participant][k]+=ds[k];
    if((p.consumerMask|bit)==requiredConsumers(p.kind))++b.consumedPackets;
    if(bit==ConsumeFilm)b.filmKineticAdvection+=p.liquidKineticAdvection;
    if(!finite(b.exchangeMass[participant])||!finite(b.exchangeEnergy[participant])||!finite(b.exchangeMomentum[participant])||!finite(b.filmKineticAdvection))
        return packetFailure(status,p,ErrorCode::Nonfinite,dE);
    for(int k=0;k<Ns;++k)if(!finite(b.exchangeSpecies[participant][k]))return packetFailure(status,p,ErrorCode::Nonfinite,k);
    if(bit==ConsumeGas)*gas=g;else if(bit==ConsumeSolid)*solid=s;else *film=f;
    budget=b;return true;
}
inline std::string packetError(const ExchangePacket& p,const DeviceStatus& status) {
    return "packet face "+std::to_string(p.face)+" stage "+std::to_string(p.stage)+" local index "+std::to_string(status.index)
        +" error "+std::to_string(status.code)+" value "+std::to_string(status.value);
}
inline bool validatePacket(const ExchangePacket& p,const PhysicsConfig& config,std::string& error) {
    DeviceStatus status{};if(!validatePacketMath(p,config.tolerances,&status)){error=packetError(p,status);return false;}
    error.clear();return true;
}
inline bool consumePacket(ExchangePacket& p,unsigned bit,GasQ* gas,SolidQ* solid,FilmQ* film,Budget& budget,std::string& error) {
    DeviceStatus status{};if(!validatePacketMath(p,Tolerances{},&status)||!applyPacketDelta(p,bit,gas,solid,film,budget,&status)){
        error=packetError(p,status);return false;
    }
    p.consumerMask|=bit;error.clear();return true;
}
inline bool consumeFilmPressureVolume(FilmPressureVolumeUpdate& u,FilmQ& q,Budget& budget,std::string& error) {
    const std::string prefix="film storage face "+std::to_string(u.face)+" stage "+std::to_string(u.stage)+": ";
    if(u.consumed){error=prefix+"duplicate pressure-volume storage conversion";return false;}
    if(u.filmFace<0||u.stage<0||u.stage>1||!finite(u.oldPV)||!finite(u.newPV)){error=prefix+"invalid identity or pressure-volume values";return false;}
    FilmQ next=q;const Real delta=filmPressureVolumeDelta(u);next.enthalpy+=delta;
    const Real nextBudget=budget.filmPressureVolume+delta;
    if(!validFilmInventory(next)||!finite(nextBudget)){error=prefix+"invalid candidate inventory";return false;}
    q=next;budget.filmPressureVolume=nextBudget;u.consumed=true;error.clear();return true;
}

CHMT_HD inline bool applyPacketTransaction(ExchangePacket& packet,GasQ* gas,SolidQ* solid,FilmQ* film,FilmPressureVolumeUpdate* storage,Budget& budget,const Tolerances& tolerances,DeviceStatus* status) {
    if(!validatePacketMath(packet,tolerances,status))return false;
    const unsigned required=requiredConsumers(packet.kind);
    if(packet.consumerMask!=0)return packetFailure(status,packet,ErrorCode::Duplicate,packet.consumerMask);
    if(required&ConsumeParticle)return packetFailure(status,packet,ErrorCode::Unsupported,ConsumeParticle);
    if(((required&ConsumeGas)&&!gas)||((required&ConsumeSolid)&&!solid)||((required&ConsumeFilm)&&!film))return packetFailure(status,packet,ErrorCode::InvalidInput,-1);
    if(storage&&(!(required&ConsumeFilm)||storage->consumed||storage->filmFace!=packet.filmFace||storage->face!=packet.face
        ||storage->step!=packet.step||storage->geometry!=packet.geometry||storage->stage!=packet.stage||!finite(storage->oldPV)||!finite(storage->newPV)))
        return packetFailure(status,packet,ErrorCode::Packet,-1);
    GasQ g=gas?*gas:GasQ{};SolidQ s=solid?*solid:SolidQ{};FilmQ f=film?*film:FilmQ{};
    Budget b=budget;ExchangePacket p=packet;
    for(unsigned bit=ConsumeGas;bit<=ConsumeFilm;bit<<=1)if(required&bit){
        if(!applyPacketDelta(p,bit,&g,&s,&f,b,status,storage!=nullptr))return false;
        p.consumerMask|=bit;
    }
    if(storage){
        // Evaluate the same storage identity with its thermal part grouped first;
        // this avoids stranding a pV roundoff remainder in a completely dry cell.
        f.enthalpy=(film->enthalpy-storage->oldPV)+packetDelta(packet).film.enthalpy+storage->newPV;
        b.filmPressureVolume+=filmPressureVolumeDelta(*storage);
        if(!validFilmInventory(f)||!finite(b.filmPressureVolume))return packetFailure(status,packet,ErrorCode::Inventory,f.enthalpy);
    }
    if(required&ConsumeGas)*gas=g;
    if(required&ConsumeSolid)*solid=s;
    if(required&ConsumeFilm)*film=f;
    budget=b;packet=p;if(storage)storage->consumed=true;return true;
}
inline bool consumePacketTransaction(ExchangePacket& packet,GasQ* gas,SolidQ* solid,FilmQ* film,FilmPressureVolumeUpdate* storage,Budget& budget,std::string& error) {
    DeviceStatus status{};
    if(!applyPacketTransaction(packet,gas,solid,film,storage,budget,Tolerances{},&status)){error=packetError(packet,status);return false;}
    error.clear();return true;
}
} // namespace chmt
#endif
