#ifndef CHMT_COUPLING_EXCHANGELEDGER_IMPLEMENTATION
#define CHMT_COUPLING_EXCHANGELEDGER_IMPLEMENTATION
#include "ExchangeLedger.H"
#include "core/AuditArithmetic.H"
namespace chmt {
CHMT_HD inline bool packetFailure(DeviceStatus* status,const ExchangePacket& p,ErrorCode code,Real value,ugkp::diagnostics::DeviceContext* diagnostics=nullptr,ugkp::diagnostics::Code detail=ugkp::diagnostics::Code::InvalidPhysicalState) {
    // A persistent face is uint64; index carries local film/gas/solid location,
    // while the host diagnostic below retains the full persistent identifier.
    int index=p.filmFace>=0?p.filmFace:(p.gasCell>=0?p.gasCell:p.solidCell);
    if(diagnostics){
        using namespace ugkp::diagnostics;
        const Code fatalCode=code==ErrorCode::Nonfinite?Code::NonFinite:
            code==ErrorCode::InvalidInput?Code::MissingOperator:detail;
        recordFatal(diagnostics,makeRecord(fatalCode,4301,38,p.stage,EntityKind::Face,index,-1,value));
    }
    return failStatus(status,code,index,value,ErrorLocation::Face);
}
CHMT_HD inline bool packetClose(Real value,Real target,Real absolute,Real relative,Real scale) {
    if(!finite(value)||!finite(target)||!finite(absolute)||!finite(relative)||!finite(scale)||absolute<0||relative<0||scale<0)return false;
    const Real residual=auditDifference(value,target);
    const Real tolerance=auditTolerance(absolute,relative,maxValue(scale,maxValue(absValue(value),absValue(target))));
    return finite(residual)&&finite(tolerance)&&absValue(residual)<=tolerance;
}
CHMT_HD inline void packetResidual(ugkp::diagnostics::DeviceContext* diagnostics,Real observed,Real reference,int index=-1,int stage=0) {
    using namespace ugkp::diagnostics;
    recordWarning(diagnostics,makeRecord(Code::ConservationResidual,4302,38,stage,
        EntityKind::Face,index,-1,observed,reference,observed,Quantity::Conservation));
}
CHMT_HD inline bool validatePacketMath(const ExchangePacket& p,const Tolerances& tol,DeviceStatus* status) {
    const unsigned required=requiredConsumers(p.kind);
    if(required==0||p.stage<0||p.stage>1||(p.consumerMask&~required)!=0)
        return packetFailure(status,p,ErrorCode::Packet,p.consumerMask,tol.diagnostics);
    if(((required&ConsumeGas)&&p.gasCell<0)||((required&ConsumeSolid)&&p.solidCell<0)
        ||((required&ConsumeFilm)&&p.filmFace<0)||((required&ConsumeParticle)&&p.particleIndex<0))
        return packetFailure(status,p,ErrorCode::Packet,-1,tol.diagnostics,ugkp::diagnostics::Code::InvalidIndex);
    const Real fields[]={p.mass,p.momentum.x,p.momentum.y,p.momentum.z,p.energy,p.conductive,
        p.advective,p.pressureWork,p.viscousWork,p.radiation,p.liquidKineticAdvection};
    for(int field=0;field<11;++field)if(!finite(fields[field]))
        return packetFailure(status,p,ErrorCode::Nonfinite,fields[field],tol.diagnostics);
    Real sum=0,scale=0,donor=0,donorScale=0;
    for(int s=0;s<Ns;++s){
        if(!finite(p.species[s])||!finite(p.pore[s])||!finite(p.poreSweep[s]))return packetFailure(status,p,ErrorCode::Nonfinite,!finite(p.species[s])?p.species[s]:!finite(p.pore[s])?p.pore[s]:p.poreSweep[s],tol.diagnostics);
        if(p.kind!=ExchangeKind::PoreGas&&p.poreSweep[s]!=0)return packetFailure(status,p,ErrorCode::Packet,p.poreSweep[s],tol.diagnostics);
        sum=auditAdd(sum,p.species[s]);scale=auditAdd(scale,absValue(p.species[s]));donor=auditAdd(donor,p.pore[s]);donorScale=auditAdd(donorScale,absValue(p.pore[s]));
    }
    for(int c=0;c<Nc;++c){if(!finite(p.condensed[c]))return packetFailure(status,p,ErrorCode::Nonfinite,p.condensed[c],tol.diagnostics);donor=auditAdd(donor,p.condensed[c]);donorScale=auditAdd(donorScale,absValue(p.condensed[c]));}
    // Nonfinite audit reductions are reported below; physical components were checked above.
    // Deposition has no gas receiver and therefore no fictitious gas-species
    // equivalent. Its actual condensed/pore receiver still closes below.
    if(p.kind==ExchangeKind::ParticleWall){
        if(scale!=0)return packetFailure(status,p,ErrorCode::Composition,scale,tol.diagnostics);
    }else if(!packetClose(sum,p.mass,tol.absoluteMass,tol.relativeMass,scale))packetResidual(tol.diagnostics,sum,p.mass,p.filmFace,p.stage);
    if((required&ConsumeSolid)&&!packetClose(donor,p.mass,tol.absoluteMass,tol.relativeMass,donorScale))
        packetResidual(tol.diagnostics,donor,p.mass,p.solidCell,p.stage);
    if(!(required&ConsumeSolid)&&donorScale!=0)return packetFailure(status,p,ErrorCode::Packet,donorScale,tol.diagnostics);
    if(p.kind==ExchangeKind::PoreGas){
        for(int c=0;c<Nc;++c)if(p.condensed[c]!=0)return packetFailure(status,p,ErrorCode::Packet,p.condensed[c],tol.diagnostics);
        for(int s=0;s<Ns;++s)if(!packetClose(p.pore[s],p.species[s],tol.absoluteMass,tol.relativeMass,absValue(p.species[s])))
            packetResidual(tol.diagnostics,p.pore[s],p.species[s],p.solidCell,p.stage);
    }
    if(!(required&ConsumeFilm)&&p.liquidKineticAdvection!=0)return packetFailure(status,p,ErrorCode::Packet,p.liquidKineticAdvection,tol.diagnostics);
    const Real parts=auditAdd(auditAdd(auditAdd(auditAdd(p.conductive,p.advective),p.pressureWork),p.viscousWork),p.radiation);
    const Real partScale=auditAdd(auditAdd(auditAdd(auditAdd(absValue(p.conductive),absValue(p.advective)),absValue(p.pressureWork)),absValue(p.viscousWork)),absValue(p.radiation));
    // Overflow of the audit sum or absolute scale cannot invalidate finite packet fields.
    if(!packetClose(parts,p.energy,tol.absoluteEnergy,tol.relativeEnergy,partScale))packetResidual(tol.diagnostics,parts,p.energy,p.filmFace,p.stage);
    return true;
}
CHMT_HD inline bool inventoryFailure(ugkp::diagnostics::DeviceContext* diagnostics,Real value) {
    if(diagnostics){
        using namespace ugkp::diagnostics;
        recordFatal(diagnostics,makeRecord(finite(value)?Code::InvalidPhysicalState:Code::NonFinite,
            4303,38,0,EntityKind::None,-1,-1,value));
    }
    return false;
}
CHMT_HD inline bool validGasInventory(const GasQ& q,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if(!finite(q.mass)||q.mass<=0)return inventoryFailure(diagnostics,q.mass);
    if(!finite(q.momentum))return inventoryFailure(diagnostics,std::numeric_limits<Real>::quiet_NaN());
    if(!finite(q.energy))return inventoryFailure(diagnostics,q.energy);
    Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0)return inventoryFailure(diagnostics,q.species[s]);sum=auditAdd(sum,q.species[s]);}
    if(!packetClose(sum,q.mass,1e-14,1e-10,sum))packetResidual(diagnostics,sum,q.mass);
    return true;
}
CHMT_HD inline bool validSolidInventory(const SolidQ& q,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if(!finite(q.energy))return inventoryFailure(diagnostics,q.energy);
    if(!finite(q.porosity)||q.porosity<0||q.porosity>=1)return inventoryFailure(diagnostics,q.porosity);
    Real mass=0;for(int c=0;c<Nc;++c){if(!finite(q.condensed[c])||q.condensed[c]<0)return inventoryFailure(diagnostics,q.condensed[c]);mass=auditAdd(mass,q.condensed[c]);}
    for(int s=0;s<Ns;++s){if(!finite(q.pore[s])||q.pore[s]<0)return inventoryFailure(diagnostics,q.pore[s]);mass=auditAdd(mass,q.pore[s]);}
    for(int r=0;r<Nr;++r)if(!finite(q.progress[r])||q.progress[r]<0)return inventoryFailure(diagnostics,q.progress[r]);
    if(!finite(mass))packetResidual(diagnostics,mass,0);
    if(mass==0&&q.energy!=0)packetResidual(diagnostics,q.energy,0);
    return true;
}
CHMT_HD inline bool validFilmInventory(const FilmQ& q,bool pendingStorage=false,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if(!finite(q.mass)||q.mass<0)return inventoryFailure(diagnostics,q.mass);
    if(!finite(q.enthalpy))return inventoryFailure(diagnostics,q.enthalpy);
    Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0)return inventoryFailure(diagnostics,q.species[s]);sum=auditAdd(sum,q.species[s]);}
    if(!packetClose(sum,q.mass,1e-14,1e-10,sum))packetResidual(diagnostics,sum,q.mass);
    if(!pendingStorage&&q.mass==0&&q.enthalpy!=0)packetResidual(diagnostics,q.enthalpy,0);
    return true;
}
CHMT_HD inline bool applyPacketDelta(const ExchangePacket& p,unsigned bit,GasQ* gas,SolidQ* solid,FilmQ* film,DeviceStatus* status,bool pendingFilmStorage,ugkp::diagnostics::DeviceContext* diagnostics) {
    if(bit==0||(bit&(bit-1))!=0||(bit&requiredConsumers(p.kind))==0)return packetFailure(status,p,ErrorCode::Packet,bit,diagnostics);
    if(p.consumerMask&bit)return packetFailure(status,p,ErrorCode::Duplicate,bit,diagnostics);
    const PacketDelta d=packetDelta(p);
    GasQ g{};SolidQ s{};FilmQ f{};
    if(bit==ConsumeGas){
        if(!gas)return packetFailure(status,p,ErrorCode::InvalidInput,bit,diagnostics);
        g=*gas+d.gas;if(!validGasInventory(g,diagnostics))return packetFailure(status,p,ErrorCode::Inventory,g.mass,diagnostics);
    }else if(bit==ConsumeSolid){
        if(!solid)return packetFailure(status,p,ErrorCode::InvalidInput,bit,diagnostics);
        s=*solid;for(int c=0;c<Nc;++c)s.condensed[c]+=d.solid.condensed[c];
        for(int k=0;k<Ns;++k)s.pore[k]+=d.solid.pore[k];
        s.energy+=d.solid.energy;
        if(!validSolidInventory(s,diagnostics))return packetFailure(status,p,ErrorCode::Inventory,s.energy,diagnostics);
    }else if(bit==ConsumeFilm){
        if(!film)return packetFailure(status,p,ErrorCode::InvalidInput,bit,diagnostics);
        f=*film;f.mass+=d.film.mass;f.enthalpy+=d.film.enthalpy;
        for(int k=0;k<Ns;++k)f.species[k]+=d.film.species[k];
        if(!validFilmInventory(f,pendingFilmStorage,diagnostics))return packetFailure(status,p,ErrorCode::Inventory,f.mass,diagnostics);
    }else return packetFailure(status,p,ErrorCode::Unsupported,bit,diagnostics);
    if(bit==ConsumeGas)*gas=g;else if(bit==ConsumeSolid)*solid=s;else *film=f;
    return true;
}
inline std::string packetError(const ExchangePacket& p,const DeviceStatus& status) {
    return "packet face "+std::to_string(p.face)+" stage "+std::to_string(p.stage)+" local index "+std::to_string(status.index)
        +" error "+std::to_string(status.code)+" value "+std::to_string(status.value);
}
inline bool validatePacket(const ExchangePacket& p,const PhysicsConfig& config,std::string& error) {
    Tolerances tol=config.tolerances;tol.diagnostics=config.diagnostics?config.diagnostics:tol.diagnostics;
    DeviceStatus status{};if(!validatePacketMath(p,tol,&status)){error=packetError(p,status);return false;}
    error.clear();return true;
}
inline bool consumePacket(ExchangePacket& p,unsigned bit,GasQ* gas,SolidQ* solid,FilmQ* film,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics) {
    Tolerances tol;tol.diagnostics=diagnostics;
    DeviceStatus status{};if(!validatePacketMath(p,tol,&status)||!applyPacketDelta(p,bit,gas,solid,film,&status,false,diagnostics)){
        error=packetError(p,status);return false;
    }
    p.consumerMask|=bit;error.clear();return true;
}
inline bool consumeFilmPressureVolume(FilmPressureVolumeUpdate& u,FilmQ& q,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics) {
    const std::string prefix="film storage face "+std::to_string(u.face)+" stage "+std::to_string(u.stage)+": ";
    if(u.consumed){inventoryFailure(diagnostics,1);error=prefix+"duplicate pressure-volume storage conversion";return false;}
    if(u.filmFace<0||u.stage<0||u.stage>1||!finite(u.oldPV)||!finite(u.newPV)){inventoryFailure(diagnostics,!finite(u.oldPV)?u.oldPV:u.newPV);error=prefix+"invalid identity or pressure-volume values";return false;}
    FilmQ next=q;const Real delta=filmPressureVolumeDelta(u);next.enthalpy+=delta;
    if(!validFilmInventory(next,false,diagnostics)){error=prefix+"invalid candidate inventory";return false;}
    q=next;u.consumed=true;error.clear();return true;
}

CHMT_HD inline bool applyPacketTransaction(ExchangePacket& packet,GasQ* gas,SolidQ* solid,FilmQ* film,FilmPressureVolumeUpdate* storage,const Tolerances& tolerances,DeviceStatus* status) {
    if(!validatePacketMath(packet,tolerances,status))return false;
    const unsigned required=requiredConsumers(packet.kind);
    if(packet.consumerMask!=0)return packetFailure(status,packet,ErrorCode::Duplicate,packet.consumerMask,tolerances.diagnostics);
    if(required&ConsumeParticle)return packetFailure(status,packet,ErrorCode::Unsupported,ConsumeParticle,tolerances.diagnostics);
    if(((required&ConsumeGas)&&!gas)||((required&ConsumeSolid)&&!solid)||((required&ConsumeFilm)&&!film))return packetFailure(status,packet,ErrorCode::InvalidInput,-1,tolerances.diagnostics);
    if(storage&&(!finite(storage->oldPV)||!finite(storage->newPV)))
        return packetFailure(status,packet,ErrorCode::Nonfinite,!finite(storage->oldPV)?storage->oldPV:storage->newPV,tolerances.diagnostics);
    if(storage&&(!(required&ConsumeFilm)||storage->consumed||storage->filmFace!=packet.filmFace||storage->face!=packet.face
        ||storage->step!=packet.step||storage->geometry!=packet.geometry||storage->stage!=packet.stage))
        return packetFailure(status,packet,ErrorCode::Packet,-1,tolerances.diagnostics);
    GasQ g=gas?*gas:GasQ{};SolidQ s=solid?*solid:SolidQ{};FilmQ f=film?*film:FilmQ{};
    ExchangePacket p=packet;
    for(unsigned bit=ConsumeGas;bit<=ConsumeFilm;bit<<=1)if(required&bit){
        if(!applyPacketDelta(p,bit,&g,&s,&f,status,storage!=nullptr,tolerances.diagnostics))return false;
        p.consumerMask|=bit;
    }
    if(storage){
        // Evaluate the same storage identity with its thermal part grouped first;
        // this avoids stranding a pV roundoff remainder in a completely dry cell.
        f.enthalpy=(film->enthalpy-storage->oldPV)+packetDelta(packet).film.enthalpy+storage->newPV;
        if(!validFilmInventory(f,false,tolerances.diagnostics))return packetFailure(status,packet,ErrorCode::Inventory,f.enthalpy,tolerances.diagnostics);
    }
    if(required&ConsumeGas)*gas=g;
    if(required&ConsumeSolid)*solid=s;
    if(required&ConsumeFilm)*film=f;
    packet=p;if(storage)storage->consumed=true;return true;
}
inline bool consumePacketTransaction(ExchangePacket& packet,GasQ* gas,SolidQ* solid,FilmQ* film,FilmPressureVolumeUpdate* storage,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics) {
    Tolerances tol;tol.diagnostics=diagnostics;
    DeviceStatus status{};
    if(!applyPacketTransaction(packet,gas,solid,film,storage,tol,&status)){error=packetError(packet,status);return false;}
    error.clear();return true;
}
} // namespace chmt
#endif
