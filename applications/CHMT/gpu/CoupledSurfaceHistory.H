#ifndef CHMT_COUPLED_SURFACE_HISTORY_H
#define CHMT_COUPLED_SURFACE_HISTORY_H
#include "coupling/IntervalContracts.H"
#include "../../../common/Diagnostics.H"
namespace chmt {


struct CoupledSurfaceDownload {
    GasPrimitive bulk{};GasGradient gradient{};PhysicalVec3 traction{};
    WallFaceSample acceptedWall{};
    ExchangePacket primary{},pore{};
    EnergyReal radiationEnergy=0,particleWallEnergy=0;unsigned radiationReceiver=0;
    int failed=0;
};


struct CoupledWallDownload {
    GasWallMatchingSample matching{};
    int transportCode=0;
    ugkwp::gaswall::WallStatus status{};
};
CHMT_HD inline ExchangePacket materialHistoryPacket(const CoupledSurfaceDownload& r,unsigned slot){
    if(slot==0)return r.primary;
    if(slot==1)return r.pore;
    ExchangePacket p{};p.kind=ExchangeKind::ParticleWall;
    p.step=r.primary.step;p.geometry=r.primary.geometry;p.face=r.primary.face;
    p.stage=2;p.gasCell=r.primary.gasCell;p.solidCell=r.primary.solidCell;
    p.filmFace=r.primary.filmFace;p.consumerMask=ConsumeParticle;
    p.energy=p.conductive=r.particleWallEnergy;return p;
}
inline void appendCoupledHostPackets(std::vector<ExchangePacket>& result,const CoupledSurfaceDownload& surface,bool particles){
    result.push_back(surface.primary);
    if(hasPacketTransfer(surface.pore))result.push_back(surface.pore);
    if(particles){
        const auto contact=materialHistoryPacket(surface,2);
        if(hasPacketTransfer(contact))result.push_back(contact);
    }
}
struct CoupledRecordFailure {int wall,owner;};
CHMT_HD inline int coupledPacketOwnerFailure(const ExchangePacket& packet,int solids,int films){
    const auto consumers=requiredConsumers(packet.kind);
    if((consumers&ConsumeSolid)&&(packet.solidCell<0||packet.solidCell>=solids))return 3;
    if((consumers&ConsumeFilm)&&(packet.filmFace<0||packet.filmFace>=films))return 4;
    return 0;
}
CHMT_HD inline int coupledRecordOwnerFailure(const CoupledSurfaceDownload& record,int solids,int films){
    int error=coupledPacketOwnerFailure(record.primary,solids,films);if(error)return error;
    if(hasPacketTransfer(record.pore)){
        error=coupledPacketOwnerFailure(record.pore,solids,films);if(error)return error;
    }
    if(record.particleWallEnergy!=0&&(record.primary.solidCell<0||record.primary.solidCell>=solids))return 3;
    return 0;
}




CHMT_HD inline void reportCoupledRecordFailure(const CoupledRecordFailure& failure,int count,
    const CoupledSurfaceDownload* records,const CoupledWallDownload* walls,const int* faces,
    int solids,int films,ugkp::diagnostics::DeviceContext* context){
    using namespace ugkp::diagnostics;
    if(hasFatal(context))return;
    if(failure.wall<count){
        const int slot=failure.wall;const auto& item=records[slot];
        if(item.failed==2&&walls){const auto& wall=walls[slot];
            auto record=makeRecord(Code::InvalidPhysicalState,2632,26,0,EntityKind::Face,faces[slot],wall.status.node,
                wall.status.residual,double(wall.status.iteration),double(wall.transportCode));
            record.metadata=std::uint32_t(wall.status.code);recordFatal(context,record);
        }else recordFatal(context,makeRecord(Code::InvalidPhysicalState,2631,26,0,EntityKind::Face,faces[slot],-1,item.failed));
    }else if(failure.owner<count){
        const int slot=failure.owner;const auto& item=records[slot];
        const int code=coupledRecordOwnerFailure(item,solids,films);
        const auto& packet=coupledPacketOwnerFailure(item.primary,solids,films)?item.primary:
            (hasPacketTransfer(item.pore)&&coupledPacketOwnerFailure(item.pore,solids,films)?item.pore:item.primary);
        recordFatal(context,makeRecord(Code::InvalidIndex,code==3?2633:2634,26,0,EntityKind::Face,faces[slot],-1,
            code==3?packet.solidCell:packet.filmFace,code==3?solids:films,0,Quantity::Index));
    }
}
#if defined(__CUDACC__)
static __global__ void resetCoupledRecordFailure(CoupledRecordFailure* failure,int count){
    if(!blockIdx.x&&!threadIdx.x){failure->wall=count;failure->owner=count;}
}

#endif
}
#endif
