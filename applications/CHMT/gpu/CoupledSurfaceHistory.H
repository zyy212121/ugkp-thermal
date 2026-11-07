#ifndef CHMT_COUPLED_SURFACE_HISTORY_H
#define CHMT_COUPLED_SURFACE_HISTORY_H
#include "coupling/IntervalContracts.H"
#include "../../../common/Diagnostics.H"
namespace chmt {
// Accepted physical history stays resident until a consumer publishes it.
// Raw fluxes and unweighted stage packets remain device-only.
struct CoupledSurfaceDownload {
    GasPrimitive bulk{};GasGradient gradient{};Vec3 traction{};
    WallFaceSample acceptedWall{};
    ExchangePacket primary{},pore{};
    Real radiationEnergy=0,particleWallEnergy=0;unsigned radiationReceiver=0;
    int failed=0;
};
// Matching and failure status are consumed by the material history. Full node
// diagnostics remain resident until the gas-window endpoint is published.
struct CoupledWallDownload {
    GasWallMatchingSample matching{};
    int transportCode=0;
    ugkwp::gaswall::WallStatus status{};
};
struct CoupledRecordFailure {int wall=0,owner=0;};
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
// Existing transaction validity decisions remain on the producer stream.
// Wall failures take precedence over owner failures, and each category retains
// its first face. The existing sticky diagnostic prevents later gas consumers
// from advancing a rejected transaction without a per-microstep host read.
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
__global__ void resetCoupledRecordFailure(CoupledRecordFailure* failure,int count){
    if(!blockIdx.x&&!threadIdx.x){failure->wall=count;failure->owner=count;}
}
__global__ void validateCoupledRecordKernel(const CoupledRecordFailure* failure,int count,
    const CoupledSurfaceDownload* records,const CoupledWallDownload* walls,const int* faces,
    int solids,int films,ugkp::diagnostics::DeviceContext* context){
    if(!blockIdx.x&&!threadIdx.x)reportCoupledRecordFailure(*failure,count,records,walls,faces,solids,films,context);
}
#endif
}
#endif
