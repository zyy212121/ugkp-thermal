#pragma once
#include "gpu/ResidentMaterialDrive.H"
#include "../../../common/IntervalSlice.H"
namespace chmt {




struct MaterialSliceAccumulators {
    ExchangePacket* packets=nullptr;
    Real* radiation=nullptr;
    Real* externalRadiation=nullptr;
};
struct ResidentMaterialSlice {
    ResidentMaterialDriveHistory history{};
    const ugkp::IntervalSlicePlan* plan=nullptr;
    MaterialSliceAccumulators used{};
    bool particles=false;
    const IntervalSliceRecordPart* selectedParts=nullptr;
    Real externalFraction=0;bool externalLast=false,hostSelected=false;
    CHMT_HD bool recordLast(std::size_t r)const{
        return hostSelected?selectedParts[r-plan->begin.record].last:
            ugkp::sliceRecordStop(history,*plan,r)==history.interval[r].end;
    }
    CHMT_HD Real recordFraction(std::size_t r)const{
        return hostSelected?selectedParts[r-plan->begin.record].fraction:
            ugkp::sliceRecordFraction(history,*plan,r);
    }
    CHMT_HD std::size_t slotCount()const{
        return plan->error==ugkp::IntervalSliceError::None?(plan->past-plan->begin.record)*history.faces*3:0;
    }
    CHMT_HD bool active(std::size_t slot)const{
        const auto local=slot%(history.faces*3);const unsigned kind=local%3;
        if(kind==0)return true;
        if(kind==2&&!particles)return false;
        const auto r=plan->begin.record+slot/(history.faces*3);
        return hasPacketTransfer(materialHistoryPacket(history.surface[r*history.faces+local/3],kind));
    }
    CHMT_HD ExchangePacket operator[](std::size_t slot)const{return packet(slot);}
    CHMT_HD ExchangePacket packet(std::size_t slot)const{
        const auto local=slot%(history.faces*3),r=plan->begin.record+slot/(history.faces*3);
        const auto p=materialHistoryPacket(history.surface[r*history.faces+local/3],local%3);
        if(recordLast(r))
            return subtractIntervalPacket(p,r==plan->begin.record?used.packets[local]:ExchangePacket{});
        return scaledIntervalPacket(p,recordFraction(r));
    }
};
struct ResidentMaterialSliceRow {
    ResidentMaterialSlice slice{};
    CHMT_HD int operator()(int i)const{
        if(std::size_t(i)>=slice.slotCount()||!slice.active(i))return -1;
        const auto packet=slice[std::size_t(i)];
        return (requiredConsumers(packet.kind)&ConsumeSolid)?packet.solidCell:-1;
    }
};
}
