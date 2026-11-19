#pragma once
#include "gpu/CoupledSurfaceHistory.H"
#include "../../../common/CompensatedTime.H"
#include "../../../common/gasTransport/GasDriveChange.H"
namespace chmt {



struct ResidentMaterialDriveRecord {
    const CoupledSurfaceDownload* surface=nullptr;
    const CoupledWallDownload* wall=nullptr;
    std::size_t faces=0,matchingFaces=0;
    CHMT_HD std::size_t bulkCount()const{return faces;}
    CHMT_HD std::size_t tractionCount()const{return faces;}
    CHMT_HD std::size_t gradientCount()const{return faces;}
    CHMT_HD std::size_t matchingCount()const{return matchingFaces;}
    CHMT_HD std::size_t receiverCount()const{return faces;}
    CHMT_HD const GasPrimitive& bulk(std::size_t i)const{return surface[i].bulk;}
    CHMT_HD const Vec3& traction(std::size_t i)const{return surface[i].traction;}
    CHMT_HD const GasGradient& gradient(std::size_t i)const{return surface[i].gradient;}
    CHMT_HD const GasWallMatchingSample& matching(std::size_t i)const{return wall[i].matching;}
    CHMT_HD unsigned receiver(std::size_t i)const{return surface[i].radiationReceiver;}
};
struct ResidentMaterialDriveInterval {
    ugkp::CompensatedTime begin{},end{};
    Real actualDt=0;
};
struct ResidentMaterialDriveHistory {
    const CoupledSurfaceDownload* surface=nullptr;
    const CoupledWallDownload* wall=nullptr;
    const ResidentMaterialDriveInterval* interval=nullptr;
    std::size_t frames=0,faces=0;


    void* producerStream=nullptr;


    const std::size_t* deviceFrames=nullptr;
    CHMT_HD ResidentMaterialDriveHistory resolved()const{
        auto view=*this;
#if defined(__CUDA_ARCH__)
        if(deviceFrames)view.frames=*deviceFrames;
#endif
        view.deviceFrames=nullptr;return view;
    }
    CHMT_HD ResidentMaterialDriveRecord record(std::size_t frame)const{
        const auto offset=frame*faces;
        return {surface?surface+offset:nullptr,wall?wall+offset:nullptr,faces,wall?faces:0};
    }
};


}
