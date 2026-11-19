#pragma once
#include "gpu/ResidentMaterialDrive.H"
#include <limits>
namespace chmt {
enum class StreamedMaterialPhase : int {Select,Held,Search,Transport,BeginPackets,Packets,Finish,Commit,Complete};
struct StreamedMaterialProgress {
    StreamedMaterialPhase phase=StreamedMaterialPhase::Select;
    std::size_t held=0,search=0,packetRecord=0;
    std::size_t requestFrame=std::numeric_limits<std::size_t>::max();
    int packetSlot=-1;
    std::size_t packetPast=0;
};



struct StreamedMaterialChunk {
    ResidentMaterialDriveHistory history{};
    std::size_t first=0;
    const int *packetOffsets=nullptr,*packetIndices=nullptr;
    CHMT_HD bool contains(std::size_t frame)const{
        return frame>=first&&frame-first<history.frames;
    }
};
}
