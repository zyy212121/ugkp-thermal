#pragma once
#include "PolyhedralGeometry.H"
#include "../gasWall/WallGeometryTypes.H"
namespace ugkp { namespace gpu { namespace geometry {
struct WallMatchingDefinition {
    int cells=0;
    std::vector<int> faceOffsets,owner,neighbour,walls;
    std::vector<unsigned char> physicalWall,constrained,gasCell;
};
struct WallMatchingDeviceView {
    int rows=0;
    const ugkwp::gaswall::WallMatchingGeometry* geometry=nullptr;
    const ugkwp::gaswall::WallMatchingStatus* status=nullptr;
};






class ResidentWallMatching {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentWallMatching(const WallMatchingDefinition&,TopologyView deviceTopology,bool reconstructMatching);
    ~ResidentWallMatching();
    WallMatchingDeviceView view()const;
    void prepare(const Point* points,const CellResult* cells,
        const unsigned char* constrained=nullptr,const unsigned char* gasCell=nullptr,void* stream=nullptr);
};
}}}
