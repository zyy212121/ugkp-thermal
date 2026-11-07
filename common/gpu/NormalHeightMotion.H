#ifndef UGKP_GPU_NORMAL_HEIGHT_MOTION_H
#define UGKP_GPU_NORMAL_HEIGHT_MOTION_H
#include "PolyhedralGeometry.H"
#include "NormalHeightMotionMath.H"
namespace ugkp { namespace gpu { namespace geometry {
struct NormalHeightMotionInput {
    const Point* reference=nullptr;
    const Point* points=nullptr;
    const Point* displacement=nullptr;
    const unsigned char* fixed=nullptr;
    double geometricTolerance=0;
};
struct NormalHeightMotionControl {
    double minimumHeight=0,harmonicTolerance=0;
};
struct NormalHeightMotionView {
    const MotionPenalty* penalty=nullptr;
    const NormalHeightMotionControl* control=nullptr;
};
// Each point accumulates adjacent wall contributions in original interface
// order. Immutable cell vertices and wall identity are initialized once.
class ResidentNormalHeightMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentNormalHeightMotion(const Topology&,int pointCount,const std::vector<int>& owner,
        const std::vector<int>& wallFaces);
    ~ResidentNormalHeightMotion();
    NormalHeightMotionView view()const;
    void evaluate(const NormalHeightMotionInput*,void* stream=nullptr);
};
}}}
#endif
