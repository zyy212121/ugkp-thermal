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


class ResidentNormalHeightMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentNormalHeightMotion(const Topology&,int pointCount,const std::vector<int>& owner,
        const std::vector<int>& wallFaces);
    ~ResidentNormalHeightMotion();
    NormalHeightMotionView view()const;
    void evaluate(const NormalHeightMotionInput*,void* stream=nullptr);
};
struct NormalHeightPointMotionDefinition {
    Topology topology;
    std::vector<int> owner,wallFaces,offsets,neighbours;
    std::vector<Point> points,reference;
    std::vector<unsigned char> fixed;
    double geometricTolerance=0;
    int harmonicMaximumIterations=20000;
};
struct NormalHeightPointMotionInput {
    const Point* reference=nullptr;
    const Point* points=nullptr;
    const Point* displacement=nullptr;
    double geometricTolerance=0;
};
struct NormalHeightPointMotionResult {
    HarmonicResult motion;
    NormalHeightMotionControl constraints;
};


class NormalHeightPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit NormalHeightPointMotion(const NormalHeightPointMotionDefinition&);
    ~NormalHeightPointMotion();
    HarmonicDeviceView solveDevice(const NormalHeightPointMotionInput*,void* stream=nullptr);
    NormalHeightMotionView constraints()const;
    HarmonicDeviceView deviceView()const;
    const Point* initialPoints()const;
    const Point* referencePoints()const;
    const unsigned char* fixedPoints()const;



    NormalHeightPointMotionResult solve(const std::vector<Point>& displacement);
};
}}}
#endif
