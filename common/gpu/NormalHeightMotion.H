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
struct NormalHeightPointMotionDefinition {
    Topology topology;
    std::vector<int> owner,wallFaces,offsets,neighbours;
    std::vector<Point> points,reference;
    std::vector<unsigned char> fixed;
    double geometricTolerance=0;
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
// One immutable Dirichlet mask feeds both constraint assembly and the harmonic
// operator. Coordinates and boundary displacement remain borrowed on device.
class NormalHeightPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit NormalHeightPointMotion(const NormalHeightPointMotionDefinition&);
    ~NormalHeightPointMotion();
    HarmonicDeviceView solveDevice(const NormalHeightPointMotionInput*,void* stream=nullptr);
    NormalHeightMotionView constraints()const;
    // Compatibility publication for the still-host outer geometry controller.
    // Only changing boundary displacement is uploaded; constraint coefficients
    // and tolerance are never downloaded and re-uploaded between operators.
    NormalHeightPointMotionResult solve(const std::vector<Point>& displacement);
};
}}}
#endif
