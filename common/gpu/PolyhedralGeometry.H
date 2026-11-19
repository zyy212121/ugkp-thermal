#ifndef UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#define UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#include "PolyhedralGeometryMath.H"
#include "PointMotionMath.H"
#include "LinearTrajectoryMath.H"
#include "GeometryBoundaryMath.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu { class PinnedTransferQueue; namespace geometry {


struct Topology {
    std::vector<int> faceOffsets,facePoints,cellOffsets,cellFaces,cellSigns,vertexOffsets,vertices;
};
struct WallDistanceResult {double distance=0;int invalidFace=-1;};
struct BoundaryMotionFace {int face;double previousOffset,offset,thickness;Point supportVelocity;};
struct BoundaryMotionPoint {Point displacement,direction;double weight=0,thickness=0;};
struct BoundaryMotionDeviceInput {
    const Point* area=nullptr;
    const BoundaryMotionFace* faces=nullptr;
    double interval=0;
};


class ResidentBoundaryMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentBoundaryMotion(const Topology&,int pointCount,const std::vector<int>& faces,
        const std::vector<unsigned char>& sideFaces);
    ~ResidentBoundaryMotion();
    const BoundaryMotionPoint* points()const;
    void evaluate(const BoundaryMotionDeviceInput*,void* stream=nullptr);
};
struct RebuildResult {
    std::vector<FaceResult> faces;std::vector<CellResult> cells;std::vector<WallDistanceResult> distance;
};



struct PolyhedralGeometryDeviceView {
    int faces=0,cells=0;
    TopologyView topology{};
    const FaceResult* face=nullptr;
    const CellResult* cell=nullptr;
    const Point* area=nullptr;
    const Point* centre=nullptr;
    const WallDistanceResult* distance=nullptr;
    const SweepResult* sweptFaces=nullptr;
};




class ResidentPolyhedralGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;



    friend class PolyhedralGeometry;
    void configure(const Topology&,const std::vector<int>&,PinnedTransferQueue*);
public:
    ResidentPolyhedralGeometry(const Topology&,const std::vector<int>& wallFaces);
    ~ResidentPolyhedralGeometry();
    PolyhedralGeometryDeviceView view() const;
    void rebuild(const Point* devicePoints,void* stream=nullptr);
    void sweep(const Point* deviceOld,const Point* deviceEndpoint,void* stream=nullptr);
};
struct TrajectoryDiagnostic {
    double observed=0,limit=0;
    TrajectoryWarning kind=TrajectoryWarning::FanMargin;
};
struct TrajectoryCellDiagnostic {
    unsigned int count=0;
    TrajectoryDiagnostic first{};
};
struct TrajectoryStatus {

    int moving=0,firstFailure=-1;
    TrajectoryFailure failure=TrajectoryFailure::None;
};
struct TrajectoryDeviceView {
    int cells=0;
    const TrajectoryStatus* status=nullptr;
    const TrajectoryFailure* failure=nullptr;
    const TrajectoryCellDiagnostic* diagnostics=nullptr;
};





class ResidentLinearTrajectory {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentLinearTrajectory(const Topology&,int pointCount);
    ~ResidentLinearTrajectory();
    TrajectoryDeviceView view() const;
    void evaluate(const Point* old,const Point* endpoint,void* stream=nullptr);
};
enum class GeometryBoundaryFailure : int {
    None,PeriodicArea,PeriodicVertices,EmptyNormal,EmptyTangential,EmptyDuplicate
};
struct GeometryBoundaryStatus {
    unsigned long long first=0;
    GeometryBoundaryFailure failure=GeometryBoundaryFailure::None;
    int index=-1;
    double scale=0;
};







class ResidentGeometryBoundary {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentGeometryBoundary(const Topology&,TopologyView deviceTopology,int pointCount,
        const std::vector<int>& periodicPartner,int emptyAxis,const std::vector<int>& emptyMate);
    ~ResidentGeometryBoundary();
    const GeometryBoundaryStatus* periodicStatus()const;
    const GeometryBoundaryStatus* emptyTrajectoryStatus()const;
    const GeometryBoundaryStatus* emptyUniquenessStatus()const;
    void periodic(const Point* points,const FaceResult* faces,void* stream=nullptr);
    void emptyTrajectory(const Point* old,const Point* endpoint,void* stream=nullptr);
    void emptyUniqueness(const Point* points,void* stream=nullptr);
};
class PolyhedralGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    PolyhedralGeometry();~PolyhedralGeometry();
    RebuildResult rebuild(const Topology&,const std::vector<Point>& points,const std::vector<int>& wallFaces);
    std::vector<BoundaryMotionPoint> boundaryMotion(const Topology&,int pointCount,
        const std::vector<Point>& faceArea,const std::vector<BoundaryMotionFace>&,
        const std::vector<unsigned char>& sideFaces,double interval);
    std::vector<FaceResult> faces(const Topology&,const std::vector<Point>&);
    std::vector<CellResult> cells(const Topology&,const std::vector<Point>&,const std::vector<Point>& area);
    std::vector<WallDistanceResult> wallDistance(const Topology&,const std::vector<Point>& points,
        const std::vector<Point>& centres,const std::vector<int>& wallFaces);
    std::vector<EdgeResult> surfaceEdges(const Topology&,const std::vector<Point>& old,
        const std::vector<Point>& endpoint,const std::vector<SurfaceEdge>& edges,bool empty,
        int prescribedSegments=0,double meshScale=0);
    std::vector<SweepResult> sweep(const Topology&,const std::vector<Point>& old,const std::vector<Point>& endpoint);
};


class SweepLinearization {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    SweepLinearization(const Topology&,const std::vector<Point>& old,
        const std::vector<int>& faces,const std::vector<double>& scales,
        const std::vector<PlanarConstraint>& planar);
    ~SweepLinearization();
    void endpoint(const std::vector<Point>&);
    std::vector<Polynomial> planarity(const std::vector<Point>& endpoint);
    std::vector<double> column(const std::vector<Point>& mode);
    void resetCorrection();
    void accumulate(const std::vector<Point>& mode,double coefficient);
    std::vector<Point> correction();
};



struct SweepLinearizationDeviceInput {
    const Point* old=nullptr;
    const Point* endpoint=nullptr;
    const Point* mode=nullptr;
    const double* scales=nullptr;
    const PlanarConstraint* planar=nullptr;
    double coefficient=0;
    bool active=true;
    int planarCount=-1;
};
struct SweepLinearizationDeviceView {
    const Polynomial* polynomials=nullptr;
    const double* column=nullptr;
    const Point* correction=nullptr;
};
class ResidentSweepLinearization {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentSweepLinearization(const Topology&,int pointCount,
        const std::vector<int>& faces,int planarCount);
    ~ResidentSweepLinearization();
    SweepLinearizationDeviceView view()const;


    void planarity(const SweepLinearizationDeviceInput*,void* stream=nullptr);
    void column(const SweepLinearizationDeviceInput*,void* stream=nullptr);
    void resetCorrection(void* stream=nullptr);
    void accumulate(const SweepLinearizationDeviceInput*,void* stream=nullptr);
};
struct MinimumNormResult {
    std::vector<double> delta,rhs;
    int rank=0;
    bool initialized=false,valid=false;
};


class DenseMinimumNorm {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    DenseMinimumNorm();~DenseMinimumNorm();
    MinimumNormResult solve(int rows,int columns,const std::vector<double>& matrix,
        const std::vector<double>& rhs,double rankTolerance);
};


struct MinimumNormDeviceProblem {
    int rows=0,columns=0;
    double rankTolerance=0;
    const double* matrix=nullptr;
    const double* rhs=nullptr;
};
struct MinimumNormDeviceStatus {
    double largest=0,length=0;
    int rank=0,active=0,error=0;
};
struct MinimumNormDeviceView {
    const double* delta=nullptr;
    const double* rhs=nullptr;
    const MinimumNormDeviceStatus* status=nullptr;
};
class ResidentDenseMinimumNorm {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentDenseMinimumNorm(int rowCapacity,int columnCapacity);
    ~ResidentDenseMinimumNorm();
    MinimumNormDeviceView view()const;



    void solve(const MinimumNormDeviceProblem*,void* stream=nullptr);
};
struct HarmonicResult {
    std::vector<Point> points;
    double residual=0;
    int iterations=0,errorPoint=-1,errorKind=0;
};
struct HarmonicDeviceStatus {
    double residual=0;
    int iterations=0,errorPoint=-1,errorKind=0;
};
struct HarmonicDeviceView {
    const Point* points=nullptr;
    const HarmonicDeviceStatus* status=nullptr;
};
struct HarmonicMotionDeviceInput {
    const Point* points=nullptr;
    const Point* boundary=nullptr;
    const MotionPenalty* penalties=nullptr;
    const double* tolerance=nullptr;
};


class HarmonicPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    HarmonicPointMotion(const std::vector<Point>& points,const std::vector<int>& offsets,
        const std::vector<int>& neighbours,const std::vector<unsigned char>& fixed,
        double tolerance,int maximumIterations=20000,const std::vector<MotionPenalty>& penalties={});
    ~HarmonicPointMotion();
    HarmonicResult solve(const std::vector<Point>& boundary,bool representedDisplacement=false);
    HarmonicDeviceView deviceView()const;
    const Point* initialPoints()const;
    const unsigned char* fixedPoints()const;
    HarmonicDeviceView solveSelectedDevice(const HarmonicMotionDeviceInput*,
        bool representedDisplacement=false,void* stream=nullptr);







    void prepareDeviceGeometry(const Point* points,void* stream=nullptr);
    HarmonicDeviceView solvePreparedDevice(const Point* points,const Point* boundary,
        const MotionPenalty* penalties,const double* tolerance,
        bool representedDisplacement=false,void* stream=nullptr);
    HarmonicDeviceView solveDevice(const Point* points,const Point* boundary,
        const MotionPenalty* penalties,const double* tolerance,
        bool representedDisplacement=false,void* stream=nullptr);
};
}}}
#endif
