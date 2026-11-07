#ifndef UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#define UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#include "PolyhedralGeometryMath.H"
#include "PointMotionMath.H"
#include "LinearTrajectoryMath.H"
#include "GeometryBoundaryMath.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu { class PinnedTransferQueue; namespace geometry {
// Topology/validity policies belong to the application. These operators consume
// validated CSR topology and preserve the native within-face/within-cell order.
struct Topology {
    std::vector<int> faceOffsets,facePoints,cellOffsets,cellFaces,cellSigns,vertexOffsets,vertices;
};
struct WallDistanceResult {double distance=0;int invalidFace=-1;};
struct BoundaryMotionFace {int face;double previousOffset,offset,thickness;Point supportVelocity;};
struct BoundaryMotionPoint {Point displacement,direction;double weight=0,thickness=0;};
struct RebuildResult {
    std::vector<FaceResult> faces;std::vector<CellResult> cells;std::vector<WallDistanceResult> distance;
};
// Borrowed device arrays; valid for the lifetime of the owner. A rebuild
// overwrites the metric arrays, and sweep overwrites only sweptFaces. The
// caller orders producers and consumers on its stream (or with device events).
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
// Immutable topology and wall identity are uploaded once at construction.
// Coordinates are borrowed directly from their device owner. Enqueue methods
// allocate nothing, perform no transfer or synchronization, and can be captured
// in the caller's execution graph. Topology validation remains caller-owned.
class ResidentPolyhedralGeometry {
    struct Implementation;std::unique_ptr<Implementation> data_;
    // The host publication adapter may rebind topology between completed
    // calls, retaining allocation high-water marks. Resident callers keep
    // immutable topology, so their captured views cannot be invalidated here.
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
    // firstFailure equals the cell count when evaluation succeeds.
    int moving=0,firstFailure=-1;
    TrajectoryFailure failure=TrajectoryFailure::None;
};
struct TrajectoryDeviceView {
    int cells=0;
    const TrajectoryStatus* status=nullptr;
    const TrajectoryFailure* failure=nullptr;
    const TrajectoryCellDiagnostic* diagnostics=nullptr;
};
// The topology is initialized once. Every evaluation borrows device coordinates
// and leaves its decision and ordered per-cell diagnostics on device. Warning
// counts after the first failed cell are zero, matching serial publication.
// As in the shared diagnostic context, retain a count and first sample rather
// than an unbounded list of repeated warnings. Storage is linear in cell count.
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
// Geometry-dependent boundary checks for an initialized, topology-preserving
// mesh. The periodic topology and Empty extrusion/mate map are validated once
// by the mesh owner. Coordinate finiteness, face/cell validity and the remaining
// stage policies belong to that owner. These methods never publish to host.
// Empty uniqueness is checked at each original geometry sample, in addition
// to the paired, zero-normal trajectory contract; it cannot be assumed from
// valid individual cell volumes on a mesh with several disconnected cells.
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
// Geometry shared by streamed harmonic modes. The correction remains resident
// while modes are accumulated in their original order.
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
// Numerical descriptors are produced on device; immutable face identities and
// storage are initialized once. Coefficients may be selected directly from a
// resident linear solve. No host round trip is required between streamed modes.
struct SweepLinearizationDeviceInput {
    const Point* old=nullptr;
    const Point* endpoint=nullptr;
    const Point* mode=nullptr;
    const double* scales=nullptr;
    const PlanarConstraint* planar=nullptr;
    double coefficient=0;
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
    // All input arrays are borrowed, validated and kept alive by the graph
    // owner. Each phase consumes its descriptor after prior stream producers.
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
// Row-pivoted QR of the transpose, with two ordered reorthogonalization
// passes. Compatibility diagnostics remain with the caller.
class DenseMinimumNorm {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    DenseMinimumNorm();~DenseMinimumNorm();
    MinimumNormResult solve(int rows,int columns,const std::vector<double>& matrix,
        const std::vector<double>& rhs,double rankTolerance);
};
// A graph producer selects the actual dimensions and coefficient pointers on
// device. Capacities are derived from initialized topology, not policy limits.
struct MinimumNormDeviceProblem {
    int rows=0,columns=0;
    double rankTolerance=0;
    const double* matrix=nullptr;
    const double* rhs=nullptr;
};
struct MinimumNormDeviceStatus {
    double largest=0,length=0;
    int rank=0,active=0,error=0; // 1: invalid input; 2: nonfinite projected RHS
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
    // Borrows immutable device input and leaves the original matrix/RHS intact.
    // Scratch is allocated at construction; enqueue has no host publication,
    // allocation or synchronization and can be captured in the owner's graph.
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
// Immutable geometry/adjacency is shared by every boundary mode in one solve.
// All Jacobi updates and stopping decisions run on the GPU; no CPU solve path.
class HarmonicPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    HarmonicPointMotion(const std::vector<Point>& points,const std::vector<int>& offsets,
        const std::vector<int>& neighbours,const std::vector<unsigned char>& fixed,
        double tolerance,int maximumIterations=20000,const std::vector<MotionPenalty>& penalties={});
    ~HarmonicPointMotion();
    HarmonicResult solve(const std::vector<Point>& boundary,bool representedDisplacement=false);
    // Adjacency and fixed-point mask remain immutable. Coordinates, constraints
    // and tolerance may come from an upstream device geometry owner. A null
    // tolerance uses the constructor value; a null penalty means no penalty.
    // Result/status stay on device and are ordered on the caller's stream.
    HarmonicDeviceView solveDevice(const Point* points,const Point* boundary,
        const MotionPenalty* penalties,const double* tolerance,
        bool representedDisplacement=false,void* stream=nullptr);
};
}}}
#endif
