#ifndef UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#define UGKP_GPU_POLYHEDRAL_GEOMETRY_H
#include "PolyhedralGeometryMath.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu { namespace geometry {
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
struct HarmonicResult {
    std::vector<Point> points;
    double residual=0;
    int iterations=0,errorPoint=-1,errorKind=0;
};
// Immutable geometry/adjacency is shared by every boundary mode in one solve.
// All Jacobi updates and stopping decisions run on the GPU; no CPU solve path.
class HarmonicPointMotion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    HarmonicPointMotion(const std::vector<Point>& points,const std::vector<int>& offsets,
        const std::vector<int>& neighbours,const std::vector<unsigned char>& fixed,
        double tolerance,int maximumIterations=20000);
    ~HarmonicPointMotion();
    HarmonicResult solve(const std::vector<Point>& boundary,bool representedDisplacement=false);
};
}}}
#endif
