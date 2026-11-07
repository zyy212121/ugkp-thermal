#ifndef UGKP_GPU_LINEAR_TRAJECTORY_MATH_H
#define UGKP_GPU_LINEAR_TRAJECTORY_MATH_H
#include "TrajectoryPolynomialMath.H"
#ifdef __CUDACC__
#define UGKP_LINEAR_TRAJECTORY_HD __host__ __device__
#else
#define UGKP_LINEAR_TRAJECTORY_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
enum class TrajectoryFailure : int {None,ZeroRadius,NonfiniteDeterminant,CollapsedFan,NonStarVolume};
enum class TrajectoryWarning : int {FanMargin,SolidAngle};
namespace trajectoryDetail {
constexpr double epsilon=std::numeric_limits<double>::epsilon();
inline long double norm(long double value){return std::sqrt(value);}
inline bool positive(long double value){return value>0;}
inline long double angle(long double y,long double x){return std::atan2(y,x);}
inline long double numeric(long double value){return value;}
inline long double sphere(long double){return 4*std::acos(-1.L);}
UGKP_LINEAR_TRAJECTORY_HD inline extended::Wide norm(extended::Wide value){return extended::squareRoot(value);}
UGKP_LINEAR_TRAJECTORY_HD inline bool positive(extended::Wide value){return value.high>0;}
UGKP_LINEAR_TRAJECTORY_HD inline double numeric(extended::Wide value){return extended::value(value);}
UGKP_LINEAR_TRAJECTORY_HD inline double sphere(extended::Wide){return 4*::acos(-1.);}

// atan2 is homogeneous in its two arguments. Scale both extended values by
// the same power of two before the final FP64 transcendental, retaining very
// small determinants/denominators without narrowing their absolute exponent.
UGKP_LINEAR_TRAJECTORY_HD inline double angle(extended::Wide y,extended::Wide x){
    const int scale=y.high==0?x.exponent:x.high==0?y.exponent:
        (y.exponent>x.exponent?y.exponent:x.exponent);
    const double yy=::ldexp(y.high,y.exponent-scale)+::ldexp(y.low,y.exponent-scale);
    const double xx=::ldexp(x.high,x.exponent-scale)+::ldexp(x.low,x.exponent-scale);
    return ::atan2(yy,xx);
}
}
// One cell of the original oriented fan certificate. Vertex CSR must have the
// canonical sorted/unique order, and topology/manifold validity belongs to the
// immutable mesh owner. Invoke for every cell when ANY point moves: skipping
// just locally stationary cells would change the existing trajectory contract.
// The warning callback preserves the order and values within this cell; its
// owner decides when to publish diagnostics. No repair or geometric tolerance
// is substituted for the positivity or single-covering tests.
template<class Extended=extended::Wide,class Vec=Point,class Warning>
UGKP_LINEAR_TRAJECTORY_HD inline TrajectoryFailure certifyLinearCell(
    int cell,TopologyView topology,const Vec* points,const Vec* endpoint,Warning warning){
    using Wide=Extended;
    Vec centre{},endCentre{};
    const int first=topology.vertexOffsets[cell],last=topology.vertexOffsets[cell+1];
    for(int k=first;k<last;++k){const int v=topology.vertices[k];centre+=points[v];endCentre+=endpoint[v];}
    centre=centre/double(last-first);endCentre=endCentre/double(last-first);
    Wide solidAngle{};int triangles=0;
    for(int k=topology.cellOffsets[cell];k<topology.cellOffsets[cell+1];++k){
        const int face=topology.cellFaces[k],sign=topology.cellSigns[k];
        const int begin=topology.faceOffsets[face],end=topology.faceOffsets[face+1],ia=topology.facePoints[begin];
        for(int j=begin+1;j+1<end;++j){
            const int ib=topology.facePoints[j],ic=topology.facePoints[j+1];
            const Vec r=points[ia]-centre,u=points[ib]-centre,v=points[ic]-centre;
            const Vec R=endpoint[ia]-endCentre,U=endpoint[ib]-endCentre,V=endpoint[ic]-endCentre;
            Wide unit[3][3]={{Wide(r.x),Wide(r.y),Wide(r.z)},{Wide(u.x),Wide(u.y),Wide(u.z)},{Wide(v.x),Wide(v.y),Wide(v.z)}};
            for(auto& w:unit){
                const Wide norm=trajectoryDetail::norm(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
                if(!trajectoryDetail::positive(norm))return TrajectoryFailure::ZeroRadius;
                for(auto& x:w)x=x/norm;
            }
            const auto& a0=unit[0];const auto& b0=unit[1];const auto& c0=unit[2];
            const Wide determinant=a0[0]*(b0[1]*c0[2]-b0[2]*c0[1])
                +a0[1]*(b0[2]*c0[0]-b0[0]*c0[2])+a0[2]*(b0[0]*c0[1]-b0[1]*c0[0]);
            Wide denominator(1);
            for(int component=0;component<3;++component)
                denominator=denominator+(a0[component]*b0[component]+b0[component]*c0[component]+c0[component]*a0[component]);
            solidAngle=solidAngle+Wide(sign*2)*Wide(trajectoryDetail::angle(determinant,denominator));++triangles;
            double tetra[4]={sign*dot(r,cross(u,v)),
                sign*(dot(R,cross(u,v))+dot(r,cross(U,v))+dot(r,cross(u,V)))/3,
                sign*(dot(R,cross(U,v))+dot(R,cross(u,V))+dot(r,cross(U,V)))/3,
                sign*dot(R,cross(U,V))};
            for(double value:tetra)if(!std::isfinite(value))return TrajectoryFailure::NonfiniteDeterminant;
            double lower=0,upper=0;trajectoryPolynomialBounds<Wide>(tetra,lower,upper);
            if(!std::isfinite(lower)||!(tetra[0]>0)||lower<=0)return TrajectoryFailure::CollapsedFan;
            const double margin=128*trajectoryDetail::epsilon*tetra[0];
            if(lower<=margin)warning(lower,margin,TrajectoryWarning::FanMargin);
        }
    }
    const auto angle=trajectoryDetail::numeric(solidAngle),sphere=trajectoryDetail::sphere(solidAngle);
    if(!std::isfinite(angle)||std::round(angle/sphere)!=1)return TrajectoryFailure::NonStarVolume;
    const auto residual=std::abs(angle-sphere);const double tolerance=1024*trajectoryDetail::epsilon*triangles;
    if(residual>tolerance)warning(static_cast<double>(residual),tolerance,TrajectoryWarning::SolidAngle);
    return TrajectoryFailure::None;
}
}}}
#undef UGKP_LINEAR_TRAJECTORY_HD
#endif
