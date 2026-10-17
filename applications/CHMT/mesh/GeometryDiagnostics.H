#ifndef CHMT_MESH_GEOMETRY_DIAGNOSTICS_H
#define CHMT_MESH_GEOMETRY_DIAGNOSTICS_H
#include "Diagnostics.H"
#include <limits>
#include <cmath>
namespace chmt {
// Ratios only rank/report finite geometric residuals. Their representability
// must not decide whether a physical trajectory is valid.
inline double geometryResidualRatio(double residual,double tolerance) {
    if(!std::isfinite(residual)||residual<0||!(tolerance>0))
        return std::numeric_limits<double>::quiet_NaN();
    const long double ratio=static_cast<long double>(residual)/tolerance;
    const double limit=std::numeric_limits<double>::max();
    return ratio>limit?limit:static_cast<double>(ratio);
}
// Caller-owned per-operation context, drained at the existing host boundary.
inline void geometryWarning(ugkp::diagnostics::DeviceContext* context,double observed,
    double reference,int entity,bool conservation=true,
    ugkp::diagnostics::EntityKind kind=ugkp::diagnostics::EntityKind::Face) {
    using namespace ugkp::diagnostics;
    recordWarning(context,makeRecord(conservation?Code::ConservationResidual:Code::AccuracyResidual,
        conservation?3401:3402,34,0,kind,entity,-1,observed,reference,observed,
        conservation?Quantity::Conservation:Quantity::Accuracy));
}
}
#endif
