#ifndef CHMT_MESH_GEOMETRY_DIAGNOSTICS_H
#define CHMT_MESH_GEOMETRY_DIAGNOSTICS_H
#include "Diagnostics.H"
namespace chmt {
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
