#ifndef CHMT_MATERIALS_MATERIALKERNELS_H
#define CHMT_MATERIALS_MATERIALKERNELS_H
#include "gpu/DeviceViews.H"
#include "materials/Reaction.H"
namespace chmt {
int launchMaterialRhs(SolidView, GeometryView, PacketView, const PhysicsConfig&, Real stageDt,
    DeviceStatus*, void* stream);
int launchMaterialValidate(SolidView, GeometryView, const PhysicsConfig&, DeviceStatus*, void* stream);
}
#endif
