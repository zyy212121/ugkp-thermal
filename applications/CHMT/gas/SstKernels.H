#ifndef CHMT_GAS_SSTKERNELS_H
#define CHMT_GAS_SSTKERNELS_H
#include "gpu/DeviceViews.H"
namespace chmt {
    // Called internally by launchGasRhs before turbulent mean-flow stress/heat.
    int launchSstPrepare(GasView, GeometryView, const PhysicsConfig&, void*);
    int launchSstRhs(GasView, GeometryView, const PhysicsConfig&, double, void*);
    // Call after endpoint gas/interface state recovery, for predictor and full endpoints.
    // GeometryView.wallDistance/boundaryPrimitive must be explicit endpoint arrays.
    // Mandatory final gate after the endpoint wall projection, before budget/commit.
    // Rejects unfinished (nonfinite) projection diagnostics and strict SST minima.
    int launchSstValidate(GasView,const PhysicsConfig&,DeviceStatus*,void*);
    int launchSstConstrainWalls(GasView, GeometryView, const PhysicsConfig&, DeviceStatus*, void*);
    int launchSstUpdate(const SstQ*, GasView, GeometryView, const PhysicsConfig&, double, DeviceStatus*, void*);
}
#endif
