#ifndef CHMT_GAS_GASKERNELS_H
#define CHMT_GAS_GASKERNELS_H
#include "gpu/DeviceViews.H"
namespace chmt {
    // Endpoint bootstrap: same primitive/gradient kernels, no SST algebra or state.
    int launchGasMeanPrepare(GasView, GeometryView, const PhysicsConfig&, double, void*);
    // Idempotent scratch-only primitive/gradient/SST preparation; never changes q or budgets.
    int launchGasPrepare(GasView, GeometryView, const PhysicsConfig&, double, void*);
    int launchGasRhs(GasView, GeometryView, PacketView, const PhysicsConfig&, double, void*);
    int launchGasUpdate(const GasQ* base, GasView, const GeometryView&, double, DeviceStatus*, void*);
    int launchGasValidate(GasView, GeometryView, const PhysicsConfig&, void*);
}
#endif
