#ifndef CHMT_ABLATION_INTERFACEKERNELS_H
#define CHMT_ABLATION_INTERFACEKERNELS_H
#include "gpu/DeviceViews.H"
#include "ablation/InterfaceMath.H"
namespace chmt {
constexpr int InterfacePacketsPerFace=3;
constexpr int InterfacePrimarySlot=0, InterfacePoreSlot=1, InterfacePhaseSlot=2;
enum class InterfaceTraceMode { ProvisionalModel, DiscreteAcceptedFlux };
int launchInterfacePackets(GasView, SolidView, FilmView, GeometryView, SurfaceView, PacketView,
    const PhysicsConfig&, Real stageDt, std::uint64_t step, int stage, DeviceStatus*, void* stream);
int launchInterfaceState(GasView, SolidView, FilmView, GeometryView, SurfaceView,
    const PhysicsConfig&, InterfaceTraceMode, PacketView acceptedFlux, Real fluxInterval,
    DeviceStatus*, void* stream);
}
#endif
