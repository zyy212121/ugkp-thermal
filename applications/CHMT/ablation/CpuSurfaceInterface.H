#ifndef CHMT_ABLATION_CPUSURFACEINTERFACE_H
#define CHMT_ABLATION_CPUSURFACEINTERFACE_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
#include "ablation/InterfaceMath.H"
namespace chmt {
struct CpuSurfaceResult {
    std::vector<SurfacePhysicsResult> physics;
    std::vector<ExchangePacket> phasePackets;
    std::vector<WallFaceSample> wall;
};
// This constructs constitutive PROPOSALS and CPU-internal SolidFilm packets.
// It never applies/recomputes the gas interval's already accepted energy.
bool evaluateCpuSurface(const HostState&,const PhysicsConfig&,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
    Real dt,std::uint64_t sequence,CpuSurfaceResult&,std::string&,
    const std::vector<Vec3>* filmPressureGradient=nullptr,const std::vector<Real>* sideMeshVolumeRate=nullptr);
}
#endif
