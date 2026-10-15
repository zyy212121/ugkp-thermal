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
    const std::vector<Real>* sideMeshVolumeRate=nullptr,
    const std::vector<GasWallClosureContext>* wallContexts=nullptr,bool* numericalFailure=nullptr);
namespace detail {
bool completeCpuSurface(const HostState&,const PhysicsConfig&,const std::vector<GasPrimitive>&,const std::vector<GasGradient>&,const std::vector<SolidQ>&,const std::vector<FilmQ>&,Real,std::uint64_t,CpuSurfaceResult&,std::string&,const std::vector<Real>*,const std::vector<GasWallClosureContext>*,bool*);
}

}
#endif
