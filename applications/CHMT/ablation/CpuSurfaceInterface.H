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


bool evaluateCpuSurface(const HostState&,const PhysicsConfig&,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
    Real dt,std::uint64_t sequence,CpuSurfaceResult&,std::string&,
    const std::vector<Real>* sideMeshVolumeRate=nullptr,
    const std::vector<GasWallClosureContext>* wallContexts=nullptr,bool* numericalFailure=nullptr);


}
#endif
