#ifndef CHMT_MATERIALS_MATERIALTRANSPORT_H
#define CHMT_MATERIALS_MATERIALTRANSPORT_H
#include "core/HostState.H"
#include "materials/MaterialCaloric.H"
namespace chmt {
struct MaterialTransportResult {
    std::vector<SolidQ> rates,faceFlux;
    std::vector<Vec3> poreVelocity;
    Real dtLimit=std::numeric_limits<Real>::max(),bodyPower=0;
};
bool filmThicknessFromMass(Real mass,Real area,Real density,Real& thickness);
bool filmSweepsCompatible(Real oldMass,Real newMass,Real density,Real gasOutwardSweep,Real solidOutwardSweep,Real sideSweep,const Tolerances&);
bool applyMaterialPacketBatch(const std::vector<SolidQ>&,const std::vector<ExchangePacket>&,std::vector<SolidQ>&,Real& numericalMassRoundoff,std::string&);
bool materialPointsClose(const HostMesh&,const std::vector<Vec3>&,const Tolerances&);
bool materialSweepsCompatible(const HostMesh&,const SurfaceMesh&,const std::vector<Real>& targets,const std::vector<Real>& actual,const Tolerances&);
// Conservative physical Darcy + ALE transport only. Conduction is exclusively
// the OpenFOAM implicit operator; reactions have their independent local clock.
bool evaluateMaterialTransport(const HostMesh&,const std::vector<SolidQ>&,
    const PhysicsConfig&,const std::vector<Real>& sweptVolumes,Real geometryInterval,
    Real transportCfl,MaterialTransportResult&,std::string&);
// First-order conservative chemistry with an extent-Jacobian bound and a
// full-versus-two-half-step local defect check. reactionCfl controls local
// rate/thermal variation, not a global error guarantee. Failure leaves both
// output and acceptedSubsteps untouched; estimator trials are not counted.
bool advanceMaterialReactions(const SolidQ&,Real volume,const PhysicsConfig&,Real interval,
    Real maxSubstep,Real reactionCfl,int maxSubsteps,SolidQ&,
    std::uint64_t& acceptedSubsteps,std::string&);
}
#endif
