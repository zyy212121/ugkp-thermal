#ifndef CHMT_MATERIALS_MATERIALTRANSPORT_H
#define CHMT_MATERIALS_MATERIALTRANSPORT_H
#include "core/HostState.H"
#include "materials/MaterialCaloric.H"
namespace chmt {
struct MaterialTransportResult {
    std::vector<SolidQ> rates,faceFlux;
    std::vector<PhysicalReal> faceLossLeft,faceLossRight;
    std::vector<PhysicalVec3> poreVelocity;
    Real dtLimit=std::numeric_limits<Real>::max();
};
bool applyMaterialPacketBatch(const std::vector<SolidQ>&,const std::vector<ExchangePacket>&,std::vector<SolidQ>&,std::string&,bool* numericalFailure=nullptr);
bool materialPointsClose(const HostMesh&,const std::vector<Vec3>&,const Tolerances&);
bool materialSweepsCompatible(const HostMesh&,const SurfaceMesh&,const std::vector<Real>& targets,const std::vector<Real>& actual,const Tolerances&);


bool validateMaterialTransportMesh(const HostMesh&,std::size_t cells,std::string& error);
bool evaluateMaterialTransport(const HostMesh&,const std::vector<SolidQ>&,
    const PhysicsConfig&,const std::vector<Real>& sweptVolumes,Real geometryInterval,
    PhysicalReal transportCfl,MaterialTransportResult&,std::string&,bool* numericalFailure=nullptr,
    const std::vector<MaterialPrimitive>* endpoint=nullptr);


bool advanceMaterialReactions(const SolidQ&,Real volume,const PhysicsConfig&,Real interval,
    Real maxSubstep,PhysicalReal reactionCfl,int maxSubsteps,SolidQ&,
    std::uint64_t& acceptedSubsteps,std::string&,bool* numericalFailure=nullptr);

}
#endif
