#ifndef CHMT_CORE_HOSTSTATE_H
#define CHMT_CORE_HOSTSTATE_H
#include "Configuration.H"
#include <vector>
#include <array>
namespace chmt {
struct HostMesh {
    std::vector<Vec3> referencePoints{}; // immutable prescribed-motion reference, persisted.
    std::vector<Vec3> points{},oldPoints{},cellCentres{},faceCentres{},areaVectors{},meshVelocity{};
    std::vector<Real> volumes{},oldVolumes{},wallDistance{};
    std::vector<int> faceOffsets{},facePoints{},owner{},neighbour{},periodicPartner{},cellFaceOffsets{},cellFaces{},cellFaceSigns{};
    std::vector<BoundaryKind> boundaryKind{};std::vector<GasPrimitive> boundaryPrimitive{};
    // Empty is the homogeneous zeroGradient shorthand for programmatic meshes.
    // Imported meshes always carry one independent policy per face.
    std::vector<ThermalBoundaryKind> thermalBoundary{};
    std::vector<SstPrimitive> boundarySst{};std::vector<std::uint64_t> faceIds{};
    std::uint64_t geometryVersion=0,topologyHash=0;
};
struct SurfaceMesh {
    std::vector<Vec3> prescribedTopTraction{},prescribedPressureGradient{};
    std::vector<Vec3> edgeOwnerOffset{},edgeNeighbourOffset{}; // edge centre minus face centre in each periodic image [m].
    std::vector<Real> gasDistance{},solidDistance{}; // nFaces, positive normal cell-to-interface distances [m].
    std::vector<int> gasFace{},solidFace{},solidCell{},edgeOwner{},edgeNeighbour{};
    std::vector<std::uint64_t> persistentId{};
    std::vector<Real> edgeLength{},area{},oldArea{},sweptEdgeArea{};
    std::vector<Vec3> edgeConormal{},centre{},normal{},meshVelocity{},baseVelocity{};
    // CSR map from a surface donor integral to gas-face receivers; weights >=0
    // sum to one for each donor. Identity mapping may leave all three empty.
    std::vector<int> gasMapOffsets{},gasMapFaces{};std::vector<Real> gasMapWeights{};
    // Normal-layer centres/volumes, concatenated with per-face CSR offsets.
    std::vector<int> normalOffsets{};std::vector<Real> normalCoordinates{},normalLayerMass{},normalLayerVolume{},oldNormalLayerVolume{};
    std::vector<Real> normalThickness{}; // nLayers, m; persistent geometry.
    std::vector<Real> normalPressure{},normalPressureRate{}; // nFaces, Pa and material Pa/s.
    std::vector<Real> bottomTemperature{},topTemperature{},bottomHeatFlux{},topHeatFlux{};
};
struct HostStageGeometry {
    Real interval=0;
    std::uint64_t geometryVersion=0,topologyHash=0;
    std::vector<Real> oldVolume{},newVolume{},evaluationVolume{},sweptVolume{};
    std::vector<Vec3> areaVector{},cellCentre{},faceCentre{},oldPoints{},newPoints{};
};
// Accepted cumulative diagnostics from the common SST source/transport owner.
// Net source is not relabelled as gross production or dissipation.
struct GasSstIntegralAudit {
    Real transportK=0,transportOmega=0,sourceK=0,sourceOmega=0,constraintK=0,constraintOmega=0;
};
struct HostState {
    std::array<HostStageGeometry,2> gasStages{},solidStages{};
    Real nextDt=0,lastAcceptedDt=0;

    HostMesh gasMesh{},solidMesh{};SurfaceMesh surface{};
    std::vector<ugkwp::ChemistryAudit<Real,Ns>> gasChemistryAudit{}; // accepted cumulative cell diagnostics only
    std::vector<GasQ> gas{};std::vector<SolidQ> solid{};std::vector<FilmQ> film{};
    std::vector<FilmAux> filmAux{};std::vector<ParticleQ> particles{};std::vector<SstQ> sst{};
    // Numerical geometry carry, indexed by surface.solidFace [signed m^3]:
    // cumulative physical consumed-volume target minus measured actual sweep.
    // Empty means zero; never a mass, energy, or physical source correction.
    std::vector<Real> solidSweepRemainder{};
    std::vector<Real> gasVoidFraction{},normalEnthalpy{};
    std::vector<Real> normalLiquidFraction{}; // nLayers, dimensionless phase state.
    std::vector<Real> normalAcceptedWallHeat{}; // nFaces, cumulative accepted J.
    // Trial boundary heat rates are Backend-owned candidate scratch, not state;
    // fractions/accepted heat are rolled back and checkpointed with normal H.
    std::vector<ExchangePacket> ledger{};std::vector<FilmPressureVolumeUpdate> filmStorage{};
    // Reconstructible diagnostic snapshot, not physical state. Deliberately
    // absent from checkpoints; restart output marks it NOT_AVAILABLE until a
    // new accepted gas stage supplies it. Rollback still copies it atomically.
    std::vector<ugkwp::gaswall::WallOutput<Real,Ns>> gasWallDiagnostics{};
    Real gasWallDiagnosticTime=-1;
    GasSstIntegralAudit gasSstAudit{};
    Real time=0;std::uint64_t acceptedSteps=0,rejectedSteps=0,commitSequence=0;Budget budget{};
};
struct StepReport {
    Real acceptedDt=0,nextDt=0,couplingResidual=0,gclResidual=0,minimumVolume=0,minimumMass=0,minimumTemperature=0;
    Real filmReducedResidual=0,filmKineticDefect=0;int iterations=0,rejectedTrials=0;
    std::uint64_t commitSequence=0;Budget budget{};
};
} // namespace chmt
#endif
