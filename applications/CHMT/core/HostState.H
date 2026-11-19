#ifndef CHMT_CORE_HOSTSTATE_H
#define CHMT_CORE_HOSTSTATE_H
#include "Configuration.H"
#include "AcceptedLedger.H"
#include "../../../common/SharedSnapshotVector.H"
#include <vector>
#include <array>
#include <memory>
namespace chmt {
struct HostMesh {
    std::shared_ptr<const ugkp::boundary::Definition> gasBoundary;
    std::vector<Vec3> referencePoints{};
    std::vector<Vec3> points{},oldPoints{},cellCentres{},faceCentres{},areaVectors{},meshVelocity{};
    std::vector<Real> volumes{},oldVolumes{},wallDistance{};
    std::vector<int> faceOffsets{},facePoints{},owner{},neighbour{},periodicPartner{},cellFaceOffsets{},cellFaces{},cellFaceSigns{};
    std::vector<BoundaryKind> boundaryKind{};std::vector<GasPrimitive> boundaryPrimitive{};


    std::vector<ThermalBoundaryKind> thermalBoundary{};
    std::vector<SstPrimitive> boundarySst{};std::vector<std::uint64_t> faceIds{};
    std::uint64_t geometryVersion=0,topologyHash=0;
};
struct SurfaceMesh {
    std::vector<unsigned char> ablating{};
    std::vector<Vec3> prescribedTopTraction{},prescribedPressureGradient{};
    std::vector<Vec3> edgeOwnerOffset{},edgeNeighbourOffset{};
    std::vector<Real> gasDistance{},solidDistance{};
    std::vector<int> gasFace{},solidFace{},solidCell{},edgeOwner{},edgeNeighbour{};
    std::vector<std::uint64_t> persistentId{};
    std::vector<Real> edgeLength{},area{},oldArea{},sweptEdgeArea{};
    std::vector<Vec3> edgeConormal{},centre{},normal{},meshVelocity{},baseVelocity{};


    std::vector<int> gasMapOffsets{},gasMapFaces{};std::vector<Real> gasMapWeights{};

    std::vector<int> normalOffsets{};std::vector<Real> normalCoordinates{},normalLayerMass{},normalLayerVolume{},oldNormalLayerVolume{};
    std::vector<Real> normalThickness{};
    std::vector<Real> normalPressure{},normalPressureRate{};
    std::vector<Real> bottomTemperature{},topTemperature{},bottomHeatFlux{},topHeatFlux{};
};
struct HostStageGeometry {
    Real interval=0;
    std::uint64_t geometryVersion=0,topologyHash=0;
    std::vector<Real> oldVolume{},newVolume{},evaluationVolume{},sweptVolume{};
    std::vector<Vec3> areaVector{},cellCentre{},faceCentre{},oldPoints{},newPoints{};
};


struct GasSstIntegralAudit {
    Real transportK=0,transportOmega=0,sourceK=0,sourceOmega=0,constraintK=0,constraintOmega=0;
};



struct HostDynamicState {
    Real nextDt=0,lastAcceptedDt=0;
    Real radiationTime=0;
    std::vector<int> particleSourceFaces;
    std::vector<Real> particleSourceResidualMass;
    SurfaceMesh surface{};
    std::vector<ugkwp::ChemistryAudit<Real,Ns>> gasChemistryAudit{};
    std::vector<GasQ> gas{};std::vector<SolidQ> solid{};std::vector<FilmQ> film{};
    std::vector<Real> gasTemperature{};
    std::vector<FilmAux> filmAux{};std::vector<ParticleQ> particles{};std::vector<SstQ> sst{};



    std::vector<Real> solidSweepRemainder{};
    std::vector<Real> gasVoidFraction{},normalEnthalpy{};
    std::vector<Real> normalLiquidFraction{};
    std::vector<Real> normalAcceptedWallHeat{};


    AcceptedLedger ledger{};std::vector<FilmPressureVolumeUpdate> filmStorage{};



    std::vector<ugkwp::gaswall::WallOutput<Real,Ns>> gasWallDiagnostics{};
    Real gasWallDiagnosticTime=-1;
    GasSstIntegralAudit gasSstAudit{};
    Real time=0;std::uint64_t acceptedSteps=0,rejectedSteps=0,commitSequence=0;Budget budget{};
};
struct HostState : HostDynamicState {
    std::array<HostStageGeometry,2> gasStages{},solidStages{};
    HostMesh gasMesh{},solidMesh{};
};


inline Real gasPhaseVolume(const HostState& state,std::size_t cell) {
    return state.gasMesh.volumes[cell]*(state.gasVoidFraction.empty()?Real(1):state.gasVoidFraction[cell]);
}
struct StepReport {
    Real acceptedDt=0,nextDt=0,couplingResidual=0,gclResidual=0,minimumVolume=0,minimumMass=0,minimumTemperature=0;
    Real filmReducedResidual=0,filmKineticDefect=0;int iterations=0,rejectedTrials=0;
    std::uint64_t commitSequence=0;Budget budget{};
};
}
#endif
