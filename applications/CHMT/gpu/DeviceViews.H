#ifndef CHMT_GPU_DEVICEVIEWS_H
#define CHMT_GPU_DEVICEVIEWS_H
#include "core/Configuration.H"
namespace chmt {
// Zero counts mean no allocation. Every non-optional pointer is valid for its
// count when its operator is launched. All allocation belongs to Backend.
struct SstView {
    int nCells=0,nFaces=0;SstQ* q=nullptr;SstQ* rhs=nullptr;SstPrimitive* primitive=nullptr;
    Vec3* gradientK=nullptr;Vec3* gradientOmega=nullptr;SstQ* faceFlux=nullptr;
    const Real* wallDistance=nullptr;const SstPrimitive* boundary=nullptr;
    Real* production=nullptr;Real* dissipation=nullptr;
    Real* omegaConstraint=nullptr; // nCells, actual endpoint rhoOmega projection delta [kg/s].
    // nCells, J/s modeled-k rates. RHS writes evaluated rates; SST update
    // replaces dissipation with actual exponential-integrated destruction/dt.
    // Coordinator integrates these once for an accepted stage; no omega mixing.
};
struct GasView {
    const GasQ* stageBase=nullptr; // accepted-step donor inventory; both midpoint stages.
    int nCells=0,nFaces=0;GasQ* q=nullptr;GasQ* rhs=nullptr;GasPrimitive* primitive=nullptr;
    GasGradient* gradient=nullptr;GasQ* faceFlux=nullptr;SstView sst{};
    const Real* voidFraction=nullptr; // optional iff particles disabled, implies 1.
    DeviceStatus* status=nullptr; // optional only for callers that check status separately.
};
struct SolidView {
    const SolidQ* stageBase=nullptr; // donor bounds use this base, rates evaluate q.
    int nCells=0,nFaces=0;SolidQ* q=nullptr;SolidQ* rhs=nullptr;
    Real* temperature=nullptr;Real* porePressure=nullptr;Vec3* gradientTemperature=nullptr;Vec3* poreVelocity=nullptr;
    Vec3* gradientPressure=nullptr; // nCells, original-cell Darcy pressure gradient.
    Real* pressureGradientSensitivity=nullptr; // nCells, least-squares L1 bound [1/m].
    Real* reactionRate=nullptr; // nCells*Nr mol/s, optional iff reactions disabled.
    SolidQ* faceFlux=nullptr;DeviceStatus* status=nullptr;
};
struct NormalThermalView {
    int nFaces=0,nLayers=0; // nFaces is number of normal thermal columns.
    const int* offsets=nullptr; // nFaces+1 CSR layer offsets.
    const Real* coordinate=nullptr;const Real* thickness=nullptr; // nLayers, m.
    const Real* area=nullptr; // nFaces, m^2 reference column area.
    const Real* pressure=nullptr;const Real* pressureRate=nullptr;
    // nFaces, Pa and Pa/s. pressureRate is prescribed material Dp/Dt along
    // the same normal-column material trajectory used by mass/enthalpy update.
    // Its V*Dp/Dt true-H source is evolved here; do NOT add a second film
    // delta(pV) source to normal enthalpy. U diagnostics subtract actual pV.
    // When normal true H is authoritative, film H aggregates these same layers;
    // it must not receive an independent parallel pressure-volume conversion.
    const Real* mass=nullptr;const Real* oldVolume=nullptr;const Real* newVolume=nullptr; // nLayers: mass [kg], old/new layer volume [m^3].
    Real* enthalpy=nullptr;Real* rhs=nullptr;Real* temperature=nullptr;
    Real* liquidFraction=nullptr; // nLayers, dimensionless, persistent phase state.
    Real* bottomHeatRate=nullptr;Real* topHeatRate=nullptr; // nFaces, W inward.
    Real* acceptedWallHeat=nullptr; // nFaces, cumulative accepted net wall heat, J.
    // RHS/corrector kernels may write trial heat RATES only. They must never
    // increment acceptedWallHeat. Backend commits stage-1 integral exactly once.
    const Real* bottomTemperature=nullptr;const Real* topTemperature=nullptr;
    const Real* bottomHeatFlux=nullptr;const Real* topHeatFlux=nullptr;
    // Boundary arrays have nFaces entries: temperature [K], inward heat flux [W/m^2].
    // Exactly one boundary temperature or signed inward heat-flux array is
    // provided per side; absent side means no boundary prescription is supplied.
};
struct FilmView {
    const FilmQ* stageBase=nullptr; // donor bounds/dry transfer base; not another persistent state.
    FilmRateBudget* rateBudget=nullptr; // nFaces trial W rates, overwritten per RHS.
    int nFaces=0,nEdges=0;FilmQ* q=nullptr;FilmQ* rhs=nullptr;FilmAux* aux=nullptr;
    FilmQ* edgeFlux=nullptr;Vec3* pressureGradient=nullptr;Vec3* gasShear=nullptr;
    Real* pressureRate=nullptr;Real* bottomTemperature=nullptr;Real* topTemperature=nullptr;
    Real* reducedResidual=nullptr;Real* kineticDefect=nullptr; // obsolete: leave null; Backend computes actual accepted deltaK+kinetic outflow.
    NormalThermalView normal{};
    FilmPressureVolumeUpdate* storage=nullptr;DeviceStatus* status=nullptr;
};
struct GeometryView {
    int nCells=0,nFaces=0,nPoints=0;
    const int* owner=nullptr;const int* neighbour=nullptr;const int* periodicPartner=nullptr;
    const Vec3* cellCentre=nullptr;const Vec3* faceCentre=nullptr;const Vec3* areaVector=nullptr;
    const Real* oldVolume=nullptr;const Real* newVolume=nullptr;const Real* sweptVolume=nullptr;
    const Real* evaluationVolume=nullptr; // nCells, m^3 of RHS evaluation state.
    // Gas primitive recovery uses evaluationVolume at the midpoint; old/new
    // remain accepted->endpoint volume/GCL/update data. Required by gas RHS.
    const BoundaryKind* boundaryKind=nullptr;const GasPrimitive* boundaryPrimitive=nullptr;
    const ThermalBoundaryKind* thermalBoundary=nullptr; // null means zeroGradient
    const std::uint64_t* faceIds=nullptr;
    const int* cellFaceOffsets=nullptr;const int* cellFaces=nullptr;const int* cellFaceSigns=nullptr;
    const int* faceOffsets=nullptr;const int* facePoints=nullptr;
    const Vec3* oldPoints=nullptr;const Vec3* newPoints=nullptr;const Vec3* meshVelocity=nullptr;
    const Real* wallDistance=nullptr; // required only for SST.
    std::uint64_t geometryVersion=0,topologyHash=0;
};
struct SurfaceView {
    int nSolidFaces=0; // borrowed arrays below have this full-solid-face count.
    const Vec3* solidAreaVector=nullptr; // actual stage physical base area vectors, indexed by solidFace.
    const Real* solidSweptVolume=nullptr; // outward-from-solid stage swept m^3, indexed by solidFace.
    Real* solidRecessionResidual=nullptr; // nFaces, m^3/s: actual minus constitutive solid sweep.
    Real* filmVolumeResidual=nullptr; // nFaces, m^3/s: film/gap volume rate minus actual top/base/edge sweeps.
    // Required also when dry: zero inventory requires coincident gas/solid swept boundaries.

    const Vec3* prescribedTopTraction=nullptr; // immutable standalone Pa input.
    const Vec3* prescribedPressureGradient=nullptr; // external pressure-drop forcing, Pa/m.
    GasPrimitive* interfaceGasState=nullptr; // nFaces trial output, actual interface T/p/Y/rho/u for boundary scatter.
    Real* radiationPower=nullptr; // nFaces, W inward, trial output (not geometric persistence).
    const Vec3* edgeOwnerOffset=nullptr;const Vec3* edgeNeighbourOffset=nullptr; // nEdges, m, each side's own image.
    const Real* gasDistance=nullptr;const Real* solidDistance=nullptr; // nFaces, m; unused only for an absent side.
    int nFaces=0,nEdges=0,nMapEntries=0;
    const int* gasFace=nullptr;const int* solidFace=nullptr;const int* solidCell=nullptr;
    const std::uint64_t* persistentId=nullptr;const int* edgeOwner=nullptr;const int* edgeNeighbour=nullptr;
    const Real* edgeLength=nullptr;const Vec3* edgeConormal=nullptr;
    const Real* area=nullptr;const Real* oldArea=nullptr;const Real* sweptEdgeArea=nullptr;
    const Vec3* centre=nullptr;const Vec3* normal=nullptr;const Vec3* meshVelocity=nullptr;const Vec3* baseVelocity=nullptr;
    const int* gasMapOffsets=nullptr;const int* gasMapFaces=nullptr;const Real* gasMapWeights=nullptr;
    // All mapping pointers null iff identity gasFace mapping is used.
};
struct PacketView { int count=0;ExchangePacket* packets=nullptr; };
struct ParticleView { int count=0;ParticleQ* q=nullptr;ParticleQ* rhs=nullptr; };
static_assert(std::is_trivially_copyable<SstView>::value,"SstView device ABI");
static_assert(std::is_trivially_copyable<GasView>::value,"GasView device ABI");
static_assert(std::is_trivially_copyable<SolidView>::value,"SolidView device ABI");
static_assert(std::is_trivially_copyable<FilmView>::value,"FilmView device ABI");
static_assert(std::is_trivially_copyable<NormalThermalView>::value,"NormalThermalView device ABI");
static_assert(std::is_trivially_copyable<GeometryView>::value,"GeometryView device ABI");
static_assert(std::is_trivially_copyable<SurfaceView>::value,"SurfaceView device ABI");
} // namespace chmt
#endif
