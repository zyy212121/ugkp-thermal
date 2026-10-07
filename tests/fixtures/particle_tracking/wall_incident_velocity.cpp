// Execute unmodified production bodies on the host. Only CUDA qualifiers,
// trap reporting and unrelated cold-wall profile initialization are adapted.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define __device__
#define __forceinline__ inline
#include "GpuPrecisionTypes.H"
#include "GpuSommerfeldSticking.H"
#include "GpuFiniteWallContact.H"
#include "GpuColdWallSolidification.H"
#include "gasNumerics/GpuDragModelsFSHCHT.cuh"

#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
#define GPU_OPERATOR_THERMAL 1
#define GPU_PERIODIC_FACE_KIND 6
#if INCIDENT_FSH
#define GPU_GEOMETRY_ROUNDING_AWARE 0
#define GPU_WALL_COORDINATE(hit, eps, normal) hit - eps*normal
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) hit - shift + eps*normal
#define GPU_RESET_CONTACT_AGE(s, i)
#define GPU_CONTACT_DURATION(value) static_cast<float>(value)
#define GPU_CONTACT_PEAK(value) static_cast<float>(value)
using ContactTime = float;
#else
#define GPU_GEOMETRY_ROUNDING_AWARE (UGKWP_GPU_REAL_BITS == 32)
#define GPU_WALL_COORDINATE(hit, eps, normal) insideFaceCoordinate(hit, normal, eps)
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) insideFaceCoordinate(hit-shift, -normal, eps)
#define GPU_RESET_CONTACT_AGE(s, i) s.pContactAge[i] = GpuTime(0);
#define GPU_CONTACT_DURATION(value) static_cast<GpuTime>(value)
#define GPU_CONTACT_PEAK(value) Foam::gpuThermal::finiteContactPeakFractionForStorage(value)
using ContactTime = GpuTime;
#endif

template<class T> T clampMin(T a, T b) { return std::max(a, b); }
template<class T> T clampRange(T a, T b, T c) { return std::min(std::max(a, b), c); }
template<class T> T finiteOr(T a, T b) { return std::isfinite(a) ? a : b; }
template<class T> bool nonFiniteDevice(T a) { return !std::isfinite(a); }

struct DeviceState
{
    int particleCapacity = 1, nCells = 1, nFaces = 6, maxFaceWalkHops = 32;
    int particleStuckModelConfigured = 1;
    int pStatus[1] = {1}, pCellId[1] = {0};
    int cellPlaneStart[3] = {0, 2, 4}, cellPlaneCount[3] = {2, 2, 2};
    int cellFaceKind[6] = {5, 5, 5, 5, 5, 5};
    int cellFaceNeighbor[6] = {-1, -1, -1, -1, -1, -1};
    int cellFaceId[6] = {0, 1, 2, 3, 4, 5};
    unsigned char pStuck[1] = {0}, particleStuckCandidateMask[6] = {1, 1, 1, 1, 1, 1};
    int pStuckFaceId[1] = {-1};
    GpuReal px[1] = {GPU_R(-0.01)}, py[1] = {}, pz[1] = {};
    GpuReal pux[1] = {GPU_R(1)}, puy[1] = {}, puz[1] = {};
    GpuReal puxOld[1] = {GPU_R(1)}, puyOld[1] = {}, puzOld[1] = {};
    GpuReal pd[1] = {GPU_R(1e-5)}, pT[1] = {GPU_R(2800)}, pTheta[1] = {GPU_R(0.5)};
    GpuReal planeNx[6] = {1, -1, 1, -1, 1, -1};
    GpuReal planeNy[6] = {}, planeNz[6] = {}, planeD[6] = {0, 1, 2, -1, 3, -2};
    GpuReal cellLength[3] = {1, 1, 1};
    GpuReal cellFaceRestitution[6] = {GPU_R(0.5), GPU_R(0.5), GPU_R(0.5), GPU_R(0.5), GPU_R(0.5), GPU_R(0.5)};
    GpuReal faceCx[6] = {0, -1, 2, 1, 3, 2}, faceCy[6] = {}, faceCz[6] = {};
    GpuReal facePeriodicDx[6] = {}, facePeriodicDy[6] = {}, facePeriodicDz[6] = {};
    GpuReal gasBoundaryUx[6] = {}, gasBoundaryUy[6] = {}, gasBoundaryUz[6] = {};
    GpuReal sommerfeldThreshold = GPU_R(20);
    GpuReal particleWallContactAngleCosine = GPU_R(-0.8191520442889918);
    float pDepositionArea[1] = {}, pContactMaximumArea[1] = {}, pContactPeakFraction[1] = {};
    GpuTime pContactAge[1] = {0.25};
    ContactTime pContactDuration[1] = {};
    GpuReal gasMu = GPU_R(1.8e-5), rhoSolid = GPU_R(3000), invRhoSolid = GPU_R(1.0/3000);
    GpuReal gravityX = 0, gravityY = 0, gravityZ = 0;
    GpuReal gradPx[3] = {}, gradPy[3] = {}, gradPz[3] = {};
    int initialized1D = 0, initialized2D = 0, cleared1D = 0, cleared2D = 0;
};

struct DeviceTrap { int line; };
#define asm(...) throw DeviceTrap{__LINE__}
bool isPeriodicFace(const DeviceState& s, int f) { return s.cellFaceKind[f] == 6; }
void clearColdWallParticleState(DeviceState& s, int) { ++s.cleared1D; }
void clearColdWall2DParticleState(DeviceState& s, int) { ++s.cleared2D; }
void initialiseColdWallParticleState(DeviceState& s, int, GpuReal) { ++s.initialized1D; }
void initialiseColdWall2DParticleState(DeviceState& s, int, GpuReal) { ++s.initialized2D; }
#include "operators/pointInsideCell.cuh"
#include "GpuParticleTransport.cuh"
#include "operators/dragInverseTimeDevice.cuh"

void advanceDrag(DeviceState& s, GpuTime dt, GpuReal invTauDrag, GpuReal ugx)
{
    const int i = 0, c = 0;
    const GpuReal ux = s.pux[0], uy = s.puy[0], uz = s.puz[0], ugy = 0, ugz = 0;
    s.puxOld[0] = ux;
    s.puyOld[0] = uy;
    s.puzOld[0] = uz;
#include "operators/ParticleDragMomentum.inl"
}

void require(bool condition, const char* message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

void near(double actual, double expected, const char* message)
{
    const double tolerance = UGKWP_GPU_REAL_BITS == 32 ? 2e-6 : 2e-12;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance*std::max(std::abs(expected), 1e-20))
    {
        std::fprintf(stderr, "FAIL: %s: actual=%.17g expected=%.17g\n", message, actual, expected);
        std::exit(1);
    }
}

struct Incident { GpuReal x, y, z; };

void checkContact(DeviceState& s, GpuTime dt, int face, int cell, Incident incident,
                  unsigned char model, bool deposit)
{
    using namespace Foam::gpuThermal;
    const GpuReal nx = s.planeNx[face], ny = s.planeNy[face], nz = s.planeNz[face];
    const GpuReal un = incident.x*nx + incident.y*ny + incident.z*nz;
    const GpuReal normalSpeed = std::abs((incident.x-s.gasBoundaryUx[face])*nx
        + (incident.y-s.gasBoundaryUy[face])*ny + (incident.z-s.gasBoundaryUz[face])*nz);
    const auto impact = evaluateSommerfeldImpact(s.pT[0], normalSpeed, s.pd[0], GPU_R(1));
    require(impact.valid && impact.sommerfeld > GPU_R(0), "valid incident material and Sommerfeld state");
    s.sommerfeldThreshold = impact.sommerfeld*(deposit ? GPU_R(0.75) : GPU_R(1.5));
    const auto expected = evaluateFiniteWallContactImpact(
        s.pT[0], s.pd[0], normalSpeed, s.particleWallContactAngleCosine);
    require(expected.valid, "valid finite-contact witness");
    for (auto& mask : s.particleStuckCandidateMask) { mask = model; }
    require(pointInsideCell(s, 0, s.px[0], s.py[0], s.pz[0]), "witness starts inside cell");
    trackOneParticleLocalFaceWalk(s, 0, dt);
    require(s.pStatus[0] == 1 && s.pCellId[0] == cell, "particle survives in expected cell");
    require(s.pStuckFaceId[0] == face, "expected contact face after all hops");
    const auto state = model == particleWallReboundContact || !deposit
        ? particleWallTransientRebound : particleWallTransientDeposit;
    require(s.pStuck[0] == state, "Sommerfeld selection uses current segment");
    require(s.pux[0] == 0 && s.puy[0] == 0 && s.puz[0] == 0, "bound velocity is zero");
    // Deliberately retain the pre-existing laboratory-frame rebound convention,
    // including for moving walls. This is not a moving-wall physics correction.
    const GpuReal factor = (GPU_R(1) + s.cellFaceRestitution[face])*un;
    near(s.puxOld[0], incident.x-factor*nx, "saved rebound x uses current segment");
    near(s.puyOld[0], incident.y-factor*ny, "saved rebound y uses current segment");
    near(s.puzOld[0], incident.z-factor*nz, "saved rebound z uses current segment");
    near(s.pContactDuration[0], GPU_CONTACT_DURATION(expected.contactDurationS), "contact duration");
    near(s.pContactMaximumArea[0], static_cast<float>(expected.maximumAreaM2), "contact maximum area");
    near(s.pContactPeakFraction[0], GPU_CONTACT_PEAK(expected.peakTimeFraction), "contact peak fraction");
    require(s.pContactDuration[0] > 0 && s.pContactMaximumArea[0] > 0
        && s.pContactPeakFraction[0] > 0 && s.pContactPeakFraction[0] < 1,
        "stored finite-contact properties are positive and finite");
    require(s.pTheta[0] == 0 && s.pDepositionArea[0] == 0, "contact starts with no age or damage");
#if !INCIDENT_FSH
    require(s.pContactAge[0] == 0, "CHT native contact age reset");
#endif
    require(s.initialized1D == (model == particleWallSolidifyingDeposition)
        && s.initialized2D == (model == particleWallColdWall2D)
        && s.cleared1D == (model != particleWallSolidifyingDeposition)
        && s.cleared2D == (model != particleWallColdWall2D), "cold-wall initialization route");
    require(pointInsideCell(s, cell, s.px[0], s.py[0], s.pz[0]), "contact position remains inside");

    // A later tracking call must preserve the rebound seed during contact.
    const Incident saved{s.puxOld[0], s.puyOld[0], s.puzOld[0]};
    trackOneParticleLocalFaceWalk(s, 0, dt);
    require(s.puxOld[0] == saved.x && s.puyOld[0] == saved.y && s.puzOld[0] == saved.z,
            "transient tracking preserves saved rebound");
}

void run(const char* scenario, unsigned char model, bool deposit)
{
    DeviceState s;
    GpuTime dt = 0.1;
    Incident incident{1, 0, 0};
    int face = 0, cell = 0;
    const auto is = [&](const char* name) { return std::strcmp(scenario, name) == 0; };
    if (is("force_zero_endpoint"))
    {
        dt = 1.0/1024;
        s.gravityX = GPU_R(-9.8125);
        s.pux[0] = GPU_R(9.8125/1024);
        const GpuReal invTau = dragInverseTimeDevice(s, GPU_R(1), GPU_R(1), s.pd[0],
            GPU_R(0), ugkwpGpuDrag::SchillerNaumannDrag{});
        advanceDrag(s, dt, invTau, s.pux[0]);
        require(s.pux[0] == 0, "production forced update has exactly zero endpoint");
        incident.x = GPU_R(0.5)*s.puxOld[0];
        s.px[0] = GPU_R(-0.5*dt*incident.x);
    }
    else if (is("drag_zero_endpoint"))
    {
        const GpuReal invTau = dragInverseTimeDevice(s, GPU_R(1), GPU_R(1), s.pd[0],
            GPU_R(2), ugkwpGpuDrag::SchillerNaumannDrag{});
        dt = std::log(2.0)/invTau;
        advanceDrag(s, dt, invTau, GPU_R(-1));
        require(s.pux[0] == 0, "production drag update has exactly zero endpoint");
        incident.x = GPU_R(0.5);
        s.px[0] = GPU_R(-0.25*dt);
    }
    else if (is("reversal") || is("oblique") || is("moving_wall"))
    {
        s.puxOld[0] = GPU_R(4); s.pux[0] = GPU_R(-2); incident.x = GPU_R(1);
        s.puyOld[0] = GPU_R(3); s.puy[0] = GPU_R(1); incident.y = GPU_R(2);
        s.puzOld[0] = GPU_R(2); s.puz[0] = GPU_R(-4); incident.z = GPU_R(-1);
        if (is("oblique"))
        {
            s.planeNx[0] = GPU_R(1.0/3); s.planeNy[0] = s.planeNz[0] = GPU_R(2.0/3);
            s.px[0] = -GPU_R(0.01)*s.planeNx[0];
            s.py[0] = -GPU_R(0.01)*s.planeNy[0];
            s.pz[0] = -GPU_R(0.01)*s.planeNz[0];
        }
        if (is("moving_wall"))
        {
            s.gasBoundaryUx[0] = GPU_R(0.25);
            s.gasBoundaryUy[0] = GPU_R(-0.5);
            s.gasBoundaryUz[0] = GPU_R(0.75);
        }
    }
    else if (is("internal_hops") || is("periodic_hop"))
    {
        s.px[0] = GPU_R(0.25);
        s.puxOld[0] = GPU_R(3); s.pux[0] = 0; incident.x = GPU_R(1.5);
        s.planeD[0] = s.faceCx[0] = 1;
        s.planeD[1] = s.faceCx[1] = 0;
        s.nCells = 3; dt = 2;
        s.cellFaceKind[0] = s.cellFaceKind[2] = 0;
        s.cellFaceNeighbor[0] = 1; s.cellFaceNeighbor[2] = 2;
        cell = 2; face = 4;
        if (is("periodic_hop"))
        {
            s.nCells = 2; dt = 1.5;
            s.cellFaceKind[0] = 6; s.cellFaceKind[2] = 5;
            s.facePeriodicDx[0] = GPU_R(-3);
            s.faceCx[2] = s.planeD[2] = 5;
            s.faceCx[3] = 4; s.planeD[3] = -4;
            cell = 1; face = 2;
        }
    }
    else if (is("prior_reflection"))
    {
        s.px[0] = 0; s.planeD[0] = s.faceCx[0] = 1;
        s.cellFaceKind[0] = 1;
        s.puxOld[0] = GPU_R(4); s.pux[0] = 0;
        s.puyOld[0] = GPU_R(3); s.puy[0] = GPU_R(1);
        // Segment (2,2,0) reflects with e=0.5 at x=1 before contacting x=-1.
        // Re-averaging the original old velocity and reflected endpoint is wrong.
        incident = {GPU_R(-1), GPU_R(2), 0};
        dt = 3; face = 1;
    }
    else if (std::strncmp(scenario, "invalid_", 8) == 0)
    {
        if (is("invalid_relative_speed")) { s.gasBoundaryUx[0] = incident.x; }
        else if (is("invalid_diameter")) { s.pd[0] = GPU_R(1e-6); }
        else if (is("invalid_temperature")) { s.pT[0] = GPU_R(-1); }
        else if (is("invalid_candidate")) { s.particleStuckCandidateMask[0] = 0; }
        else { require(false, "known invalid scenario"); }
        bool trapped = false;
        try { trackOneParticleLocalFaceWalk(s, 0, dt); }
        catch (const DeviceTrap&) { trapped = true; }
        require(trapped, "invalid input still traps");
        require(s.pux[0] == 1 && s.puxOld[0] == 1 && s.pStuck[0] == 0
            && s.pContactDuration[0] == 0, "validation precedes bound-state mutation");
        return;
    }
    else if (std::strncmp(scenario, "bound_", 6) == 0)
    {
        s.pStuck[0] = is("bound_deposited") ? 1 : (is("bound_rebound") ? 2 : 3);
        s.puxOld[0] = GPU_R(-1); s.puyOld[0] = GPU_R(2); s.puzOld[0] = GPU_R(3);
        s.puy[0] = GPU_R(4); s.puz[0] = GPU_R(5);
        trackOneParticleLocalFaceWalk(s, 0, dt);
        require(s.pux[0] == 0 && s.puy[0] == 0 && s.puz[0] == 0, "existing wall state stays stationary");
        const bool deposited = s.pStuck[0] == 1;
        require(s.puxOld[0] == (deposited ? 0 : -1) && s.puyOld[0] == (deposited ? 0 : 2)
            && s.puzOld[0] == (deposited ? 0 : 3), "existing wall-state rebound seed policy");
        return;
    }
    else { require(is("constant"), "known contact scenario"); }
    checkContact(s, dt, face, cell, incident, model, deposit);
}

int main(int argc, char** argv)
{
    if (argc < 2) { return 2; }
    try { run(argv[1], argc > 2 ? std::atoi(argv[2]) : 1, argc > 3 && std::atoi(argv[3])); }
    catch (const DeviceTrap& trap)
    {
        std::fprintf(stderr, "FAIL: unexpected production trap at line %d (%s, FP%d)\n",
                     trap.line, argv[1], UGKWP_GPU_REAL_BITS);
        return 1;
    }
    std::printf("PASS %s model=%s deposit=%s FP%d %s\n", argv[1], argc > 2 ? argv[2] : "-",
                argc > 3 ? argv[3] : "-", UGKWP_GPU_REAL_BITS, INCIDENT_FSH ? "FSH" : "CHT");
    return 0;
}
