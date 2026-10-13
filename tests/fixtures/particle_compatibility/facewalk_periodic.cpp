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
#define GPU_WALL_COORDINATE(hit, eps, normal) hit - eps*normal
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) hit - shift + eps*normal
#define GPU_RESET_CONTACT_AGE(s, i)
#define GPU_CONTACT_DURATION(value) static_cast<double>(value)
#define GPU_CONTACT_PEAK(value) static_cast<double>(value)
using ContactTime = double;
#else
#define GPU_WALL_COORDINATE(hit, eps, normal) insideFaceCoordinate(hit, normal, eps)
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) insideFaceCoordinate(hit-shift, -normal, eps)
#define GPU_RESET_CONTACT_AGE(s, i) s.pContactAge[i] = GpuTime(0);
#define GPU_CONTACT_DURATION(value) static_cast<GpuTime>(value)
#define GPU_CONTACT_PEAK(value) static_cast<double>(value)
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
    double pDepositionArea[1] = {}, pContactMaximumArea[1] = {}, pContactPeakFraction[1] = {};
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

int main(){DeviceState s;const GpuTime dt=1e17;s.nCells=2;s.cellFaceKind[0]=6;s.cellFaceNeighbor[0]=1;s.facePeriodicDx[0]=GpuReal(dt);s.planeD[2]=GPU_R(32);s.planeD[3]=GpuReal(2*dt);s.faceCx[2]=GPU_R(32);s.faceCx[3]=GpuReal(-2*dt);s.px[0]=GPU_R(-.25);s.pux[0]=s.puxOld[0]=GPU_R(1);trackOneParticleLocalFaceWalk(s,0,dt);printf("periodic hopcap=32 dt=1e17 status=%d cell=%d x=%.17g vx=%.17g\n",s.pStatus[0],s.pCellId[0],double(s.px[0]),double(s.pux[0]));}
