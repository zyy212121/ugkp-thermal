#pragma once
#include "gasTransport/GasPseudoTime.H"
#include "gasTransport/GasConservativeMatching.H"
// Separate primitive/gradient/limiter buffers for PB. No writes to Q while
// evaluating its reference operator; shared geometry and property tables are
// read-only.
struct WallProjectionView
    : ugkwp::GasStateView<double, double, ugkwp::SstCoefficients> {
  ugkwp::GasSpeciesState<double, 2> gasSpecies;
};
template <int M, class State>
__global__
__launch_bounds__(32) void reconstructConservativeWallPool(State *state) {
  if constexpr (ugkwp::GasStateTraits<State>::speciesCount == 2) {
    auto &s = *state;
    auto &m = s.gasBoundaryLayerModel;
    auto &d = m.conservative;
    int worker = blockIdx.x;
    if (worker >= m.workspaceCount || !d.enabled)
      return;
    ugkwp::gaswall::detail::CudaWallTeam team;
    if (team.broadcast(team.leader() &&
                       ugkp::diagnostics::hasFatal(s.diagnostics)))
      return;
    auto &w = static_cast<ugkwp::gaswall::WallWorkspace<double, 2, M> *>(
        m.workspace)[worker];
    for (int slot = worker; slot < s.gasBoundaryLayer.count;
         slot += m.workspaceCount) {
      if (team.leader())
        d.active[slot] = 0;
      team.sync();
      if (d.columns[slot].cells[0] < 0 ||
          !ugkwp::gasBoundaryLayerSlotReady(s, slot))
        continue;
      const auto &in = m.input[slot];
      const double *profile =
          m.candidateHistory + std::size_t(slot) * m.config.nodes * 6;
      for (int i = team.rank(); i < m.config.nodes; i += team.size()) {
        w.y[i] = in.matchingDistance *
                 ::pow(double(i) / (m.config.nodes - 1), m.config.stretch);
        for (int j = 0; j < 6; ++j)
          w.state[i][j] = profile[i * 6 + j];
      }
      team.sync();
      auto profileConfig=m.config;
      profileConfig.coherentDiscretization=ugkwp::gasBoundaryLayerUsesCoherent(m,slot);
      ugkwp::reconstructConservativeWall(s, slot, worker, in, profileConfig, w,
                                         m.output[slot], team);
    }
  }
}
template <class State>
__global__ void initializeWallProjectionView(State *state) {
  if (blockIdx.x || threadIdx.x)
    return;
  auto &s = *state;
  auto &d = s.gasBoundaryLayerModel.conservative;
  auto &v = *static_cast<WallProjectionView *>(d.projectedState);
  // This metadata must exist even when a prior stage failed: subsequent shared
  // kernels read the projected view's diagnostic pointer before any arrays.
  v.diagnostics = s.diagnostics;
  if(ugkp::diagnostics::hasFatal(s.diagnostics))return;
#define UGKWP_GAS_STATE_FIELD(Type, Name) v.Name = s.Name;
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
  if constexpr (ugkwp::GasGeometryCapability<State>::value)
    v.gasGeometry = s.gasGeometry;
  else
    v.gasGeometry = {};
  v.gasBoundaryLayer = {};
  v.gasThermalConductivityConfigured = true;
  v.gasThermalConductivity = 0;
  v.gasMu = 0;
  v.gasViscosity = {}; // PB reference operator remains inviscid.
  v.turbulenceModel = 0;
  v.sstWallTreatment = 3;
  if constexpr (ugkwp::compiledGasSpecies == 2)
    v.gasSpecies = s.gasSpecies;
  v.gasSpecies.diffusivity = nullptr;
  std::size_t k = 0;
#define WALL_PROJECTION_CELL(Name) v.Name = d.projectedCells + (k++) * s.nCells;
#include "gasTransport/ResidentWallCellFields.inc"
#undef WALL_PROJECTION_CELL
  v.gasSpecies.rho = d.projectedCells + k * s.nCells;
  k += 2;
  v.gasSpecies.gradX = d.projectedCells + k * s.nCells;
  k += 2;
  v.gasSpecies.gradY = d.projectedCells + k * s.nCells;
  k += 2;
  v.gasSpecies.gradZ = d.projectedCells + k * s.nCells;
  k += 2;
  v.gasSpecies.limiter = d.projectedCells + (k++) * s.nCells;
  v.gasSpecies.soundSpeed = d.projectedCells + (k++) * s.nCells;
  v.gasSpecies.heatCapacity = d.projectedCells + (k++) * s.nCells;
  v.gasSpecies.gasConstant = d.projectedCells + (k++) * s.nCells;
  v.gasSpecies.cellStatus = d.projectedStatus;
  v.gasSpecies.faceStatus = d.projectedStatus + s.nCells;
  v.riemannBoundaryRho = d.projectedFaces;
  v.riemannBoundaryUx = d.projectedFaces + s.nFaces;
  v.riemannBoundaryUy = d.projectedFaces + 2 * s.nFaces;
  v.riemannBoundaryUz = d.projectedFaces + 3 * s.nFaces;
  v.riemannBoundaryP = d.projectedFaces + 4 * s.nFaces;
  v.riemannBoundaryT = d.projectedFaces + 5 * s.nFaces;
  v.gasSpecies.flux = d.projectedFaces + 6 * s.nFaces;
}
template <class State> __global__ void populateWallProjection(State *state) {
  if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
  auto &s = *state;
  auto &d = s.gasBoundaryLayerModel.conservative;
  auto &v = *static_cast<WallProjectionView *>(d.projectedState);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.nCells) {
    auto q = ugkwp::conservativeCellState(s, i);
    int slot = d.cellSlot[i];
    if (slot >= 0 && d.active[slot]) {
      const auto &col = d.columns[slot];
      q = d.baseInventory[2 * slot + (col.cells[1] == i)];
    }
    v.rho[i] = q[0];
    v.rhoUx[i] = q[1];
    v.rhoUy[i] = q[2];
    v.rhoUz[i] = q[3];
    v.rhoE[i] = q[4];
    v.rhoK[i] = q[5];
    v.rhoOmega[i] = q[6];
    v.k[i] = q[5] / q[0];
    v.omega[i] = q[6] / q[0];
    v.gasSpecies.rho[i] = q[7];
    v.gasSpecies.rho[s.nCells + i] = q[8];
    v.Tgas[i] = s.Tgas[i];
    v.gasSpecies.cellStatus[i] = 0;
  }
  if (i < s.nFaces) {
    v.riemannBoundaryRho[i] = s.riemannBoundaryRho[i];
    v.riemannBoundaryUx[i] = s.riemannBoundaryUx[i];
    v.riemannBoundaryUy[i] = s.riemannBoundaryUy[i];
    v.riemannBoundaryUz[i] = s.riemannBoundaryUz[i];
    v.riemannBoundaryP[i] = s.riemannBoundaryP[i];
    v.riemannBoundaryT[i] = s.riemannBoundaryT[i];
    v.gasSpecies.faceStatus[i] = 0;
  }
}
template <class State>
__global__ void prepareConservativeWallMatching(State *state) {
  if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
  auto &s = *state;
  auto &d = s.gasBoundaryLayerModel.conservative;
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= s.gasBoundaryLayer.count || d.columns[slot].cells[0] < 0)
    return;
  const auto &col = d.columns[slot];
  int c = col.cells[1], f = col.faces[2];
  GasPrimDevice p{};
  double y[2]{};
  if constexpr (ugkwp::compiledGasSpecies == 2)
    reconstructGasMixtureFace(s, c, f, p, y, false);
  auto &m = d.matching[slot];
  m.temperature = p.T;
  m.velocity[0] = p.ux;
  m.velocity[1] = p.uy;
  m.velocity[2] = p.uz;
  for (int j = 0; j < 2; ++j)
    m.massFraction[j] = y[j];
  if constexpr (ugkwp::compiledGasSpecies == 2) {
    if (ugkwp::gasBoundaryLayerUsesCoherent(s.gasBoundaryLayerModel, slot)) {
      const auto physicalFace = riemannFacePrimitiveForGradient(s, c, f);
      ugkwp::sampleConservativeWallThermocomposition(s, f, physicalFace.T, m);
    }
  }
  double weight = s.faceWeight[f];
  int a = s.faceOwner[f], b = s.faceNeighbour[f];
  m.k = weight * s.k[a] + (1 - weight) * s.k[b];
  m.omega = weight * s.omega[a] + (1 - weight) * s.omega[b];
  d.pressure[slot] = p.p;
}
template <class State>
__global__ void assembleConservativeWallFlux(State *state) {
  if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
  auto &s = *state;
  auto &d = s.gasBoundaryLayerModel.conservative;
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= s.nFaces || d.faceSlot[f] < 0)
    return;
  int slot = d.faceSlot[f];
  if (!d.active[slot])
    return;
  auto &v = d.face[3 * slot + d.faceIndex[f]];
  auto &pb = *static_cast<WallProjectionView *>(d.projectedState);
  double q[5], b[5];
  if (!computeRiemannGasFaceFluxDevice<false>(s, f, q[0], q[1], q[2], q[3],
                                              q[4], false, true) ||
      !computeRiemannGasFaceFluxDevice<false>(pb, f, b[0], b[1], b[2], b[3],
                                              b[4], false, true))
    return;
  double result[9]{};
  for (int j = 0; j < 5; ++j)
    result[j] = q[j] - b[j] + v.orientation * (v.reference[j] + v.diffusion[j]);
  int own = s.faceOwner[f], nei = coupledFaceNeighbour(s, f),
      up = result[0] >= 0 ? own : nei;
  auto trace = v.base;
  const auto &in = s.gasBoundaryLayerModel.input[slot];
  double blowing = in.massFlux[0] + in.massFlux[1];
  if (blowing != 0 && up >= 0 && d.cellSlot[up] == slot)
    trace = v.trace[result[0] / v.orientation >= 0 ? 0 : 1];
  else if (up >= 0) {
    trace.T = s.Tgas[up];
    trace.k = s.k[up];
    trace.omega = s.omega[up];
    for (int j = 0; j < 2; ++j)
      trace.Y[j] = s.gasSpecies.rho[j * s.nCells + up] / s.rho[up];
  }
  v.kUpwind = trace.k;
  v.omegaUpwind = trace.omega;
  for (int j = 0; j < 2; ++j) {
    double old = s.gasSpecies.flux[j * s.nFaces + f] -
                 pb.gasSpecies.flux[j * s.nFaces + f] +
                 v.orientation * v.reference[7 + j];
    double adv = result[0] * trace.Y[j];
    result[4] += ugkwp::speciesH(j, trace.T, in.model.thermo) * (adv - old);
    s.gasSpecies.flux[j * s.nFaces + f] =
        adv + v.orientation * v.diffusion[7 + j];
  }
  s.gasPhiRho[f] = result[0];
  s.gasPhiRhoUx[f] = result[1];
  s.gasPhiRhoUy[f] = result[2];
  s.gasPhiRhoUz[f] = result[3];
  s.gasPhiRhoE[f] = result[4];
}
template <class State>
__global__ void conservativeKFluxBudget(State *state, double stepDt) {
  if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
  auto &s = *state;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= s.nCells)
    return;
  const double dt = ugkwp::gasCellTimeStep(s, c, stepDt);

  auto &d = s.gasBoundaryLayerModel.conservative;
  double oldVolume = s.V[c], newVolume = s.V[c];
  const bool moving = ugkwp::gasMovingGeometry(s);
  if (moving) {
    bool available = false;
    if (!ugkwp::gasGeometryCellVolumes(s, c, dt, oldVolume, newVolume,
                                       &available))
      return;
  }
  double outgoing = 0, div = 0;
  for (int j = 0; j < s.cellPlaneCount[c]; ++j) {
    int f = s.cellFaceId[s.cellPlaneStart[c] + j];
    double sign = s.faceOwner[f] == c ? 1 : -1;
    outgoing += fmax(0., sign * s.sstPhiRhoK[f]);
    double volumeFlux = s.gasPhiRho[f] / sstFaceDensity(s, f);
    if (moving) {
      ugkwp::GasFaceFrame<double> frame;
      bool available = false;
      if (!ugkwp::gasGeometryFaceFrame(s, f, dt, frame, &available))
        return;
      volumeFlux += frame.meshVolumeRate;
    }
    div += sign * volumeFlux / oldVolume;
  }
  double source = 0, omegaSource = 0;
  if (!sstSourcesForCell(s, c, div, source, omegaSource))
    return;
  d.sourceK[c] = source;
  d.sourceOmega[c] = omegaSource;
  double available =
      (s.rhoK[c] - s.rho[c] * s.sstKMin + dt * source) * oldVolume;
  d.kFluxScale[c] = dt > 0 && outgoing > 0 && dt * outgoing > available
                        ? fmax(0., .999 * available / (dt * outgoing))
                        : 1;
}
template <class State>
__global__ void applyConservativeKFluxBudget(State *state) {
  if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
  auto &s = *state;
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= s.nFaces)
    return;
  int donor =
      s.sstPhiRhoK[f] >= 0 ? s.faceOwner[f] : coupledFaceNeighbour(s, f);
  if (donor >= 0)
    s.sstPhiRhoK[f] *= s.gasBoundaryLayerModel.conservative.kFluxScale[donor];
}
inline int prepareConservativeProjection(DeviceState *s) {
  auto &d = s->gasBoundaryLayerModel.conservative;
  if (!d.enabled)
    return 0;
  int cb = s->fixedCellBlockThreads, fb = s->fixedFaceBlockThreads,
      cg = (s->nCells + cb - 1) / cb,
      bg = (s->nFaces - s->nInternalFaces + fb - 1) / fb;
  auto *v = static_cast<WallProjectionView *>(d.projectedState);
  initializeWallProjectionView<<<1, 1, 0, s->gasCaptureStream>>>(
      s->deviceState);
  populateWallProjection<<<(std::max(s->nFaces, s->nCells) + cb - 1) / cb, cb,
                           0, s->gasCaptureStream>>>(s->deviceState);
  recoverGasPrimitivesKernel<<<cg, cb, 0, s->gasCaptureStream>>>(v);
  if (bg)
    updateRiemannBoundaryMirrorKernel<<<bg, fb, 0, s->gasCaptureStream>>>(v);
  computeGasPrimitiveGradientsKernel<<<cg, cb, 0, s->gasCaptureStream>>>(v);
  computeGasGradientLimiterKernel<<<cg, cb, 0, s->gasCaptureStream>>>(v);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    setLastError("prepare PB reference operator", error);
    return 1;
  }
  return 0;
}
