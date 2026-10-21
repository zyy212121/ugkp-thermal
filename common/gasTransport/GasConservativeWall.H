#pragma once
#include "../gasWall/reconstruction/ConservativeProfile.H"
#include "GasGeometryValidation.H"
namespace ugkwp {
struct ConservativeWallColumn {
  int cells[2]{-1, -1}, faces[3]{-1, -1, -1};
  double edge[3]{};
};
struct ConservativeWallFace {
  gaswall::reconstruction::Q reference{}, diffusion{};
  gaswall::reconstruction::Primitive trace[2]{}, base{};
  double orientation = 0, meshNormal = 0, kUpwind = 0, omegaUpwind = 0;
};
struct ConservativeWallData {
  bool enabled = false;
  void *projectedState = nullptr;
  double *projectedCells = nullptr;
  double *projectedFaces = nullptr;
  int *projectedStatus = nullptr;

  const ConservativeWallColumn *columns =
      nullptr; // indexed by the existing wall slot
  const int *cellSlot = nullptr;
  const int *faceSlot = nullptr;
  const int *faceIndex = nullptr;
  gaswall::reconstruction::Q *baseInventory = nullptr; // two entries per wall
  ConservativeWallFace *face = nullptr;                // three entries per wall
  double *sourceK = nullptr;
  double *sourceOmega = nullptr;
  double *kFluxScale = nullptr;
  double *sourceActivity = nullptr;
  double *momentResidual = nullptr;
  int *active =
      nullptr; // stage-local, all four coupling consumers use this flag
  int *fallbacks = nullptr;
  gaswall::reconstruction::ProfileScratch *scratch =
      nullptr; // bounded worker pool
  gaswall::WallMatchingState<double, 2> *matching = nullptr;
  double *pressure = nullptr;
};
template <class State>
UGKWP_GAS_HD inline gaswall::reconstruction::Q
conservativeCellState(const State &s, int c) {
  return {s.rho[c],      s.rhoUx[c],          s.rhoUy[c],
          s.rhoUz[c],    s.rhoE[c],           s.rhoK[c],
          s.rhoOmega[c], s.gasSpecies.rho[c], s.gasSpecies.rho[s.nCells + c]};
}
template <class State>
UGKWP_GAS_HD inline gaswall::reconstruction::MacroGradients
conservativeCellGradients(const State &s, int c) {
  gaswall::reconstruction::MacroGradients g;
  const double u[3][3] = {{s.gradUxX[c], s.gradUxY[c], s.gradUxZ[c]},
                          {s.gradUyX[c], s.gradUyY[c], s.gradUyZ[c]},
                          {s.gradUzX[c], s.gradUzY[c], s.gradUzZ[c]}};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      g.U[a][b] = u[a][b];
  g.T[0] = s.gradTX[c];
  g.T[1] = s.gradTY[c];
  g.T[2] = s.gradTZ[c];
  g.k[0] = s.gradKX[c];
  g.k[1] = s.gradKY[c];
  g.k[2] = s.gradKZ[c];
  g.omega[0] = s.gradOmegaX[c];
  g.omega[1] = s.gradOmegaY[c];
  g.omega[2] = s.gradOmegaZ[c];
  for (int j = 0; j < 2; ++j) {
    int i = j * s.nCells + c;
    g.Y[j][0] = s.gasSpecies.gradX[i];
    g.Y[j][1] = s.gasSpecies.gradY[i];
    g.Y[j][2] = s.gasSpecies.gradZ[i];
  }
  return g;
}
template <int M, class State, class Team>
UGKWP_GAS_HD inline void reconstructConservativeWall(
    State &s, int slot, int worker, const gaswall::WallInput<double, 2> &in,
    const gaswall::WallModelConfig<double> &cfg,
    const gaswall::WallWorkspace<double, 2, M> &w,
    const gaswall::WallOutput<double, 2> &out, const Team &team) {
  auto &data = s.gasBoundaryLayerModel.conservative;
  if (!data.enabled || data.columns[slot].cells[0] < 0)
    return;
  const auto &col = data.columns[slot];
  gaswall::reconstruction::Profile p(in, cfg, w, data.scratch[worker].transformSlope, team);
  p.setWallFlux(out);
  for (int i = 0; i < 2; ++i) {
    gaswall::reconstruction::Segment seg{};
    seg.lo = col.edge[i];
    seg.hi = col.edge[i + 1];
    seg.target = conservativeCellState(s, col.cells[i]);
    seg.algebraicOmega = i == 0;
    p.segments.push_back(seg);
    p.setMacroGradients(i, conservativeCellGradients(s, col.cells[i]));
  }
  double normals[3]{};
  normals[0] = in.velocity[0] * in.normal[0] + in.velocity[1] * in.normal[1] +
               in.velocity[2] * in.normal[2];
  gaswall::detail::LayerContext<double, 2> ctx;
  gaswall::detail::makeContext(in, cfg, ctx);
  normals[0] += ctx.mass / out.traceDensity;
  for (int j = 1; j < 3; ++j) {
    int f = col.faces[j], a = s.faceOwner[f], b = s.faceNeighbour[f];
    double t = s.faceWeight[f];
    const double u[3] = {t * s.Ux[a] + (1 - t) * s.Ux[b],
                         t * s.Uy[a] + (1 - t) * s.Uy[b],
                         t * s.Uz[a] + (1 - t) * s.Uz[b]};
    for (int d = 0; d < 3; ++d)
      normals[j] += u[d] * in.normal[d];
  }
  for (int i = 0; i < 2; ++i) {
    p.segments[i].normalLo = normals[i];
    p.segments[i].normalHi = normals[i + 1];
  }
  p.reconstructInternalTraces();
  if (!p.fitTeam(data.scratch[worker], team)) {
    if (team.leader())
      ++data.fallbacks[slot];
    return;
  }
  if (team.leader())
    data.active[slot] = 1;
  if (team.leader())
    data.momentResidual[slot] = p.maxResidual;
  for (int i = 0; i < 2; ++i) {
    auto source = p.sourceTeam(i, team, cfg.coherentDiscretization ? 1 : 4);
    if (team.leader()) {
      auto q = p.baseAverage(p.segments[i]);
      if (i == 0)
        q[6] = q[0] * out.ownerOmega;
      data.baseInventory[2 * slot + i] = q;
      const double length = col.edge[i + 1] - col.edge[i];
      data.sourceK[col.cells[i]] = source[0] / length;
      data.sourceActivity[col.cells[i]] = source[2] / length;
      data.sourceOmega[col.cells[i]] = i ? source[1] / length : 0;
    }
  }
  for (int j = team.rank(); j < 3; j += team.size()) {
    int f = col.faces[j];
    auto &v = data.face[3 * slot + j];
    double y = col.edge[j];
    v.orientation = s.Sfx[f] * in.normal[0] + s.Sfy[f] * in.normal[1] +
                    s.Sfz[f] * in.normal[2];
    double mesh = 0;
    if constexpr (GasGeometryCapability<State>::value) {
      if (gasMovingGeometry(s)) {
        GasFaceFrame<double> frame;
        bool available = false;
        gasGeometryFaceFrame(s, f, s.gasGeometry.interval, frame, &available);
        if (available)
          mesh = frame.wx * in.normal[0] + frame.wy * in.normal[1] +
                 frame.wz * in.normal[2];
      }
    }
    v.meshNormal = mesh;
    v.base = p.base(y);
    v.reference = p.inviscid(v.base, mesh);
    v.diffusion = p.diffusiveFlux(y, mesh);
    v.trace[0] = p.point(y, -1);
    v.trace[1] = p.point(y, 1);
  }
  team.sync();
}
} // namespace ugkwp
