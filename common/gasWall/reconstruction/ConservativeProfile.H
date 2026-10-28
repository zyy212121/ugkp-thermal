#pragma once
#include "../WallModel.H"
#include "PhysicalPrimitiveFlux.H"
#include "ProfileStorage.H"
#include <algorithm>
#include <cmath>
namespace ugkwp {
namespace gaswall {
namespace reconstruction {
using Input = ugkwp::gaswall::WallInput<double, 2>;
using Config = ugkwp::gaswall::WallModelConfig<double>;
using Work = ugkwp::gaswall::WallWorkspace<double, 2, 128>;
namespace d = ugkwp::gaswall::detail;
using Q = Array<double, 9>; // rho,rhoU3,rhoE,rhok,rhoomega,rhoY0,rhoY1
struct Primitive {
  double rho, U[3], T, k, omega, Y[2];
};
struct MacroGradients {
  double U[3][3]{}, T[3]{}, k[3]{}, omega[3]{}, Y[2][3]{};
};
struct Segment {
  double lo, hi;
  Q target;
  double aRho = 0, aU[3]{}, aT = 0, aK = 0, aOmega = 0;
  bool algebraicOmega = false;
  double normalLo = std::numeric_limits<double>::quiet_NaN(),
         normalHi = std::numeric_limits<double>::quiet_NaN(), aY = 0;
  bool endpointOverride = false;
  Primitive left{}, right{}, baseLeft{}, baseRight{};
  double seedK = 0;
  bool hasMacroGradients = false;
  MacroGradients macroGradients{};
};
struct Sample {
  Primitive p;
  double weight, b, by, rho, U[3], Y[2], kinetic;
  double y=0, speciesBaseLogit=0, speciesBasis[2]{};
};
struct ProfileScratch {
  Sample samples[128 * 8];
  // Worker-owned active slope cache, shared by the same node team as samples.
  double transformSlope[128][3];
  // One bounded worker buffer. Points are evaluated cooperatively, then the
  // original quadrature-order FP64 sum is retained by the leader.
  Q orderedTerms[128 * 8];
  Q baseMeans[2];
};
struct Profile {
  struct WorkView {
    const double *y;
    const double (*state)[6];
  };
  // Optional borrowed coherent slope cache; no cache is needed by the legacy
  // representation. Coherent callers without storage compute slopes on demand.
  const double (*transformSlope)[3];
  const Input &in;
  const Config &cfg;
  WorkView w;
  d::LayerContext<double, 2> ctx;
  FixedVector<Segment, 2> segments;
  double maxResidual = 0, wallSlopeDelta[3]{}, wallSpeciesSlopeDelta0 = 0;
  // Frozen-mixture/no-blow/equal-D accuracy choice; other existing paths remain.
  bool affineHelpersOk=true,affineRawFinite=true;
  int affineSpeciesMode=0, affineMomentEvaluations=0, affineIterations=0, affineSamples=0;
  double affineSpeciesA=0, affineSpeciesB=0, affineInterface=0, affineMobilityLeft=0, affineMobilityRight=0;
  template <int M, class Team = d::SerialWallTeam>
  UGKWP_GAS_HD Profile(const Input &i, const Config &c,
                       const ugkwp::gaswall::WallWorkspace<double, 2, M> &v,
                       double (*slopes)[3] = nullptr, const Team &team = Team{})
      : transformSlope(c.coherentDiscretization ? slopes : nullptr), in(i), cfg(c), w{v.y, v.state} {
    d::makeContext(in, cfg, ctx);
    // All cooperative callers supply the same storage and enter this barrier.
    // No cache is required by the existing three-argument scalar interface.
    if (cfg.coherentDiscretization && slopes) {
      for (int index = team.rank(); index < 3 * cfg.nodes;
           index += team.size()) {
        const int channel = index / cfg.nodes, node = index % cfg.nodes;
        slopes[node][channel] = pchipSlope(node, channel == 2 ? 2 : channel + 4);
      }
      team.sync();
    }
  }
  UGKWP_GAS_HD double pchipSlope(int i, int j) const {
    double slope = 0;
    if (i == 0)
      slope = (w.state[1][j] - w.state[0][j]) / (w.y[1] - w.y[0]);
    else if (i == cfg.nodes - 1) {
      if (i == 1)
        return (w.state[i][j] - w.state[i - 1][j]) / (w.y[i] - w.y[i - 1]);
      const double h0 = w.y[i] - w.y[i - 1], h1 = w.y[i - 1] - w.y[i - 2],
                   d0 = (w.state[i][j] - w.state[i - 1][j]) / h0,
                   d1 = (w.state[i - 1][j] - w.state[i - 2][j]) / h1;
      slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (slope * d0 <= 0)
        slope = 0;
      else if (d0 * d1 < 0 && std::abs(slope) > 3 * std::abs(d0))
        slope = 3 * d0;
    } else {
      const double hl = w.y[i] - w.y[i - 1], hr = w.y[i + 1] - w.y[i],
                   dl = (w.state[i][j] - w.state[i - 1][j]) / hl,
                   dr = (w.state[i + 1][j] - w.state[i][j]) / hr;
      if ((dl > 0 && dr > 0) || (dl < 0 && dr < 0)) {
        const double wl = 2 * hr + hl, wr = hr + 2 * hl;
        slope = (wl + wr) / (wl / dl + wr / dr);
      }
    }
    return slope;
  }
  UGKWP_GAS_HD void
  setWallFlux(const ugkwp::gaswall::WallOutput<double, 2> &out) {
    d::LayerPoint<double, 2> wall;
    d::pointState(in, cfg, ctx, w.state[0], wall);
    double dy = w.y[1] - w.y[0];
    for (int t = 0; t < 2; ++t) {
      double tau = 0;
      for (int j = 0; j < 3; ++j)
        tau += out.traction[j] * ctx.tangent[t][j];
      wallSlopeDelta[t] =
          tau / wall.mu - (w.state[1][t] - w.state[0][t]) / dy;
    }
    double conductivity =
        in.model.molecularPrandtl > 0
            ? wall.mu * wall.cp / wallMolecularPrandtl(in,wall.Y)
            : in.model.conductivity;
    wallSlopeDelta[2] = -out.conductiveHeatFlux / conductivity -
                        (w.state[1][2] - w.state[0][2]) / dy;
    d::LayerPoint<double, 2> edge;
    d::pointState(in, cfg, ctx, w.state[1], edge);
    double D = wall.rho * (wallMolecularDiffusivity(in,0,wall.T,in.pressure) * wall.Y[1] +
                           wallMolecularDiffusivity(in,1,wall.T,in.pressure) * wall.Y[0]);
    if (D > 0)
      wallSpeciesSlopeDelta0 = -(in.massFlux[0] - ctx.mass * wall.Y[0]) / D -
                               (edge.Y[0] - wall.Y[0]) / dy;
  }
  // Snapshot only coefficients independent of auxiliary q/traction. The caller
  // owns its lifetime and must rebuild it when the input or workspace changes.
  struct WallFluxBasis {
    double tangentSlope[2], temperatureSlope, conductivity, viscosity, speciesSlope = 0;
    bool correctSpecies = false;
  };
  UGKWP_GAS_HD WallFluxBasis wallFluxBasis() const {
    WallFluxBasis basis;
    d::LayerPoint<double, 2> wall, edge;
    d::pointState(in, cfg, ctx, w.state[0], wall);
    const double dy = w.y[1] - w.y[0];
    for (int t = 0; t < 2; ++t)
      basis.tangentSlope[t] = (w.state[1][t] - w.state[0][t]) / dy;
    basis.viscosity = wall.mu;
    basis.conductivity = in.model.molecularPrandtl > 0
        ? wall.mu * wall.cp / wallMolecularPrandtl(in,wall.Y)
        : in.model.conductivity;
    basis.temperatureSlope = (w.state[1][2] - w.state[0][2]) / dy;
    d::pointState(in, cfg, ctx, w.state[1], edge);
    const double D = wall.rho * (wallMolecularDiffusivity(in,0,wall.T,in.pressure) * wall.Y[1] +
                                wallMolecularDiffusivity(in,1,wall.T,in.pressure) * wall.Y[0]);
    basis.correctSpecies = D > 0;
    if (basis.correctSpecies)
      basis.speciesSlope = -(in.massFlux[0] - ctx.mass * wall.Y[0]) / D -
                           (edge.Y[0] - wall.Y[0]) / dy;
    return basis;
  }
  UGKWP_GAS_HD void setWallFlux(const ugkwp::gaswall::WallOutput<double, 2> &out,
                                const WallFluxBasis &basis) {
    for (int t = 0; t < 2; ++t) {
      double tau = 0;
      for (int j = 0; j < 3; ++j)
        tau += out.traction[j] * ctx.tangent[t][j];
      wallSlopeDelta[t] = tau / basis.viscosity - basis.tangentSlope[t];
    }
    wallSlopeDelta[2] = -out.conductiveHeatFlux / basis.conductivity -
                        basis.temperatureSlope;
    if (basis.correctSpecies)
      wallSpeciesSlopeDelta0 = basis.speciesSlope;
  }
  UGKWP_GAS_HD double nodeSlope(int i, int j) const {
    if (j >= 4 || j == 2)
      return transformSlope ? transformSlope[i][j == 2 ? 2 : j - 4]
                            : pchipSlope(i, j);
    if (i == cfg.nodes - 1)
      return (w.state[i][j] - w.state[i - 1][j]) / (w.y[i] - w.y[i - 1]);
    return d::logShearDerivative(w.state[i - 1][j], w.state[i][j],
                                w.state[i + 1][j], w.y[i] - w.y[i - 1],
                                w.y[i + 1] - w.y[i], w.y[i]);
  }
  UGKWP_GAS_HD Array<double, 2> hermite(int l, int j, double t) const {
    double h = w.y[l + 1] - w.y[l], a = w.state[l][j], b = w.state[l + 1][j],
           ga = nodeSlope(l, j), gb = nodeSlope(l + 1, j);
    return {(2*t*t*t-3*t*t+1)*a+(-2*t*t*t+3*t*t)*b+
                (t*t*t-2*t*t+t)*h*ga+(t*t*t-t*t)*h*gb,
            6*t*(1-t)*(b-a)/h+(3*t*t-4*t+1)*ga+(3*t*t-2*t)*gb};
  }
  // Selected coherent representation: log-shear U and PCHIP sqrt(k), z.
  // Temperature remains piecewise linear outside the first interval.
  UGKWP_GAS_HD bool hermiteField(int j) const {
    return cfg.coherentDiscretization && (j < 2 || j == 4 || j == 5);
  }
  UGKWP_GAS_HD Primitive base(double y) const {
    int l = 0;
    while (l + 1 < cfg.nodes - 1 && w.y[l + 1] < y)
      ++l;
    double t = (y - w.y[l]) / (w.y[l + 1] - w.y[l]);
    double state[6];
    for (int j = 0; j < 6; ++j)
      state[j] = w.state[l][j] + t * (w.state[l + 1][j] - w.state[l][j]);
    if (l > 0)
      for (int j = 0; j < 6; ++j)
        if (hermiteField(j))
          state[j] = hermite(l, j, t)[0];
    if (l == 0) {
      double shape = y * std::pow(1 - y / w.y[1], 2);
      for (int j = 0; j < 3; ++j)
        state[j] += wallSlopeDelta[j] *
                    (cfg.coherentDiscretization && j == 2
                         ? y * (1 - y / w.y[1]) : shape);
      state[3] +=
          (ctx.independent[0] == 0 ? 1 : -1) * wallSpeciesSlopeDelta0 * shape;
      d::LayerPoint<double, 2> wall;
      d::pointState(in, cfg, ctx, w.state[0], wall);
      double slope = std::sqrt(wall.rho * in.model.sstCoefficients.beta1 /
                               (6 * wall.mu)),
             ell = w.state[0][5] / slope,
             power = (1 + std::sqrt(1 + 24 * in.model.sstCoefficients.betaStar /
                                            in.model.sstCoefficients.beta1)) /
                     2;
      if (cfg.enableSst)
        state[4] =
            std::sqrt(w.state[1][4] * w.state[1][4] *
                      d::nearWallKShape(y, w.y[1], ell, power,
                                        ctx.mass / (2 * wall.mu)));
    }
    d::LayerPoint<double, 2> p;
    d::pointState(in, cfg, ctx, state, p);
    Primitive v{};
    v.rho = p.rho;
    v.T = p.T;
    v.k = p.k;
    v.omega = p.omega;
    for (int j = 0; j < 2; ++j)
      v.Y[j] = p.Y[j];
    for (int j = 0; j < 3; ++j) {
      v.U[j] = p.normalVelocity * in.normal[j];
      for (int a = 0; a < 2; ++a)
        v.U[j] += (ctx.wallU[a] + p.u[a]) * ctx.tangent[a][j];
    }
    return v;
  }
  UGKWP_GAS_HD void setEndpoints(int i, const Primitive &left,
                                 const Primitive &right) {
    auto &s = segments[i];
    s.endpointOverride = true;
    s.left = left;
    s.right = right;
    s.baseLeft = base(s.lo);
    s.baseRight = base(s.hi);
  }
  UGKWP_GAS_HD Primitive primitiveAverage(const Q &q) const {
    Primitive p{};
    p.rho = q[0];
    double kinetic = 0;
    for (int j = 0; j < 3; ++j) {
      p.U[j] = q[j + 1] / q[0];
      kinetic += q[j + 1] * q[j + 1] / (2 * q[0]);
    }
    for (int j = 0; j < 2; ++j)
      p.Y[j] = q[7 + j] / q[0];
    p.k = q[5] / q[0];
    p.omega = q[6] / q[0];
    double mass[2] = {q[7], q[8]};
    p.T = in.temperature;
    ugkwp::ThermoInversionControls<double> controls;
    bool available = false;
    ugkwp::invertMixtureEnergy(mass, q[4] - kinetic, in.model.thermo, controls,
                               p.T, &available);
    return p;
  }
  UGKWP_GAS_HD void reconstructInternalTraces() {
    int n = segments.size();
    Primitive a[2]{}, b[2]{};
    for (int i = 0; i < n; ++i) {
      a[i] = primitiveAverage(segments[i].target);
      b[i] = primitiveAverage(baseAverage(segments[i]));
    }
    reconstructInternalTracesFromAverages(a, b);
  }
  template <class Team>
  UGKWP_GAS_HD void reconstructInternalTracesTeam(ProfileScratch &scratch,
                                                  const Team &team) {
    Primitive a[2]{}, b[2]{};
    for (int i = 0; i < int(segments.size()); ++i) {
      auto q = baseAverageTeam(segments[i], scratch, team);
      for (int j = 0; j < 9; ++j)
        q[j] = team.broadcast(q[j]);
      if (team.leader())
        scratch.baseMeans[i] = q;
      a[i] = primitiveAverage(segments[i].target);
      b[i] = primitiveAverage(q);
    }
    team.sync();
    reconstructInternalTracesFromAverages(a, b);
  }
  UGKWP_GAS_HD void reconstructInternalTracesFromAverages(
      const Primitive *a, const Primitive *b) {
    const int n = segments.size();
    Primitive trace[3]{};
    trace[0] = base(segments.front().lo);
    trace[n] = base(segments.back().hi);
    for (int i = 1; i < n; ++i) {
      double y = segments[i].lo,
             cl = (segments[i - 1].lo + segments[i - 1].hi) / 2,
             cr = (segments[i].lo + segments[i].hi) / 2,
             t = (y - cl) / (cr - cl);
      auto p = base(y);
      auto ratio = [&](double al, double bl, double ar, double br) {
        return (1 - t) * al / bl + t * ar / br;
      };
      p.rho *= ratio(a[i - 1].rho, b[i - 1].rho, a[i].rho, b[i].rho);
      p.T *= ratio(a[i - 1].T, b[i - 1].T, a[i].T, b[i].T);
      for (int j = 0; j < 3; ++j)
        p.U[j] += (1 - t) * (a[i - 1].U[j] - b[i - 1].U[j]) +
                  t * (a[i].U[j] - b[i].U[j]);
      if (b[i - 1].k > 0 && b[i].k > 0)
        p.k *= ratio(a[i - 1].k, b[i - 1].k, a[i].k, b[i].k);
      else
        p.k = (1 - t) * a[i - 1].k + t * a[i].k;
      if (!segments[i - 1].algebraicOmega && b[i - 1].omega > 0 &&
          b[i].omega > 0)
        p.omega *=
            ratio(a[i - 1].omega, b[i - 1].omega, a[i].omega, b[i].omega);
      double ysum = 0;
      for (int j = 0; j < 2; ++j) {
        if (b[i - 1].Y[j] > 0 && b[i].Y[j] > 0)
          p.Y[j] *= ratio(a[i - 1].Y[j], b[i - 1].Y[j], a[i].Y[j], b[i].Y[j]);
        else
          p.Y[j] = (1 - t) * a[i - 1].Y[j] + t * a[i].Y[j];
        ysum += p.Y[j];
      }
      for (int j = 0; j < 2; ++j)
        p.Y[j] /= ysum;
      trace[i] = p;
    }
    for (int i = 0; i < n; ++i)
      setEndpoints(i, trace[i], trace[i + 1]);
  }
  UGKWP_GAS_HD static double logRatio(double target, double reference) {
    return target > 0 && reference > 0 ? std::log(target / reference) : 0;
  }
  UGKWP_GAS_HD static double bubble(double x) { return 6 * x * (1 - x); }
  UGKWP_GAS_HD static double speciesBubble(double x, bool first) {
    return first ? 12 * x * x * (1 - x) : bubble(x);
  }
  UGKWP_GAS_HD static double fraction(double y, double a, double b) {
    return a >= 0 ? (y + a * b) / (1 + a * b) : y / (1 - a * b);
  }
  UGKWP_GAS_HD static double speciesLogistic(double z) {
    if(z>=0){double e=::exp(-z);return 1/(1+e);}double e=::exp(z);return e/(1+e);
  }
  UGKWP_GAS_HD Array<double,2> affineCoordinate(double y,int side=0) const {
    int i=segment(y,side);const auto& s=segments[i];double h=s.hi-s.lo,t=(y-s.lo)/h;
    return i?Array<double,2>{affineInterface+(1-affineInterface)*t,(1-affineInterface)/h}:Array<double,2>{affineInterface*t,affineInterface/h};
  }
  UGKWP_GAS_HD double affineValue(double y,double baseY,int side=0) const {
    if(affineSpeciesMode==2)return 0;if(affineSpeciesMode==3)return 1;
    double r=affineCoordinate(y,side)[0];return speciesLogistic(::log(baseY/(1-baseY))+affineSpeciesA*(1-r*r)+affineSpeciesB*r*r*(1-r));
  }
  UGKWP_GAS_HD double affineGradient(double y,double baseY,double baseGradient,int side=0) const {
    if(affineSpeciesMode!=1)return 0;auto c=affineCoordinate(y,side);double r=c[0],Y=affineValue(y,baseY,side);
    return Y*(1-Y)*(baseGradient/(baseY*(1-baseY))+(-2*affineSpeciesA*r+affineSpeciesB*(2*r-3*r*r))*c[1]);
  }
  UGKWP_GAS_HD Primitive pointIn(double y, const Segment &s,
                                  const Primitive *basePoint = nullptr) const {
    auto p = basePoint ? *basePoint : base(y);
    const double speciesBaseY=p.Y[0];
    double x = (y - s.lo) / (s.hi - s.lo);
    if (s.endpointOverride) {
      auto lift = [&](double a, double b, double ar, double br) {
        return std::exp((1 - x) * logRatio(a, ar) + x * logRatio(b, br));
      };
      p.rho *= lift(s.left.rho, s.right.rho, s.baseLeft.rho, s.baseRight.rho);
      p.T *= lift(s.left.T, s.right.T, s.baseLeft.T, s.baseRight.T);
      if (s.baseLeft.k == 0 && s.baseRight.k == 0) {
        double t = s.lo == 0 ? x * x * (3 - 2 * x) : x;
        p.k = (1 - t) * s.left.k + t * s.right.k;
      } else
        p.k *= lift(s.left.k, s.right.k, s.baseLeft.k, s.baseRight.k);
      double ol = logRatio(s.left.omega, s.baseLeft.omega);
      if (s.lo == 0 && s.algebraicOmega)
        ol = std::log(s.baseLeft.rho / s.left.rho);
      p.omega *= std::exp((1 - x) * ol +
                          x * logRatio(s.right.omega, s.baseRight.omega));
      for (int j = 0; j < 3; ++j)
        p.U[j] += (1 - x) * (s.left.U[j] - s.baseLeft.U[j]) +
                  x * (s.right.U[j] - s.baseRight.U[j]);
      if (s.left.Y[0] != s.baseLeft.Y[0] || s.right.Y[0] != s.baseRight.Y[0]) {
        double t = s.lo == 0 ? x * x * (3 - 2 * x) : x;
        auto map = [](double value, double target, double reference) {
          return target >= reference
                     ? (reference < 1
                            ? value + (target - reference) * (1 - value) /
                                          (1 - reference)
                            : value)
                     : (reference > 0 ? value * target / reference : value);
        };
        p.Y[0] = (1 - t) * map(p.Y[0], s.left.Y[0], s.baseLeft.Y[0]) +
                 t * map(p.Y[0], s.right.Y[0], s.baseRight.Y[0]);
        p.Y[1] = 1 - p.Y[0];
      }
    }
    if (std::isfinite(s.normalLo) && std::isfinite(s.normalHi)) {
      double un = 0;
      for (int j = 0; j < 3; ++j)
        un += p.U[j] * in.normal[j];
      double prescribed = (1 - x) * s.normalLo + x * s.normalHi;
      for (int j = 0; j < 3; ++j)
        p.U[j] += (prescribed - un) * in.normal[j];
    }
    double b = bubble(x);
    p.rho *= std::exp(s.aRho * b);
    p.Y[0] = fraction(p.Y[0], s.aY, speciesBubble(x, s.lo == 0));
    p.Y[1] = 1 - p.Y[0];
    for (int j = 0; j < 3; ++j)
      p.U[j] += s.aU[j] * b;
    p.T *= std::exp(s.aT * b);
    p.k = (p.k + s.seedK * speciesBubble(x, s.lo == 0)) * std::exp(s.aK * b);
    if (!s.algebraicOmega)
      p.omega *= std::exp(s.aOmega * b);
    if(affineSpeciesMode){p.Y[0]=affineValue(y,speciesBaseY,s.lo==0?-1:1);p.Y[1]=1-p.Y[0];}
    return p;
  }
  UGKWP_GAS_HD int segment(double y, int side = 0) const {
    for (int i = 0; i < int(segments.size()); ++i)
      if (y < segments[i].hi || (y == segments[i].hi && side <= 0))
        return i;
    return segments.size() - 1;
  }
  UGKWP_GAS_HD Primitive point(double y, int side = 0) const {
    return pointIn(y, segments[segment(y, side)]);
  }
  UGKWP_GAS_HD Q conserved(const Primitive &p) const {
    double u2 = 0;
    for (double u : p.U)
      u2 += u * u;
    return {p.rho,
            p.rho * p.U[0],
            p.rho * p.U[1],
            p.rho * p.U[2],
            p.rho * (ugkwp::mixtureEnergy(p.Y, p.T, in.model.thermo) + u2 / 2),
            p.rho * p.k,
            p.rho * p.omega,
            p.rho * p.Y[0],
            p.rho * p.Y[1]};
  }
  template <class Fn>
  UGKWP_GAS_HD void quadrature(const Segment &s, Fn fn,
                               int subdivisions = 1) const {
    constexpr double x[4] = {.1834346424956498, .5255324099163290,
                             .7966664774136267, .9602898564975363},
                     wt[4] = {.3626837833783620, .3137066458778873,
                              .2223810344533745, .1012285362903763};
    for (int i = 0; i + 1 < cfg.nodes; ++i) {
      double lo = s.lo > w.y[i] ? s.lo : w.y[i],
             hi = s.hi < w.y[i + 1] ? s.hi : w.y[i + 1];
      if (!(hi > lo))
        continue;
      for (int k = 0; k < subdivisions; ++k) {
        double a = lo + (hi - lo) * k / subdivisions,
               b = lo + (hi - lo) * (k + 1) / subdivisions;
        for (int g = 0; g < 4; ++g)
          for (int sign = -1; sign <= 1; sign += 2)
            fn((a + b) / 2 + sign * (b - a) * x[g] / 2, (b - a) * wt[g] / 2);
      }
    }
  }
  UGKWP_GAS_HD Q baseAverage(const Segment &s) const {
    Q q{};
    quadrature(s, [&](double y, double weight) {
      auto v = conserved(base(y));
      for (int j = 0; j < 9; ++j)
        if (j != 6 || !s.algebraicOmega)
          q[j] += weight * v[j] / (s.hi - s.lo);
    });
    return q;
  }
  UGKWP_GAS_HD Q average(const Segment &s) const {
    Q q{};
    quadrature(s, [&](double y, double weight) {
      auto v = conserved(pointIn(y, s));
      for (int j = 0; j < 9; ++j)
        if (j != 6 || !s.algebraicOmega)
          q[j] += weight * v[j] / (s.hi - s.lo);
    });
    return q;
  }
  template <bool BaseOnly, class Team>
  UGKWP_GAS_HD Q orderedAverageTeam(const Segment &s, ProfileScratch &scratch,
                                    const Team &team) const {
    if (team.size() == 1) {
      if constexpr (BaseOnly)
        return baseAverage(s);
      else
        return average(s);
    }
    int count = 0;
    quadrature(s, [&](double y, double weight) {
      const int index = count++;
      if (index % team.size() != team.rank())
        return;
      Q v;
      if constexpr (BaseOnly)
        v = conserved(base(y));
      else
        v = conserved(pointIn(y, s));
      for (int j = 0; j < 9; ++j)
        if (j != 6 || !s.algebraicOmega)
          scratch.orderedTerms[index][j] = weight * v[j] / (s.hi - s.lo);
    });
    team.sync();
    Q q{};
    if (team.leader())
      for (int index = 0; index < count; ++index)
        for (int j = 0; j < 9; ++j)
          if (j != 6 || !s.algebraicOmega)
            q[j] += scratch.orderedTerms[index][j];
    // Publish the reads before another segment reuses the worker buffer.
    team.sync();
    return q;
  }
  template <class Team>
  UGKWP_GAS_HD Q averageTeam(const Segment &s, ProfileScratch &scratch,
                             const Team &team) const {
    return orderedAverageTeam<false>(s, scratch, team);
  }
  template <class Team>
  UGKWP_GAS_HD Q baseAverageTeam(const Segment &s, ProfileScratch &scratch,
                                 const Team &team) const {
    return orderedAverageTeam<true>(s, scratch, team);
  }
  template <class Getter>
  UGKWP_GAS_HD void monotonic(double &coef, double target, Getter value) {
    double a = -1, b = 1;
    coef = a;
    double va = value();
    coef = b;
    double vb = value();
    for (int i = 0; i < 16 && va > target; ++i) {
      a *= 2;
      coef = a;
      va = value();
    }
    for (int i = 0; i < 16 && vb < target; ++i) {
      b *= 2;
      coef = b;
      vb = value();
    }
    for (int i = 0; i < 70; ++i) {
      coef = (a + b) / 2;
      if (coef == a || coef == b)
        return;
      double v = value();
      if (v < target)
        a = coef;
      else
        b = coef;
    }
    coef = (a + b) / 2;
  }
  UGKWP_GAS_HD void fit(ProfileScratch &scratch) {
    maxResidual = 0;
    for (auto &s : segments) {
      BorrowedVector<Sample> samples{scratch.samples, 0};
      Segment plain = s;
      plain.aRho = plain.aT = plain.aK = plain.aOmega = plain.aY = 0;
      for (auto &x : plain.aU)
        x = 0;
      quadrature(s, [&](double y, double weight) {
        Sample x{};
        x.p = pointIn(y, plain);
        x.weight = weight / (s.hi - s.lo);
        x.b = bubble((y - s.lo) / (s.hi - s.lo));
        x.by = speciesBubble((y - s.lo) / (s.hi - s.lo), s.lo == 0);
        samples.push_back(x);
      });
      double maxBaseK = 0;
      for (auto &x : samples)
        maxBaseK = std::max(maxBaseK, x.p.k);
      if (maxBaseK == 0 && s.target[5] > 0) {
        s.seedK = s.target[5] / s.target[0];
        for (auto &x : samples)
          x.p.k = s.seedK * x.by;
      }
      monotonic(s.aRho, s.target[0], [&]() {
        double sum = 0;
        for (auto &x : samples)
          sum += x.weight * x.p.rho * std::exp(s.aRho * x.b);
        return sum;
      });
      for (auto &x : samples)
        x.rho = x.p.rho * std::exp(s.aRho * x.b);

      if (s.target[7] > 0 && s.target[8] > 0)
        monotonic(s.aY, s.target[7], [&]() {
          double sum = 0;
          for (auto &x : samples)
            sum += x.weight * x.rho * fraction(x.p.Y[0], s.aY, x.by);
          return sum;
        });
      double massBubble = 0, baseMomentum[3]{};
      for (auto &x : samples) {
        massBubble += x.weight * x.rho * x.b;
        for (int j = 0; j < 3; ++j)
          baseMomentum[j] += x.weight * x.rho * x.p.U[j];
        x.Y[0] = fraction(x.p.Y[0], s.aY, x.by);
        x.Y[1] = 1 - x.Y[0];
      }
      for (int j = 0; j < 3; ++j)
        s.aU[j] = (s.target[j + 1] - baseMomentum[j]) / massBubble;
      for (auto &x : samples) {
        x.kinetic = 0;
        for (int j = 0; j < 3; ++j) {
          x.U[j] = x.p.U[j] + s.aU[j] * x.b;
          x.kinetic += x.U[j] * x.U[j] / 2;
        }
      }
      monotonic(s.aT, s.target[4], [&]() {
        double sum = 0;
        for (auto &x : samples)
          sum += x.weight * x.rho *
                 (ugkwp::mixtureEnergy(x.Y, x.p.T * std::exp(s.aT * x.b),
                                       in.model.thermo) +
                  x.kinetic);
        return sum;
      });
      monotonic(s.aK, s.target[5], [&]() {
        double sum = 0;
        for (auto &x : samples)
          sum += x.weight * x.rho * x.p.k * std::exp(s.aK * x.b);
        return sum;
      });
      if (!s.algebraicOmega)
        monotonic(s.aOmega, s.target[6], [&]() {
          double sum = 0;
          for (auto &x : samples)
            sum += x.weight * x.rho * x.p.omega * std::exp(s.aOmega * x.b);
          return sum;
        });
      auto q = average(s);
      for (int j = 0; j < 9; ++j)
        if (j != 6 || !s.algebraicOmega)
          maxResidual =
              std::max(maxResidual, std::abs(q[j] - s.target[j]) /
                                        std::max(1., std::abs(s.target[j])));
    }
  }

  // Cooperative fitting uses the same quadrature and scalar brackets as the
  // delivered CPU implementation. Each lane owns disjoint cached samples.
  template <class Team>
  UGKWP_GAS_HD bool fitLegacyTeam(ProfileScratch &scratch, const Team &team) {
    maxResidual = 0;
    for (auto &s : segments) {
      Segment plain = s;
      plain.aRho = plain.aT = plain.aK = plain.aOmega = plain.aY = 0;
      for (auto &x : plain.aU)
        x = 0;
      int count = 0;
      double localMax = 0, incompatible = 0;
      quadrature(s, [&](double y, double weight) {
        int index = count++;
        if (index % team.size() != team.rank())
          return;
        Sample x{};
        x.p = pointIn(y, plain);
        x.weight = weight / (s.hi - s.lo);
        x.b = bubble((y - s.lo) / (s.hi - s.lo));
        x.by = speciesBubble((y - s.lo) / (s.hi - s.lo), s.lo == 0);
        scratch.samples[index] = x;
        localMax = d::max(localMax, x.p.k);
        // Positive density and nonnegative Y cannot integrate to zero
        // when the unfitted continuous profile contains this species.
        for (int j = 0; j < 2; ++j)
          if (s.target[7 + j] == 0 && x.p.Y[j] > 0)
            incompatible = 1;
      });
      team.sync();
      if (team.maximum(incompatible) != 0)
        return false;
      if (team.maximum(localMax) == 0 && s.target[5] > 0) {
        s.seedK = s.target[5] / s.target[0];
        for (int i = team.rank(); i < count; i += team.size())
          scratch.samples[i].p.k = s.seedK * scratch.samples[i].by;
      }
      auto sum = [&](auto term) {
        double v = 0;
        for (int i = team.rank(); i < count; i += team.size())
          v += term(scratch.samples[i]);
        return team.sum(v);
      };
      monotonic(s.aRho, s.target[0], [&]() {
        return sum([&](const Sample &x) {
          return x.weight * x.p.rho * ::exp(s.aRho * x.b);
        });
      });
      for (int i = team.rank(); i < count; i += team.size()) {
        auto &x = scratch.samples[i];
        x.rho = x.p.rho * ::exp(s.aRho * x.b);
      }
      team.sync();
      if (s.target[7] > 0 && s.target[8] > 0)
        monotonic(s.aY, s.target[7], [&]() {
          return sum([&](const Sample &x) {
            return x.weight * x.rho * fraction(x.p.Y[0], s.aY, x.by);
          });
        });
      double massBubble =
          sum([&](const Sample &x) { return x.weight * x.rho * x.b; });
      for (int j = 0; j < 3; ++j) {
        double momentum =
            sum([&](const Sample &x) { return x.weight * x.rho * x.p.U[j]; });
        s.aU[j] = (s.target[j + 1] - momentum) / massBubble;
      }
      for (int i = team.rank(); i < count; i += team.size()) {
        auto &x = scratch.samples[i];
        x.Y[0] = fraction(x.p.Y[0], s.aY, x.by);
        x.Y[1] = 1 - x.Y[0];
        x.kinetic = 0;
        for (int j = 0; j < 3; ++j) {
          x.U[j] = x.p.U[j] + s.aU[j] * x.b;
          x.kinetic += x.U[j] * x.U[j] / 2;
        }
      }
      team.sync();
      monotonic(s.aT, s.target[4], [&]() {
        return sum([&](const Sample &x) {
          return x.weight * x.rho *
                 (ugkwp::mixtureEnergy(x.Y, x.p.T * ::exp(s.aT * x.b),
                                       in.model.thermo) +
                  x.kinetic);
        });
      });
      monotonic(s.aK, s.target[5], [&]() {
        return sum([&](const Sample &x) {
          return x.weight * x.rho * x.p.k * ::exp(s.aK * x.b);
        });
      });
      if (!s.algebraicOmega)
        monotonic(s.aOmega, s.target[6], [&]() {
          return sum([&](const Sample &x) {
            return x.weight * x.rho * x.p.omega * ::exp(s.aOmega * x.b);
          });
        });
      const auto q = averageTeam(s, scratch, team);
      if (team.leader()) {
        for (int j = 0; j < 9; ++j)
          if (j != 6 || !s.algebraicOmega)
            maxResidual =
                d::max(maxResidual, ::fabs(q[j] - s.target[j]) /
                                        d::max(1., ::fabs(s.target[j])));
      }
      maxResidual = team.broadcast(maxResidual);
    }
    return true;
  }
  template<class Team>
  UGKWP_GAS_HD bool fitTeam(ProfileScratch& scratch,const Team& team) {
    affineHelpersOk=affineRawFinite=true;affineSpeciesMode=0;affineSpeciesA=affineSpeciesB=0;affineMomentEvaluations=affineIterations=affineSamples=0;
    int requested=0;
    if(!in.model.molecularViscosity.diffusion&&cfg.coherentDiscretization&&segments.size()==2&&in.model.mode==ugkwp::GasMode::MixtureFrozen&&in.massFlux[0]==0&&in.massFlux[1]==0&&in.model.diffusivity[0]>0&&in.model.diffusivity[0]==in.model.diffusivity[1]) {
      bool interior=in.matching.massFraction[0]>0&&in.matching.massFraction[1]>0;
      for(const auto& seg:segments)interior=interior&&seg.target[7]>0&&seg.target[8]>0;
      if(interior)requested=1;
      if(in.matching.massFraction[0]==0&&in.matching.massFraction[1]==1&&segments[0].target[7]==0&&segments[1].target[7]==0)requested=2;
      if(in.matching.massFraction[0]==1&&in.matching.massFraction[1]==0&&segments[0].target[8]==0&&segments[1].target[8]==0)requested=3;
    }
    if(!requested)return fitLegacyTeam(scratch,team); // Existing general computation; not a new rejection.
    maxResidual=0;int offsets[3]{},cell=0;
    for(auto &seg:segments) {
      seg.aY=seg.aT=0;Segment plain=seg;plain.aRho=plain.aT=plain.aK=plain.aOmega=plain.aY=0;for(auto &u:plain.aU)u=0;
      int count=0,begin=offsets[cell];double localMax=0;
      quadrature(seg,[&](double y,double weight){int index=count++;if(index%team.size()!=team.rank())return;Sample x{};x.p=pointIn(y,plain);x.y=y;x.weight=weight/(seg.hi-seg.lo);double t=(y-seg.lo)/(seg.hi-seg.lo);x.b=bubble(t);x.by=speciesBubble(t,seg.lo==0);if(requested==1){auto v=base(y);x.speciesBaseLogit=::log(v.Y[0]/v.Y[1]);}scratch.samples[begin+index]=x;localMax=d::max(localMax,x.p.k);});team.sync();
      if(team.maximum(localMax)==0&&seg.target[5]>0){seg.seedK=seg.target[5]/seg.target[0];for(int i=team.rank();i<count;i+=team.size())scratch.samples[begin+i].p.k=seg.seedK*scratch.samples[begin+i].by;}
      auto sum=[&](auto term){double value=0;for(int i=team.rank();i<count;i+=team.size())value+=term(scratch.samples[begin+i]);return team.sum(value);};
      monotonic(seg.aRho,seg.target[0],[&](){return sum([&](const Sample&x){return x.weight*x.p.rho*::exp(seg.aRho*x.b);});});
      for(int i=team.rank();i<count;i+=team.size()){auto&x=scratch.samples[begin+i];x.rho=x.p.rho*::exp(seg.aRho*x.b);}team.sync();
      double massBubble=sum([&](const Sample&x){return x.weight*x.rho*x.b;});
      for(int j=0;j<3;++j){double momentum=sum([&](const Sample&x){return x.weight*x.rho*x.p.U[j];});seg.aU[j]=(seg.target[j+1]-momentum)/massBubble;}
      for(int i=team.rank();i<count;i+=team.size()){auto&x=scratch.samples[begin+i];x.kinetic=0;for(int j=0;j<3;++j){x.U[j]=x.p.U[j]+seg.aU[j]*x.b;x.kinetic+=x.U[j]*x.U[j]/2;}}team.sync();
      monotonic(seg.aK,seg.target[5],[&](){return sum([&](const Sample&x){return x.weight*x.rho*x.p.k*::exp(seg.aK*x.b);});});
      if(!seg.algebraicOmega)monotonic(seg.aOmega,seg.target[6],[&](){return sum([&](const Sample&x){return x.weight*x.rho*x.p.omega*::exp(seg.aOmega*x.b);});});
      offsets[++cell]=begin+count;
    }
    affineSamples=offsets[2];
    if(requested==1) {
      if(team.leader()){double y=segments[0].hi,h=segments[0].hi-segments[0].lo,g=segments[1].hi-segments[1].lo,A[2]{};for(int s=0;s<2;++s){auto v=physicalPrimitive(y,s?1:-1);physical::Face<double,2>f;physical::Source<double,2>src;affineHelpersOk=physical::evaluatePrimitive(in,cfg,v,y,in.normal,0.,f,src)&&affineHelpersOk;A[s]=v.rho*(in.model.diffusivity[0]+src.nut/cfg.turbulentSchmidt);}affineRawFinite=affineRawFinite&&std::isfinite(A[0])&&std::isfinite(A[1]);affineMobilityLeft=A[0];affineMobilityRight=A[1];affineInterface=h*A[1]/(h*A[1]+g*A[0]);}affineInterface=team.broadcast(affineInterface);affineMobilityLeft=team.broadcast(affineMobilityLeft);affineMobilityRight=team.broadcast(affineMobilityRight);affineHelpersOk=team.broadcast(affineHelpersOk?1.:0.)!=0;affineRawFinite=team.broadcast(affineRawFinite?1.:0.)!=0;
      for(int c=0;c<2;++c){int count=offsets[c+1]-offsets[c];for(int i=team.rank();i<count;i+=team.size()){auto&x=scratch.samples[offsets[c]+i];double r=affineCoordinate(x.y,c?1:-1)[0];x.speciesBasis[0]=1-r*r;x.speciesBasis[1]=r*r*(1-r);}}team.sync();
      struct SpeciesEvaluation{double f[2]{},j[2][2]{},norm=0;};
      auto evaluate=[&](double a,double b){++affineMomentEvaluations;SpeciesEvaluation e;for(int c=0;c<2;++c){double value=0,j0=0,j1=0;int count=offsets[c+1]-offsets[c];for(int i=team.rank();i<count;i+=team.size()){const auto&x=scratch.samples[offsets[c]+i];double Y=speciesLogistic(x.speciesBaseLogit+a*x.speciesBasis[0]+b*x.speciesBasis[1]),w=x.weight*x.rho;value+=w*Y;double dY=w*Y*(1-Y);j0+=dY*x.speciesBasis[0];j1+=dY*x.speciesBasis[1];}e.f[c]=team.sum(value)-segments[c].target[7];e.j[c][0]=team.sum(j0);e.j[c][1]=team.sum(j1);affineRawFinite=affineRawFinite&&std::isfinite(e.f[c])&&std::isfinite(e.j[c][0])&&std::isfinite(e.j[c][1]);e.norm=d::max(e.norm,::fabs(e.f[c])/d::max(1.,::fabs(segments[c].target[7])));}return e;};
      auto e=evaluate(0,0);for(int k=0;k<cfg.maxIterations;++k){double det=e.j[0][0]*e.j[1][1]-e.j[0][1]*e.j[1][0],da=(-e.f[0]*e.j[1][1]+e.f[1]*e.j[0][1])/det,db=(-e.j[0][0]*e.f[1]+e.j[1][0]*e.f[0])/det,scale=1;bool improved=false;for(int line=0;line<40;++line){double a=affineSpeciesA+scale*da,b=affineSpeciesB+scale*db;if(a==affineSpeciesA&&b==affineSpeciesB)break;auto next=evaluate(a,b);if(next.norm<e.norm){affineSpeciesA=a;affineSpeciesB=b;e=next;improved=true;break;}scale*=.5;}++affineIterations;if(!improved)break;}
    }
    // Last numerical fit remains publishable: observations do not add a gate.
    affineSpeciesMode=requested;
    for(int c=0;c<2;++c){auto&seg=segments[c];int begin=offsets[c],count=offsets[c+1]-begin;for(int i=team.rank();i<count;i+=team.size()){auto&x=scratch.samples[begin+i];x.Y[0]=requested==2?0:requested==3?1:speciesLogistic(x.speciesBaseLogit+affineSpeciesA*x.speciesBasis[0]+affineSpeciesB*x.speciesBasis[1]);x.Y[1]=1-x.Y[0];}team.sync();monotonic(seg.aT,seg.target[4],[&](){double value=0;for(int i=team.rank();i<count;i+=team.size()){const auto&x=scratch.samples[begin+i];value+=x.weight*x.rho*(ugkwp::mixtureEnergy(x.Y,x.p.T*::exp(seg.aT*x.b),in.model.thermo)+x.kinetic);}return team.sum(value);});}
    for(const auto&seg:segments){auto q=averageTeam(seg,scratch,team);if(team.leader())for(int j=0;j<9;++j)if(j!=6||!seg.algebraicOmega)maxResidual=d::max(maxResidual,::fabs(q[j]-seg.target[j])/d::max(1.,::fabs(seg.target[j])));}maxResidual=team.broadcast(maxResidual);return true;
  }
  template <class Team>
  UGKWP_GAS_HD Array<double, 2> sourceTeam(int cell, const Team &team,
                                           int subdivisions = 4) const {
    double k = 0, omega = 0;
    int index = 0;
    const auto &s = segments[cell];
    quadrature(
        s,
        [&](double y, double weight) {
          if (index++ % team.size() != team.rank())
            return;
          auto p = physicalPrimitive(y);
          physical::Face<double, 2> f;
          physical::Source<double, 2> src;
          physical::evaluatePrimitive(in, cfg, p, y, in.normal, 0., f, src);
          k += weight * src.k;
          if (!s.algebraicOmega)
            omega += weight * src.omega;
        },
        subdivisions);
    return {team.sum(k), team.sum(omega)};
  }

  // One-sided micro derivatives, recovered symmetrically at shared macro faces.
  UGKWP_GAS_HD Primitive derivativeRaw(double y, const Segment &s,
                                        const Primitive *basePoint = nullptr) const {
    if (y == 0 && s.lo == 0) {
      auto p = basePoint ? *basePoint : base(0);
      Primitive g{};
      double micro = w.y[1] - w.y[0], db = 6 / (s.hi - s.lo),
             tbase =
                 (w.state[1][2] - w.state[0][2]) / micro + wallSlopeDelta[2],
             drgas = 0;
      for (int j = 0; j < 2; ++j) {
        double yr = (base(micro).Y[j] - p.Y[j]) / micro +
                    (j == 0 ? 1 : -1) * wallSpeciesSlopeDelta0;
        g.Y[j] = yr;
        drgas += yr * ugkwp::universalGasConstant<double>() /
                 in.model.thermo.species[j].molarMass;
      }
      double gasR = ugkwp::mixtureGasConstant(p.Y, in.model.thermo),
             rbase = -p.rho * (tbase / p.T + drgas / gasR),
             dv = -ctx.mass * rbase / (p.rho * p.rho);
      if (std::isfinite(s.normalLo) && std::isfinite(s.normalHi))
        dv = (s.normalHi - s.normalLo) / (s.hi - s.lo);
      g.T = tbase + p.T * s.aT * db;
      g.rho = rbase + p.rho * s.aRho * db;
      for (int j = 0; j < 3; ++j) {
        g.U[j] = dv * in.normal[j] + s.aU[j] * db;
        for (int a = 0; a < 2; ++a)
          g.U[j] +=
              ((w.state[1][a] - w.state[0][a]) / micro + wallSlopeDelta[a]) *
              ctx.tangent[a][j];
      }
      double slope = std::sqrt(p.rho * in.model.sstCoefficients.beta1 /
                               (6 * wallMolecularViscosity(in,p.T,p.Y))),
             ell = w.state[0][5] / slope,
             power = (1 + std::sqrt(1 + 24 * in.model.sstCoefficients.betaStar /
                                            in.model.sstCoefficients.beta1)) /
                     2;
      g.k = cfg.enableSst ? w.state[1][4] * w.state[1][4] *
                                d::nearWallKDerivativeAtWall(
                                    micro, ell, power,
                                    ctx.mass / (2 * wallMolecularViscosity(in,p.T,p.Y)))
                          : 0;
      g.omega = 0;
      if (s.endpointOverride) {
        double h = s.hi - s.lo;
        g.rho += p.rho * logRatio(s.right.rho, s.baseRight.rho) / h;
        g.T += p.T * logRatio(s.right.T, s.baseRight.T) / h;
        double normalDelta = 0;
        if (std::isfinite(s.normalLo) && std::isfinite(s.normalHi))
          for (int j = 0; j < 3; ++j)
            normalDelta += (s.right.U[j] - s.baseRight.U[j]) * in.normal[j];
        for (int j = 0; j < 3; ++j)
          g.U[j] +=
              (s.right.U[j] - s.baseRight.U[j] - normalDelta * in.normal[j]) /
              h;
      }
      return g;
    }
    {
      int l = 0;
      while (l + 1 < cfg.nodes - 1 && w.y[l + 1] < y)
        ++l;
      double h = w.y[l + 1] - w.y[l], t = (y - w.y[l]) / h,
             x = (y - s.lo) / (s.hi - s.lo), b = bubble(x),
             db = 6 * (1 - 2 * x) / (s.hi - s.lo),
             by = speciesBubble(x, s.lo == 0),
             dby = s.lo == 0 ? 12 * (2 * x - 3 * x * x) / (s.hi - s.lo) : db;
      double state[6], ds[6];
      for (int j = 0; j < 6; ++j) {
        state[j] = w.state[l][j] + t * (w.state[l + 1][j] - w.state[l][j]);
        ds[j] = (w.state[l + 1][j] - w.state[l][j]) / h;
      }
      if (l > 0)
        for (int j = 0; j < 6; ++j)
          if (hermiteField(j)) {
            auto v = hermite(l, j, t);
            state[j] = v[0];
            ds[j] = v[1];
          }
      if (l == 0) {
        double z = y / w.y[1], shape = y * (1 - z) * (1 - z),
               dshape = (1 - z) * (1 - 3 * z);
        for (int j = 0; j < 3; ++j) {
          state[j] += wallSlopeDelta[j] *
                    (cfg.coherentDiscretization && j == 2
                         ? y * (1 - y / w.y[1]) : shape);
          ds[j] += wallSlopeDelta[j] * (cfg.coherentDiscretization && j == 2 ? 1 - 2 * z : dshape);
        }
        double sign = ctx.independent[0] == 0 ? 1 : -1;
        state[3] += sign * wallSpeciesSlopeDelta0 * shape;
        ds[3] += sign * wallSpeciesSlopeDelta0 * dshape;
      }
      auto p = basePoint ? *basePoint : base(y);
      Primitive g{};
      double dy0 = (ctx.independent[0] == 0 ? 1 : -1) * ds[3],
             R = ugkwp::mixtureGasConstant(p.Y, in.model.thermo),
             dR = dy0 * ugkwp::universalGasConstant<double>() *
                  (1 / in.model.thermo.species[0].molarMass -
                   1 / in.model.thermo.species[1].molarMass),
             drho = -p.rho * (ds[2] / p.T + dR / R),
             dv = -ctx.mass * drho / (p.rho * p.rho);
      if (std::isfinite(s.normalLo) && std::isfinite(s.normalHi))
        dv = (s.normalHi - s.normalLo) / (s.hi - s.lo);
      for (int j = 0; j < 3; ++j) {
        g.U[j] = dv * in.normal[j] + s.aU[j] * db;
        for (int a = 0; a < 2; ++a)
          g.U[j] += ds[a] * ctx.tangent[a][j];
      }
      g.rho = std::exp(s.aRho * b) * (drho + p.rho * s.aRho * db);
      g.T = std::exp(s.aT * b) * (ds[2] + p.T * s.aT * db);
      double dk = 2 * state[4] * ds[4];
      if (cfg.enableSst && l == 0) {
        auto wall = base(0);
        double slope = std::sqrt(wall.rho * in.model.sstCoefficients.beta1 /
                                 (6 * wallMolecularViscosity(in,wall.T,wall.Y))),
               ell = w.state[0][5] / slope,
               power =
                   (1 + std::sqrt(1 + 24 * in.model.sstCoefficients.betaStar /
                                          in.model.sstCoefficients.beta1)) /
                   2;
        dk = w.state[1][4] * w.state[1][4] *
             d::nearWallKDerivative(y, w.y[1], ell, power,
                                    ctx.mass / (2 * wallMolecularViscosity(in,wall.T,wall.Y)));
      }
      g.k = std::exp(s.aK * b) *
            (dk + s.seedK * dby + (p.k + s.seedK * by) * s.aK * db);
      double domega = cfg.enableSst && state[5] > 0
                          ? -2 * ds[5] / (state[5] * state[5] * state[5])
                          : 0;
      g.omega = s.algebraicOmega ? domega
                                 : std::exp(s.aOmega * b) *
                                       (domega + p.omega * s.aOmega * db);
      if (s.aY >= 0) {
        double den = 1 + s.aY * by;
        g.Y[0] = (dy0 * den + s.aY * dby * (1 - p.Y[0])) / (den * den);
      } else {
        double den = 1 - s.aY * by;
        g.Y[0] = dy0 / den + p.Y[0] * s.aY * dby / (den * den);
      }
      g.Y[1] = -g.Y[0];
      return g;
    }
  }
  UGKWP_GAS_HD Primitive derivative(double y, int side = 0,
                                     const Primitive *basePoint = nullptr) const {
    const auto &s = segments[segment(y, side)];
    if (y == 0 || !s.endpointOverride) {
      auto g=derivativeRaw(y,s,basePoint);if(affineSpeciesMode){double Y=(basePoint ? basePoint->Y[0] : base(y).Y[0]);g.Y[0]=affineGradient(y,Y,g.Y[0],side);g.Y[1]=-g.Y[0];}return g;
    }
    // Evaluate the base derivative directly: no recursive Profile copy on GPU.
    Segment plain = s;
    plain.endpointOverride = false;
    plain.seedK = 0;
    plain.aRho = plain.aT = plain.aK = plain.aOmega = plain.aY = 0;
    for (int j = 0; j < 3; ++j)
      plain.aU[j] = 0;
    auto g0 = derivativeRaw(y, plain,basePoint), p0 = basePoint ? *basePoint : base(y);
    Primitive g{};
    double h = s.hi - s.lo, x = (y - s.lo) / h, b = bubble(x),
           db = 6 * (1 - 2 * x) / h;
    auto chain = [&](double value, double slope, double left, double right,
                     double bl, double br, double amp) {
      double ll = logRatio(left, bl), lr = logRatio(right, br),
             e = std::exp((1 - x) * ll + x * lr + amp * b);
      return e * (slope + value * ((lr - ll) / h + amp * db));
    };
    g.rho = chain(p0.rho, g0.rho, s.left.rho, s.right.rho, s.baseLeft.rho,
                  s.baseRight.rho, s.aRho);
    g.T = chain(p0.T, g0.T, s.left.T, s.right.T, s.baseLeft.T, s.baseRight.T,
                s.aT);
    g.k = chain(p0.k, g0.k, s.left.k, s.right.k, s.baseLeft.k, s.baseRight.k,
                s.aK);
    double ol = logRatio(s.left.omega, s.baseLeft.omega),
           orr = logRatio(s.right.omega, s.baseRight.omega);
    if (s.lo == 0 && s.algebraicOmega)
      ol = std::log(s.baseLeft.rho / s.left.rho);
    double aw = s.algebraicOmega ? 0 : s.aOmega;
    g.omega = std::exp((1 - x) * ol + x * orr + aw * b) *
              (g0.omega + p0.omega * ((orr - ol) / h + aw * db));
    double du[3], dn = 0;
    for (int j = 0; j < 3; ++j) {
      du[j] = ((s.right.U[j] - s.baseRight.U[j]) -
               (s.left.U[j] - s.baseLeft.U[j])) /
              h;
      dn += du[j] * in.normal[j];
    }
    for (int j = 0; j < 3; ++j)
      g.U[j] = g0.U[j] + du[j] -
               (std::isfinite(s.normalLo) && std::isfinite(s.normalHi)
                    ? dn * in.normal[j]
                    : 0) +
               s.aU[j] * db;
    double by = speciesBubble(x, s.lo == 0),
           dby = s.lo == 0 ? 12 * (2 * x - 3 * x * x) / h : db,
           t = s.lo == 0 ? x * x * (3 - 2 * x) : x,
           dt = s.lo == 0 ? 6 * x * (1 - x) / h : 1 / h;
    if (s.baseLeft.k == 0 && s.baseRight.k == 0) {
      double kl = (1 - t) * s.left.k + t * s.right.k;
      g.k = std::exp(s.aK * b) * ((s.right.k - s.left.k) * dt + s.seedK * dby +
                                  (kl + s.seedK * by) * s.aK * db);
    } else
      g.k += std::exp(s.aK * b) * s.seedK * (dby + by * s.aK * db);
    double y0 = p0.Y[0], dy0 = g0.Y[0];
    if (s.left.Y[0] != s.baseLeft.Y[0] || s.right.Y[0] != s.baseRight.Y[0]) {
      auto mapped = [&](double target, double reference) {
        double A = 1, B = 0;
        if (target >= reference) {
          if (reference < 1) {
            B = (target - reference) / (1 - reference);
            A = 1 - B;
          }
        } else if (reference > 0)
          A = target / reference;
        return Array<double, 2>{A * y0 + B, A * dy0};
      };
      auto l = mapped(s.left.Y[0], s.baseLeft.Y[0]),
           r = mapped(s.right.Y[0], s.baseRight.Y[0]);
      y0 = (1 - t) * l[0] + t * r[0];
      dy0 = (1 - t) * l[1] + t * r[1] + dt * (r[0] - l[0]);
    }
    if (s.aY >= 0) {
      double den = 1 + s.aY * by;
      g.Y[0] = (dy0 * den + s.aY * dby * (1 - y0)) / (den * den);
    } else {
      double den = 1 - s.aY * by;
      g.Y[0] = dy0 / den + y0 * s.aY * dby / (den * den);
    }
    g.Y[1] = -g.Y[0];
    if(affineSpeciesMode){g.Y[0]=affineGradient(y,p0.Y[0],g0.Y[0],side);g.Y[1]=-g.Y[0];}
    return g;
  }
  UGKWP_GAS_HD d::LayerPoint<double, 2> physics(double y, const Primitive &p,
                                                const Primitive &g) const {
    d::LayerPoint<double, 2> v;
    v.rho = p.rho;
    v.T = p.T;
    v.theta = p.T - in.temperature;
    v.k = p.k;
    v.omega = p.omega;
    v.z = p.omega > 0 ? 1 / std::sqrt(p.omega) : 0;
    for (int j = 0; j < 2; ++j) {
      v.Y[j] = p.Y[j];
      double cp;
      ugkwp::speciesCpH(j, p.T, in.model.thermo, cp, v.h[j]);
      v.cp += p.Y[j] * cp;
    }
    if (y > 0) {
      double du[2]{}, dv = 0;
      for (int j = 0; j < 3; ++j) {
        dv += g.U[j] * in.normal[j];
        for (int a = 0; a < 2; ++a)
          du[a] += g.U[j] * ctx.tangent[a][j];
      }
      d::applySstGradients(in, y, du[0], du[1], dv, g.k, g.omega, v);
    }
    return v;
  }
  UGKWP_GAS_HD void setMacroGradients(int i, const MacroGradients &g) {
    segments[i].macroGradients = g;
    segments[i].hasMacroGradients = true;
  }
  UGKWP_GAS_HD physical::Primitive<double, 2>
  physicalPrimitive(double y, int side = 0) const {
    const auto &s = segments[segment(y, side)];
    // These consumers observe the same immutable query state. Reuse the exact
    // base value instead of repeating its interpolation and thermochemistry.
    const auto basePoint = base(y);
    auto p = pointIn(y, s, &basePoint), g = derivative(y, side, &basePoint);
    physical::Primitive<double, 2> v;
    v.rho = p.rho;
    v.T = p.T;
    v.k = p.k;
    v.omega = p.omega;
    for (int j = 0; j < 2; ++j)
      v.Y[j] = p.Y[j];
    auto tangent = [&](const double *gradient, int direction) {
      double dot = 0;
      for (int j = 0; j < 3; ++j)
        dot += gradient[j] * in.normal[j];
      return gradient[direction] - dot * in.normal[direction];
    };
    for (int a = 0; a < 3; ++a) {
      v.U[a] = p.U[a];
      v.gradT[a] = g.T * in.normal[a];
      v.gradK[a] = g.k * in.normal[a];
      v.gradOmega[a] = g.omega * in.normal[a];
      for (int j = 0; j < 2; ++j)
        v.gradY[j][a] = g.Y[j] * in.normal[a];
      if (s.hasMacroGradients) {
        v.gradT[a] += tangent(s.macroGradients.T, a);
        v.gradK[a] += tangent(s.macroGradients.k, a);
        v.gradOmega[a] += tangent(s.macroGradients.omega, a);
        for (int j = 0; j < 2; ++j)
          v.gradY[j][a] += tangent(s.macroGradients.Y[j], a);
      }
      for (int b = 0; b < 3; ++b) {
        v.gradU[a][b] = g.U[a] * in.normal[b];
        if (s.hasMacroGradients)
          v.gradU[a][b] += tangent(s.macroGradients.U[a], b);
        else
          for (int lc = 0; lc < 3; ++lc)
            for (int ld = 0; ld < 2; ++ld)
              v.gradU[a][b] += (lc == 2 ? in.normal[a] : ctx.tangent[lc][a]) *
                               in.sstTangentialGradient[lc][ld] *
                               ctx.tangent[ld][b];
      }
    }
    return v;
  }
  UGKWP_GAS_HD Q inviscid(const Primitive &p, double meshNormal = 0) const {
    auto q = conserved(p);
    double un = 0;
    for (int j = 0; j < 3; ++j)
      un += p.U[j] * in.normal[j];
    double pressure =
               p.rho * ugkwp::mixtureGasConstant(p.Y, in.model.thermo) * p.T,
           mass = p.rho * (un - meshNormal);
    Q f{};
    f[0] = mass;
    for (int j = 0; j < 3; ++j)
      f[1 + j] = mass * p.U[j] + pressure * in.normal[j];
    f[4] = (un - meshNormal) * q[4] + pressure * un;
    f[5] = mass == 0 ? 0 : mass * p.k;
    f[6] = mass == 0 ? 0 : mass * p.omega;
    f[7] = mass * p.Y[0];
    f[8] = mass * p.Y[1];
    return f;
  }
  UGKWP_GAS_HD Q inviscidFlux(double y, double meshNormal = 0) const {
    return inviscid(point(y), meshNormal);
  }
  UGKWP_GAS_HD Q diffusiveFlux(double y, double meshNormal = 0) const {
    auto all = flux(y, meshNormal), adv = inviscidFlux(y, meshNormal);
    for (int j = 0; j < 9; ++j)
      all[j] -= adv[j];
    return all;
  }
  UGKWP_GAS_HD Q flux(double y, double meshNormal = 0) const {
    auto p = physicalPrimitive(y);
    bool affineShared=false;double commonJ=0;
    for (int i = 0; i + 1 < int(segments.size()); ++i)
      if (y == segments[i].hi) {
        auto q = physicalPrimitive(y, 1);
        if(affineSpeciesMode){affineShared=true;double gl=0,gr=0;for(int a=0;a<3;++a){gl+=p.gradY[0][a]*in.normal[a];gr+=q.gradY[0][a]*in.normal[a];}commonJ=affineSpeciesMode==1?-(affineMobilityLeft*gl+affineMobilityRight*gr)/2:0;}
        for (int a = 0; a < 3; ++a) {
          p.gradT[a] = (p.gradT[a] + q.gradT[a]) / 2;
          for (int j = 0; j < 2; ++j)
            p.gradY[j][a] = (p.gradY[j][a] + q.gradY[j][a]) / 2;
          p.gradK[a] = (p.gradK[a] + q.gradK[a]) / 2;
          p.gradOmega[a] = (p.gradOmega[a] + q.gradOmega[a]) / 2;
          for (int b = 0; b < 3; ++b)
            p.gradU[a][b] = (p.gradU[a][b] + q.gradU[a][b]) / 2;
        }
      }
    physical::Face<double, 2> f;
    physical::Source<double, 2> src;
    physical::evaluatePrimitive(in, cfg, p, y, in.normal, meshNormal, f, src);
    if(affineShared){
      // Preserve the common constitutive species flux, not an averaged gradient
      // evaluated with a different nonlinear SST mobility. Keep all unrelated
      // averaged momentum/conduction terms and correct the same enthalpy flux.
      double species[2]={f.mass*p.Y[0]+commonJ,f.mass*p.Y[1]-commonJ};
      for(int j=0;j<2;++j){f.energy+=ugkwp::speciesH(j,p.T,in.model.thermo)*(species[j]-f.species[j]);f.species[j]=species[j];}
    }
    if (y == 0)
      f.omega = 0;
    return {f.mass, f.momentum[0], f.momentum[1], f.momentum[2], f.energy,
            f.k,    f.omega,       f.species[0],  f.species[1]};
  }
  UGKWP_GAS_HD Array<double, 2> source(int cell, int subdivisions = 4) const {
    Array<double, 2> out{};
    auto &s = segments[cell];
    quadrature(
        s,
        [&](double y, double weight) {
          auto p = physicalPrimitive(y);
          physical::Face<double, 2> f;
          physical::Source<double, 2> src;
          physical::evaluatePrimitive(in, cfg, p, y, in.normal, 0., f, src);
          out[0] += weight * src.k;
          if (!s.algebraicOmega)
            out[1] += weight * src.omega;
        },
        subdivisions);
    return out;
  }
};
} // namespace reconstruction

} // namespace gaswall
} // namespace ugkwp
