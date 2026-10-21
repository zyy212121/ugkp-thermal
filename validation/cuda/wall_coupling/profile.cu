// Device-port oracle: the current shared coherent profile, evaluated by
// SerialWallTeam on the host and a full 32-lane team on the GPU. This is not an
// independent mathematical reference or a CPU cooperative speedup experiment.
#ifndef UGKP_PROFILE_CPU_ONLY
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#endif
#include "common/gasWall/WallReferenceSolver.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
namespace dp = ugkwp::gaswall::reconstruction;
using namespace ugkwp::gaswall;
#ifndef UGKP_PROFILE_CPU_ONLY
void ck(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << cudaGetErrorString(e) << '\n';
    exit(2);
  }
}
#endif
template <class T> T *managed(int n = 1) {
#ifdef UGKP_PROFILE_CPU_ONLY
  return new T[n]{};
#else
  T *p;
  ck(cudaMallocManaged(&p, sizeof(T) * n));
  for (int i = 0; i < n; ++i)
    new (p + i) T{};
  return p;
#endif
}
template <class T> void release(T *p) {
#ifdef UGKP_PROFILE_CPU_ONLY
  delete[] p;
#else
  ck(cudaFree(p));
#endif
}
enum Observation {
  AffineMode, WallY, BaseWallY, WallJ, MatchingY, LeftJ, RightJ, CommonJ,
  PublishedJ0, PublishedJ1, SumHJ, PublishedSumHJ, EnergyIdentityError,
  Moment0, Moment1, MaxSimplex, MinY, MaxY, HelpersOk, RawFinite, Observations
};
struct Result {
  double v[300]{};
  int n = 0, observation = 0;
};
UGKWP_GAS_HD void segments(dp::Profile &p, const dp::Q *targets,
                           bool manufactured = false) {
  const double mid = p.in.matchingDistance / 2;
  p.segments.push_back({0, mid, targets[0], 0, {}, 0, 0, 0, true});
  p.segments.push_back({mid, p.in.matchingDistance, targets[1]});
  if (manufactured)
    for (auto &s : p.segments) {
      s.algebraicOmega = true;
      s.normalLo = s.normalHi = 0;
    }
}
UGKWP_GAS_HD void output(dp::Profile &p, Result &r) {
  r.n = 0;
  r.v[r.n++] = p.maxResidual;
  for (int i = 0; i < 2; ++i) {
    auto a = p.average(p.segments[i]), b = p.baseAverage(p.segments[i]);
    for (int j = 0; j < 9; ++j) {
      r.v[r.n++] = a[j];
      r.v[r.n++] = b[j];
    }
    auto src = p.source(i, p.cfg.coherentDiscretization ? 1 : 4);
    for (int j = 0; j < 2; ++j)
      r.v[r.n++] = src[j];
  }
  for (int k = 0; k <= 4; ++k) {
    double y = p.in.matchingDistance * k / 4.;
    auto pt = p.point(y), g = p.derivative(y);
    auto fl = p.flux(y, .015);
    r.v[r.n++] = pt.rho;
    r.v[r.n++] = pt.T;
    r.v[r.n++] = pt.k;
    if (k)
      r.v[r.n++] = pt.omega;
    for (int j = 0; j < 3; ++j) {
      r.v[r.n++] = pt.U[j];
      r.v[r.n++] = g.U[j];
    }
    for (int j = 0; j < 2; ++j) {
      r.v[r.n++] = pt.Y[j];
      r.v[r.n++] = g.Y[j];
    }
    for (int j = 0; j < 9; ++j)
      r.v[r.n++] = fl[j];
  }
  // Observe the selected affine state AND its actual published species/energy
  // flux. Iteration counts are not acceptance criteria or a GPU speedup claim.
  r.observation = r.n;
  double *o = r.v + r.n;
  r.n += Observations;
  o[AffineMode] = p.affineSpeciesMode;
  o[HelpersOk] = p.affineHelpersOk;
  o[RawFinite] = p.affineRawFinite;
  o[WallY] = p.point(0).Y[0];
  o[BaseWallY] = p.base(0).Y[0];
  o[WallJ] = p.flux(0)[7];
  o[MatchingY] = p.point(p.in.matchingDistance).Y[0];
  const double y = p.segments[0].hi;
  auto left = p.physicalPrimitive(y, -1), right = p.physicalPrimitive(y, 1);
  physical::Face<double, 2> fl, fr;
  physical::Source<double, 2> sl, sr;
  physical::evaluatePrimitive(p.in, p.cfg, left, y, p.in.normal, 0., fl, sl);
  physical::evaluatePrimitive(p.in, p.cfg, right, y, p.in.normal, 0., fr, sr);
  o[LeftJ] = fl.species[0] - fl.mass * left.Y[0];
  o[RightJ] = fr.species[0] - fr.mass * right.Y[0];
  double gl = 0, gr = 0;
  for (int a = 0; a < 3; ++a) {
    gl += left.gradY[0][a] * p.in.normal[a];
    gr += right.gradY[0][a] * p.in.normal[a];
    left.gradT[a] = (left.gradT[a] + right.gradT[a]) / 2;
    left.gradK[a] = (left.gradK[a] + right.gradK[a]) / 2;
    left.gradOmega[a] = (left.gradOmega[a] + right.gradOmega[a]) / 2;
    for (int j = 0; j < 2; ++j)
      left.gradY[j][a] = (left.gradY[j][a] + right.gradY[j][a]) / 2;
    for (int b = 0; b < 3; ++b)
      left.gradU[a][b] = (left.gradU[a][b] + right.gradU[a][b]) / 2;
  }
  o[CommonJ] = p.affineSpeciesMode == 1
      ? -(p.affineMobilityLeft * gl + p.affineMobilityRight * gr) / 2 : 0;
  physical::evaluatePrimitive(p.in, p.cfg, left, y, p.in.normal, 0., fl, sl);
  auto published = p.flux(y);
  o[PublishedJ0] = published[7] - published[0] * left.Y[0];
  o[PublishedJ1] = published[8] - published[0] * left.Y[1];
  double expectedEnergy = fl.energy;
  o[SumHJ] = o[PublishedSumHJ] = 0;
  for (int j = 0; j < 2; ++j) {
    double h = ugkwp::speciesH(j, left.T, p.in.model.thermo);
    double common = j == 0 ? o[CommonJ] : -o[CommonJ];
    o[SumHJ] += h * common;
    o[PublishedSumHJ] += h * (published[7 + j] - published[0] * left.Y[j]);
    if (p.affineSpeciesMode)
      expectedEnergy += h * (fl.mass * left.Y[j] + common - fl.species[j]);
  }
  o[EnergyIdentityError] = published[4] - expectedEnergy;
  for (int cell = 0; cell < 2; ++cell)
    o[Moment0 + cell] = p.average(p.segments[cell])[7] - p.segments[cell].target[7];
  o[MaxSimplex] = 0;
  o[MinY] = 1;
  o[MaxY] = 0;
  for (int k = 0; k <= 32; ++k) {
    auto q = p.point(p.in.matchingDistance * k / 32.);
    o[MaxSimplex] = dp::d::max(o[MaxSimplex], ::fabs(q.Y[0] + q.Y[1] - 1));
    o[MinY] = dp::d::min(o[MinY], q.Y[0]);
    o[MaxY] = dp::d::max(o[MaxY], q.Y[0]);
  }
}
void outputCpu(dp::Profile &p, dp::ProfileScratch &scratch, Result &r) {
  p.reconstructInternalTraces();
  if (!p.fitTeam(scratch, detail::SerialWallTeam{})) {
    r.n = -1;
    return;
  }
  output(p, r);
}
#ifndef UGKP_PROFILE_CPU_ONLY
__global__ __launch_bounds__(32) void reconstruct(
    dp::Input *in, dp::Config *c, dp::Work *w, WallOutput<double, 2> *out,
    dp::Q *targets, dp::ProfileScratch *scratch, Result *r, bool manufactured) {
  detail::CudaWallTeam team;
  dp::Profile p(*in, *c, *w, scratch->transformSlope, team);
  p.setWallFlux(*out);
  segments(p, targets, manufactured);
  p.reconstructInternalTraces();
  if (!p.fitTeam(*scratch, team)) {
    if (team.leader())
      r->n = -1;
    return;
  }
  auto s0 = p.sourceTeam(0, team, p.cfg.coherentDiscretization ? 1 : 4);
  auto s1 = p.sourceTeam(1, team, p.cfg.coherentDiscretization ? 1 : 4);
  if (team.leader()) {
    output(p, *r);
    r->v[19] = s0[0];
    r->v[20] = s0[1];
    r->v[39] = s1[0];
    r->v[40] = s1[1];
  }
}
__global__ __launch_bounds__(32) void benchmark(
    dp::Input *in, dp::Config *c, dp::Work *w, WallOutput<double, 2> *out,
    dp::Q *targets, dp::ProfileScratch *scratch, Result *r, bool parallel) {
  int slot = blockIdx.x;
  detail::CudaWallTeam team;
  dp::Profile p(*in, *c, *w, scratch[slot].transformSlope, team);
  p.setWallFlux(*out);
  segments(p, targets);
  p.reconstructInternalTraces();
  if (parallel) {
    bool fitted = p.fitTeam(scratch[slot], team);
    auto a = p.sourceTeam(0, team, p.cfg.coherentDiscretization ? 1 : 4), b = p.sourceTeam(1, team, p.cfg.coherentDiscretization ? 1 : 4);
    if (team.leader()) {
      r[slot].n = fitted ? 5 : -1;
      r[slot].v[0] = p.maxResidual;
      r[slot].v[1] = a[0]; r[slot].v[2] = a[1];
      r[slot].v[3] = b[0]; r[slot].v[4] = b[1];
    }
  } else if (team.leader()) {
    // The single-lane GPU control uses the SAME qualified fit, not Profile::fit.
    bool fitted = p.fitTeam(scratch[slot], detail::SerialWallTeam{});
    auto a = p.sourceTeam(0, detail::SerialWallTeam{}, p.cfg.coherentDiscretization ? 1 : 4);
    auto b = p.sourceTeam(1, detail::SerialWallTeam{}, p.cfg.coherentDiscretization ? 1 : 4);
    r[slot].n = fitted ? 5 : -1;
    r[slot].v[0] = p.maxResidual;
    r[slot].v[1] = a[0]; r[slot].v[2] = a[1];
    r[slot].v[3] = b[0]; r[slot].v[4] = b[1];
  }
}
#endif
int main() {
  auto in = managed<dp::Input>();
  auto c = managed<dp::Config>();
  auto w = managed<dp::Work>();
  auto out = managed<WallOutput<double, 2>>();
  auto targets = managed<dp::Q>(2);
#ifdef UGKP_PROFILE_CPU_ONLY
  const int pool = 1;
#else
  const int pool = std::getenv("UGKP_PROFILE_BENCH") ? 336 : 1;
#endif
  auto scratch = managed<dp::ProfileScratch>(pool);
  auto result = managed<Result>(pool);
  auto species = managed<ugkwp::SpeciesThermoData<double>>(2);
  auto coef = managed<double>(6);
  CapturedSstWall fixture;
  memcpy(species, fixture.species, sizeof(fixture.species));
  memcpy(coef, fixture.coefficients, sizeof(fixture.coefficients));
  int failures = 0;
  std::cout << std::setprecision(17)
            << "nodes,mixture,blowing,wall_boost,seed_bvp_residual,cpu_inventory_"
               "error,gpu_inventory_error,max_relative,values,gpu_ms\n";
  std::cerr << std::setprecision(17)
            << "ORACLE,current-shared-coherent-fitTeam-SerialWallTeam,"
               "independent-math-reference=false\n";
  auto compare = [&](const Result &cr, bool manufactured, int mixture,
                     double mass, double boost, double seedResidual) {
    if (cr.n < 0) { ++failures; result->n = -1; return; }
#ifdef UGKP_PROFILE_CPU_ONLY
    // Scalar oracle only: never report this as device execution or acceleration.
    *result = cr;
    for (int j = 0; j < cr.n; ++j)
      if (!std::isfinite(cr.v[j])) ++failures;
    failures += cr.v[0] > 1e-9;
    std::cout << c->nodes << ',' << mixture << ',' << mass << ',' << boost
              << ',' << seedResidual << ',' << cr.v[0]
              << ",UNRUN,UNRUN," << cr.n << ",UNRUN\n";
#else
    in->model.thermo.species = species;
    in->model.thermo.coefficients = coef;
    reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result, manufactured);
    ck(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a));
    reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result, manufactured);
    ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
    float ms;
    ck(cudaEventElapsedTime(&ms, a, b));
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    if (cr.n != result->n || !std::isfinite(cr.v[0]) ||
        !std::isfinite(result->v[0])) {
      ++failures;
      std::cerr << "profile-invalid-result," << c->nodes << ',' << cr.n
                << ',' << result->n << ',' << cr.v[0] << ',' << result->v[0] << '\n';
      return;
    }
    double err = 0;
    for (int j = 1; j < cr.n; ++j) {
      double e = std::abs(cr.v[j] - result->v[j]) / std::max(1., std::abs(cr.v[j]));
      if (!std::isfinite(e)) { ++failures; e = INFINITY; }
      err = std::max(err, e);
    }
    // Original device parity and inventory criteria remain unchanged.
    failures += err > 1e-7 || result->v[0] > 1e-9 || cr.n != result->n;
    std::cout << c->nodes << ',' << mixture << ',' << mass << ',' << boost
              << ',' << seedResidual << ',' << cr.v[0] << ',' << result->v[0]
              << ',' << err << ',' << cr.n << ',' << ms << std::endl;
#endif
  };
  // Preserve all eight existing profile inputs. evaluateWallReference supplies
  // the legacy BVP seed only; this harness does not qualify the coherent solver.
  for (int nodes : {24, 128})
    for (int mixture : {0, 1})
      for (double mass : {0., .01}) {
        *in = fixture.input;
        *c = fixture.config;
        c->nodes = nodes;
        c->maxIterations = 100;
        c->coherentDiscretization = true;
        if (mixture) {
          in->matching.massFraction[0] = .7;
          in->matching.massFraction[1] = .3;
        }
        in->massFlux[0] = mass * (mixture ? .9 : 1.);
        in->massFlux[1] = mass * (mixture ? .1 : 0.);
        const double boost = mixture && mass > 0 ? 30 : 0;
        in->velocity[0] += boost;
        in->matching.velocity[0] += boost;
        WallReferenceStats stats;
        WallStatus st;
        WallReferenceScratch<2, 128> ss;
        bool ok = evaluateWallReference(
            *in, *c, *w, ss, (const double *)nullptr, *out, st, stats);
        if (!ok || !stats.converged) return 3;
        dp::Profile cpu(*in, *c, *w, scratch->transformSlope);
        cpu.setWallFlux(*out);
        targets[0] = targets[1] = {};
        segments(cpu, targets);
        for (int i = 0; i < 2; ++i) {
          auto &q = cpu.segments[i].target;
          q = cpu.average(cpu.segments[i]);
          q[1] *= 1.02; q[4] *= 1.001; q[5] *= .98;
          if (i) q[6] *= 1.03;
          if (mixture) { q[7] = q[0] * .69; q[8] = q[0] * .31; }
          targets[i] = q;
        }
        Result cr;
        outputCpu(cpu, *scratch, cr);
        compare(cr, false, mixture, mass, boost, stats.residual);
        if (mixture) {
          auto saved = targets[0];
          targets[0][7] = targets[0][0]; targets[0][8] = 0;
#ifdef UGKP_PROFILE_CPU_ONLY
          dp::Profile zero(*in, *c, *w, scratch->transformSlope);
          zero.setWallFlux(*out); segments(zero, targets);
          outputCpu(zero, *scratch, *result);
#else
          reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result, false);
          ck(cudaDeviceSynchronize());
#endif
          failures += result->n != -1;
          std::cerr << "zero-inventory-fallback," << nodes << ',' << mass << ','
                    << result->n << '\n';
          targets[0] = saved;
        }
#ifndef UGKP_PROFILE_CPU_ONLY
        if (pool > 1 && !mixture && mass == 0) {
          double expected[4]{};
          for (int mode = 0; mode < 2; ++mode) {
            benchmark<<<pool, 32>>>(in, c, w, out, targets, scratch, result, mode);
            ck(cudaDeviceSynchronize());
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
            ck(cudaEventRecord(a));
            for (int repeat = 0; repeat < 3; ++repeat)
              benchmark<<<pool, 32>>>(in, c, w, out, targets, scratch, result, mode);
            ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
            float elapsed;
            ck(cudaEventElapsedTime(&elapsed, a, b));
            for (int j = 0; j < 4; ++j) {
              if (!mode) expected[j] = result[0].v[j + 1];
              else if (std::abs(result[0].v[j + 1] - expected[j]) >
                       1e-8 * std::max(1., std::abs(expected[j]))) return 7;
            }
            std::cerr << "BENCH_GPU_ONLY," << nodes << ',' << pool << ','
                      << (mode ? "warp32" : "single-lane") << ',' << elapsed / 3 << '\n';
            ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
          }
        }
#endif
      }
  // Bounded subset of the qualified manufactured family: equal molar masses,
  // constant-property no-blow base, shifted quadratic Y moments and both exact
  // pure orientations. Analytic seed: no nonlinear BVP or CPU parallel solve.
  std::cerr << "MANUFACTURED_HEADER,kind,mode,moment0,moment1,wallY,baseWallY,"
               "wallJ,matchingY,leftJ,rightJ,commonJ,publishedJ0,publishedJ1,"
               "sum_hJ,published_sum_hJ,energy_identity_error,helpers,finite\n";
  species[1].molarMass = species[0].molarMass;
  for (int kind : {0, 1, 2}) {
    *in = fixture.input; *c = fixture.config; *w = {}; *out = {};
    c->nodes = 24; c->enableSst = false; c->coherentDiscretization = true;
    in->model.thermo.species = species; in->model.thermo.coefficients = coef;
    const double H = .001, T = 300, Ybar = kind == 1 ? 0 : kind == 2 ? 1 : .8;
    const double amplitude = kind == 0 ? .02 : 0;
    in->temperature = in->matching.temperature = T;
    in->pressure = ugkwp::universalGasConstant<double>() / species[0].molarMass * T;
    in->ownerDistance = H / 4; in->matchingDistance = H;
    in->matching.k = 0; in->matching.omega = 1;
    in->matching.massFraction[0] = Ybar; in->matching.massFraction[1] = 1 - Ybar;
    in->model.molecularPrandtl = 0; in->model.conductivity = .024;
    in->volumeSstClosure = in->integrateOwnerKSource = false; in->quadrature = {};
    in->massFlux[0] = in->massFlux[1] = 0;
    for (int a = 0; a < 3; ++a) in->velocity[a] = in->matching.velocity[a] = 0;
    for (auto &row : in->sstTangentialGradient) for (double &v : row) v = 0;
    detail::LayerContext<double, 2> ctx;
    detail::makeContext(*in, *c, ctx);
    for (int i = 0; i < c->nodes; ++i) {
      w->y[i] = H * std::pow(double(i) / (c->nodes - 1), c->stretch);
      w->state[i][3] = in->matching.massFraction[ctx.independent[0]];
      w->state[i][5] = 1;
    }
    dp::Profile cpu(*in, *c, *w, scratch->transformSlope);
    cpu.setWallFlux(*out);
    for (int cell = 0; cell < 2; ++cell) {
      double lo = cell / 2., hi = (cell + 1) / 2.;
      auto integral = [](double x) { return x - x * x * x / 3; };
      dp::Primitive q{}; q.rho = 1; q.T = T; q.omega = 1;
      q.Y[0] = Ybar + amplitude * (integral(hi) - integral(lo)) / (hi - lo);
      q.Y[1] = 1 - q.Y[0]; targets[cell] = cpu.conserved(q);
    }
    segments(cpu, targets, true);
    Result cr; outputCpu(cpu, *scratch, cr);
    compare(cr, true, kind + 2, 0, 0, 0);
    for (const Result *r : {&cr, result}) {
      if (r->n < 0) { ++failures; continue; }
      const double *o = r->v + r->observation;
      failures += o[AffineMode] != (kind == 0 ? 1 : kind == 1 ? 2 : 3);
      failures += o[HelpersOk] != 1 || o[RawFinite] != 1;
      failures += r->v[0] > 1e-9 || std::abs(o[WallJ]) > 1e-9;
      failures += std::abs(o[LeftJ] - o[RightJ]) > 1e-9;
      failures += std::abs(o[PublishedJ0] - o[CommonJ]) > 1e-9;
      failures += std::abs(o[PublishedJ1] + o[CommonJ]) > 1e-9;
      failures += std::abs(o[PublishedSumHJ] - o[SumHJ]) > 1e-9;
      failures += std::abs(o[EnergyIdentityError]) > 1e-9;
      failures += o[MaxSimplex] > 1e-9 || o[MinY] < 0 || o[MaxY] > 1;
      failures += std::abs(o[MatchingY] - Ybar) > 1e-9;
      if (kind == 0) failures += !(o[WallY] > o[BaseWallY]);
      else failures += o[WallY] != Ybar || o[MinY] != Ybar || o[MaxY] != Ybar;
    }
    const double *o = cr.v + cr.observation;
    std::cerr << "MANUFACTURED," << kind << ',' << o[AffineMode] << ','
              << o[Moment0] << ',' << o[Moment1] << ',' << o[WallY] << ','
              << o[BaseWallY] << ',' << o[WallJ] << ',' << o[MatchingY] << ','
              << o[LeftJ] << ',' << o[RightJ] << ',' << o[CommonJ] << ','
              << o[PublishedJ0] << ',' << o[PublishedJ1] << ',' << o[SumHJ] << ','
              << o[PublishedSumHJ] << ',' << o[EnergyIdentityError] << ','
              << o[HelpersOk] << ',' << o[RawFinite] << '\n';
  }
  release(in); release(c); release(w); release(out); release(targets);
  release(scratch); release(result); release(species); release(coef);
  return failures ? 5 : 0;
}
