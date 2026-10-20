#include "ConservativeProfile.hpp"
#include "common/gasWall/WallCudaTeam.cuh"
#include "common/gasWall/WallReferenceSolver.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace dp = ugkwp::gaswall::reconstruction;
namespace cp = conservative_profile;
using namespace ugkwp::gaswall;
void ck(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << cudaGetErrorString(e) << '\n';
    exit(2);
  }
}
template <class T> T *managed(int n = 1) {
  T *p;
  ck(cudaMallocManaged(&p, sizeof(T) * n));
  for (int i = 0; i < n; ++i)
    new (p + i) T{};
  return p;
}
struct Result {
  double v[300]{};
  int n = 0;
};
__host__ __device__ void output(dp::Profile &p, dp::ProfileScratch &scratch,
                                Result &r, bool fitted = false) {
  if (!fitted) {
    p.reconstructInternalTraces();
    p.fit(scratch);
  }
  r.n = 0;
  r.v[r.n++] = p.maxResidual;
  for (int i = 0; i < 2; ++i) {
    auto a = p.average(p.segments[i]), b = p.baseAverage(p.segments[i]);
    for (int j = 0; j < 9; ++j) {
      r.v[r.n++] = a[j];
      r.v[r.n++] = b[j];
    }
    auto src = p.source(i);
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
}
struct CpuProfile : cp::Profile {
  using cp::Profile::Profile;
  void fit(int &) { cp::Profile::fit(); }
};
void outputCpu(CpuProfile &p, int &scratch, Result &r, bool fitted = false) {
  if (!fitted) {
    p.reconstructInternalTraces();
    p.fit(scratch);
  }
  r.n = 0;
  r.v[r.n++] = p.maxResidual;
  for (int i = 0; i < 2; ++i) {
    auto a = p.average(p.segments[i]), b = p.baseAverage(p.segments[i]);
    for (int j = 0; j < 9; ++j) {
      r.v[r.n++] = a[j];
      r.v[r.n++] = b[j];
    }
    auto src = p.source(i);
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
}
__global__ void reconstruct(dp::Input *in, dp::Config *c, dp::Work *w,
                            WallOutput<double, 2> *out, dp::Q *targets,
                            dp::ProfileScratch *scratch, Result *r) {
  dp::Profile p(*in, *c, *w);
  p.setWallFlux(*out);
  double mid = in->matchingDistance / 2;
  p.segments.push_back({0, mid, targets[0], 0, {}, 0, 0, 0, true});
  p.segments.push_back({mid, in->matchingDistance, targets[1]});
  p.reconstructInternalTraces();
  detail::CudaWallTeam team;
  if (!p.fitTeam(*scratch, team)) {
    if (team.leader())
      r->n = -1;
    return;
  }
  auto s0 = p.sourceTeam(0, team);
  auto s1 = p.sourceTeam(1, team);
  if (team.leader()) {
    output(p, *scratch, *r, true);
    r->v[19] = s0[0];
    r->v[20] = s0[1];
    r->v[39] = s1[0];
    r->v[40] = s1[1];
  }
}
__global__ void benchmark(dp::Input *in, dp::Config *c, dp::Work *w,
                          WallOutput<double, 2> *out, dp::Q *targets,
                          dp::ProfileScratch *scratch, Result *r,
                          bool parallel) {
  int slot = blockIdx.x;
  dp::Profile p(*in, *c, *w);
  p.setWallFlux(*out);
  double mid = in->matchingDistance / 2;
  p.segments.push_back({0, mid, targets[0], 0, {}, 0, 0, 0, true});
  p.segments.push_back({mid, in->matchingDistance, targets[1]});
  p.reconstructInternalTraces();
  detail::CudaWallTeam team;
  if (parallel) {
    p.fitTeam(scratch[slot], team);
    auto a = p.sourceTeam(0, team), b = p.sourceTeam(1, team);
    if (team.leader()) {
      r[slot].v[0] = p.maxResidual;
      r[slot].v[1] = a[0];
      r[slot].v[2] = a[1];
      r[slot].v[3] = b[0];
      r[slot].v[4] = b[1];
    }
  } else if (team.leader()) {
    p.fit(scratch[slot]);
    auto a = p.source(0), b = p.source(1);
    r[slot].v[0] = p.maxResidual;
    r[slot].v[1] = a[0];
    r[slot].v[2] = a[1];
    r[slot].v[3] = b[0];
    r[slot].v[4] = b[1];
  }
}
int main() {
  auto in = managed<dp::Input>();
  auto c = managed<dp::Config>();
  auto w = managed<dp::Work>();
  auto out = managed<WallOutput<double, 2>>();
  auto targets = managed<dp::Q>(2);
  const int pool = std::getenv("UGKP_PROFILE_BENCH") ? 336 : 1;
  auto scratch = managed<dp::ProfileScratch>(pool);
  auto result = managed<Result>(pool);
  auto species = managed<ugkwp::SpeciesThermoData<double>>(2);
  auto coef = managed<double>(6);
  CapturedSstWall fixture;
  memcpy(species, fixture.species, sizeof(fixture.species));
  memcpy(coef, fixture.coefficients, sizeof(fixture.coefficients));
  int failures = 0;
  std::cout << std::setprecision(17)
            << "nodes,mixture,blowing,wall_boost,bvp_residual,cpu_inventory_"
               "error,gpu_inventory_"
               "error,max_relative,values,gpu_ms\n";
  for (int nodes : {24, 128})
    for (int mixture : {0, 1})
      for (double mass : {0., .01}) {
        *in = fixture.input;
        *c = fixture.config;
        c->nodes = nodes;
        c->maxIterations = 100;
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
        if (!ok || !stats.converged)
          return 3;
        CpuProfile cpu(*in, *c, *w);
        cpu.setWallFlux(*out);
        double mid = in->matchingDistance / 2;
        cpu.segments.push_back({0, mid, {}, 0, {}, 0, 0, 0, true});
        cpu.segments.push_back({mid, in->matchingDistance, {}});
        for (int i = 0; i < 2; ++i) {
          auto &q = cpu.segments[i].target;
          q = cpu.average(cpu.segments[i]);
          q[1] *= 1.02;
          q[4] *= 1.001;
          q[5] *= .98;
          if (i)
            q[6] *= 1.03;
          if (mixture) {
            q[7] = q[0] * .69;
            q[8] = q[0] * .31;
          }
          for (int j = 0; j < 9; ++j)
            targets[i][j] = q[j];
        }
        Result cr;
        int unused = 0;
        outputCpu(cpu, unused, cr);
        in->model.thermo.species = species;
        in->model.thermo.coefficients = coef;
        reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result);
        ck(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a));
        ck(cudaEventCreate(&b));
        ck(cudaEventRecord(a));
        reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result);
        ck(cudaEventRecord(b));
        ck(cudaEventSynchronize(b));
        float ms;
        ck(cudaEventElapsedTime(&ms, a, b));
        double err = 0;
        for (int j = 1; j < cr.n; ++j) {
          double e = std::abs(cr.v[j] - result->v[j]) /
                     std::max(1., std::abs(cr.v[j]));
          if (!std::isfinite(e))
            return 4;
          err = std::max(err, e);
        }
        failures += err > 1e-7 || result->v[0] > 1e-9 || cr.n != result->n;
        std::cout << nodes << ',' << mixture << ',' << mass << ',' << boost
                  << ',' << stats.residual << ',' << cr.v[0] << ','
                  << result->v[0] << ',' << err << ',' << cr.n << ',' << ms
                  << std::endl;
        ck(cudaEventDestroy(a));
        ck(cudaEventDestroy(b));
        if (mixture) {
          auto saved = targets[0];
          targets[0][7] = targets[0][0];
          targets[0][8] = 0;
          reconstruct<<<1, 32>>>(in, c, w, out, targets, scratch, result);
          ck(cudaDeviceSynchronize());
          failures += result->n != -1;
          std::cerr << "zero-inventory-fallback," << nodes << ',' << mass << ','
                    << result->n << '\n';
          targets[0] = saved;
        }
        if (pool > 1 && !mixture && mass == 0) {
          double expected[4]{};
          for (int mode = 0; mode < 2; ++mode) {
            benchmark<<<pool, 32>>>(in, c, w, out, targets, scratch, result,
                                    mode);
            ck(cudaDeviceSynchronize());
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a));
            ck(cudaEventCreate(&b));
            ck(cudaEventRecord(a));
            for (int repeat = 0; repeat < 3; ++repeat)
              benchmark<<<pool, 32>>>(in, c, w, out, targets, scratch, result,
                                      mode);
            ck(cudaEventRecord(b));
            ck(cudaEventSynchronize(b));
            float elapsed;
            ck(cudaEventElapsedTime(&elapsed, a, b));
            for (int j = 0; j < 4; ++j) {
              if (!mode)
                expected[j] = result[0].v[j + 1];
              else if (std::abs(result[0].v[j + 1] - expected[j]) >
                       1e-8 * std::max(1., std::abs(expected[j])))
                return 7;
            }
            std::cerr << "BENCH," << nodes << ',' << pool << ',' << mode << ','
                      << elapsed / 3 << '\n';
            ck(cudaEventDestroy(a));
            ck(cudaEventDestroy(b));
          }
        }
      }
  for (void *p :
       {(void *)in, (void *)c, (void *)w, (void *)out, (void *)targets,
        (void *)scratch, (void *)result, (void *)species, (void *)coef})
    ck(cudaFree(p));
  return failures ? 5 : 0;
}
