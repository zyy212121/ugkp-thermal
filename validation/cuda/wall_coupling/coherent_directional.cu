// Native CUDA coverage for the coherent FD solver, separate from profile fitting.
// The host oracle is scalar shared code; the GPU performs its entire solve on
// one native wall team. No CPU cooperative emulator or CPU fallback is used.
#ifndef UGKP_COHERENT_CPU_ONLY
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#endif
#include "common/gasWall/WallCoherentSolver.H"
#include "validation/tests/gas_wall/fixtures/coherent_directional_723.H"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
using namespace ugkwp::gaswall;
using Input = WallInput<double, 2>;
using Config = WallModelConfig<double>;
using Work = WallWorkspace<double, 2, 24>;
using Output = WallOutput<double, 2>;
struct Stats {
  int iterations = 0, residualEvaluations = 0, factorizations = 0;
  double physicalResidual = 0, seconds = 0;
};
void seed(int mode, Work &work, coherent::Cache<24> &cache) {
  work = {};
  cache = {};
  if (mode == 0) CoherentDirectional723::acceptedSeed(work);
  if (mode == 1) CoherentDirectional723::failedSeed(work);
  cache.haveState = mode != 2;
}
double relative(double got, double expected, double floor = 1e-8) {
  if (!std::isfinite(got) || !std::isfinite(expected))
    return std::numeric_limits<double>::infinity();
  return std::abs(got - expected) / std::max(floor, std::abs(expected));
}
double stateError(const Work &got, const Work &expected) {
  double error = 0;
  for (int i = 0; i < 24; ++i) {
    error = std::max(error, relative(got.y[i], expected.y[i]));
    for (int j = 0; j < 6; ++j)
      error = std::max(error, relative(got.state[i][j], expected.state[i][j]));
  }
  return error;
}
#ifndef UGKP_COHERENT_CPU_ONLY
void check(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cerr << cudaGetErrorString(code) << '\n';
    std::exit(2);
  }
}
template<class T> T *managed(int count = 1) {
  T *p = nullptr;
  check(cudaMallocManaged(&p, sizeof(T) * count));
  for (int i = 0; i < count; ++i) new (p + i) T{};
  return p;
}
__global__ __launch_bounds__(32) void solve(
    Input *in, Config *cfg, Work *work, coherent::Cache<24> *cache,
    coherent::Scratch<24> *scratch, Output *output, Stats *stats, bool *published) {
  detail::CudaWallTeam team;
  coherent::Engine<24, detail::CudaWallTeam> engine(*scratch, team);
  const bool result = engine.solve(*in, *cfg, *work, *cache, *output, *stats);
  if (team.leader()) *published = result;
}
#endif
// Host-only comparator is also exercised by the CPU-only build.
double outputError(const Output &got, const Output &expected) {
  double error = 0;
  auto compare = [&](double a, double b) { error = std::max(error, relative(a, b, 1.)); };
  compare(got.traceTemperature, expected.traceTemperature);
  compare(got.traceDensity, expected.traceDensity);
  compare(got.conductiveHeatFlux, expected.conductiveHeatFlux);
  compare(got.integratedKSource, expected.integratedKSource);
  compare(got.ownerOmega, expected.ownerOmega);
  compare(got.wallKFlux, expected.wallKFlux);
  compare(got.ownerKContent, expected.ownerKContent);
  compare(got.ownerKDestruction, expected.ownerKDestruction);
  compare(got.ownerKBalanceResidual, expected.ownerKBalanceResidual);
  compare(got.heatConduction, expected.heatConduction);
  compare(got.heatViscous, expected.heatViscous);
  compare(got.heatTransport, expected.heatTransport);
  compare(got.heatBalanceResidual, expected.heatBalanceResidual);
  for (int j = 0; j < 3; ++j) {
    compare(got.traceVelocity[j], expected.traceVelocity[j]);
    compare(got.traction[j], expected.traction[j]);
  }
  for (int j = 0; j < 2; ++j) {
    compare(got.traceMassFraction[j], expected.traceMassFraction[j]);
    compare(got.wallSpeciesFlux[j], expected.wallSpeciesFlux[j]);
    compare(got.matchingSpeciesFlux[j], expected.matchingSpeciesFlux[j]);
    compare(got.reactionIntegral[j], expected.reactionIntegral[j]);
    // Missing coarse chemistry data deliberately publishes NaN placeholders;
    // their availability flag, not finiteness, defines this optional field.
    if (expected.chemistryMismatchAvailable)
      compare(got.chemistryMismatch[j], expected.chemistryMismatch[j]);
    else if (!std::isnan(got.chemistryMismatch[j]) ||
             !std::isnan(expected.chemistryMismatch[j]))
      return std::numeric_limits<double>::infinity();
    compare(got.speciesBalanceResidual[j], expected.speciesBalanceResidual[j]);
  }
  if (got.kFluxCount != expected.kFluxCount ||
      got.chemistryMismatchAvailable != expected.chemistryMismatchAvailable)
    return std::numeric_limits<double>::infinity();
  for (int i = 0; i < expected.kFluxCount; ++i) {
    compare(got.kFluxHeight[i], expected.kFluxHeight[i]);
    compare(got.kTransportFlux[i], expected.kTransportFlux[i]);
    compare(got.omegaDiffusiveFlux[i], expected.omegaDiffusiveFlux[i]);
  }
  return error;
}
int main() {
  CoherentDirectional723 fixture;
  auto cfg = fixture.config;
  cfg.coherentDiscretization = true;
  const double tolerance = cfg.absoluteTolerance + cfg.relativeTolerance;
  auto scalarScratch = std::make_unique<coherent::Scratch<24>>();
#ifndef UGKP_COHERENT_CPU_ONLY
  auto input = managed<Input>();
  auto config = managed<Config>();
  auto work = managed<Work>();
  auto cache = managed<coherent::Cache<24>>();
  auto scratch = managed<coherent::Scratch<24>>();
  auto output = managed<Output>();
  auto stats = managed<Stats>();
  auto published = managed<bool>();
  auto species = managed<ugkwp::SpeciesThermoData<double>>(2);
  auto coefficients = managed<double>(6);
  auto distances = managed<double>(47);
  auto weights = managed<double>(47);
  std::memcpy(species, fixture.species, sizeof(fixture.species));
  std::memcpy(coefficients, fixture.coefficients, sizeof(fixture.coefficients));
  std::memcpy(distances, fixture.quadratureY, sizeof(fixture.quadratureY));
  std::memcpy(weights, fixture.quadratureWeight, sizeof(fixture.quadratureWeight));
  *input = fixture.input;
  input->model.thermo.species = species;
  input->model.thermo.coefficients = coefficients;
  input->model.thermo.diagnostics = nullptr;
  input->diagnostics = nullptr;
  input->quadrature.distance = distances;
  input->quadrature.volumeWeight = weights;
  *config = cfg;
#endif
  std::cout << std::setprecision(17)
      << "mode,cpu_ok,cpu_residual,cpu_iterations,cpu_F,cpu_J,frozen_state_error,"
         "gpu_ok,gpu_residual,gpu_iterations,gpu_F,gpu_J,state_error,output_error,residual_error\n";
  int failures = 0;
  for (int mode = 0; mode < 3; ++mode) {
    Work scalar{}, frozen{};
    coherent::Cache<24> scalarCache;
    seed(mode, scalar, scalarCache);
    *scalarScratch = {};
    Output scalarOutput{};
    Stats scalarStats{};
    coherent::Engine<24> engine(*scalarScratch);
    const bool scalarOk = engine.solve(fixture.input, cfg, scalar, scalarCache,
                                      scalarOutput, scalarStats);
    fixture.expectedRoot(mode, frozen);
    const double frozenError = stateError(scalar, frozen);
    failures += !scalarOk || !(scalarStats.physicalResidual <= tolerance) ||
                frozenError > 1e-10 || outputError(scalarOutput, scalarOutput) != 0;
    std::cout << mode << ',' << scalarOk << ',' << scalarStats.physicalResidual << ','
              << scalarStats.iterations << ',' << scalarStats.residualEvaluations << ','
              << scalarStats.factorizations << ',' << frozenError;
#ifdef UGKP_COHERENT_CPU_ONLY
    std::cout << ",UNRUN,UNRUN,UNRUN,UNRUN,UNRUN,UNRUN,UNRUN,UNRUN\n";
#else
    // Every replay starts with a fresh wall-owned LU cache. The two supplied
    // states do not reconstruct an unavailable historical cache chronology.
    seed(mode, *work, *cache);
    *scratch = {};
    *output = {};
    *stats = {};
    *published = false;
    solve<<<1, 32>>>(input, config, work, cache, scratch, output, stats, published);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    const double stateDifference = stateError(*work, scalar);
    const double outputDifference = outputError(*output, scalarOutput);
    double residualDifference = 0;
    for (int i = 0; i < cfg.nodes; ++i)
      for (int j = 0; j < 6; ++j) {
        const double a = work->residual[i][j], b = scalar.residual[i][j];
        residualDifference = std::max(residualDifference,
            std::isfinite(a) && std::isfinite(b) ? std::abs(a - b)
                : std::numeric_limits<double>::infinity());
      }
    // These are test comparisons only. Production retains its existing
    // publication/nonconvergence semantics and original physical tolerance.
    failures += !*published || !(stats->physicalResidual <= tolerance) ||
                stateDifference > 1e-5 || outputDifference > 1e-5 ||
                residualDifference > 2 * tolerance;
    std::cout << ',' << *published << ',' << stats->physicalResidual << ','
              << stats->iterations << ',' << stats->residualEvaluations << ','
              << stats->factorizations << ',' << stateDifference << ','
              << outputDifference << ',' << residualDifference << '\n';
#endif
  }
#ifndef UGKP_COHERENT_CPU_ONLY
  check(cudaFree(input)); check(cudaFree(config)); check(cudaFree(work));
  check(cudaFree(cache)); check(cudaFree(scratch)); check(cudaFree(output));
  check(cudaFree(stats)); check(cudaFree(published)); check(cudaFree(species));
  check(cudaFree(coefficients)); check(cudaFree(distances)); check(cudaFree(weights));
#endif
  return failures ? 1 : 0;
}
