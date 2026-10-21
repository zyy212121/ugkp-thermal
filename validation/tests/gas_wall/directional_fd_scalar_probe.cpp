#ifndef COHERENT_SOLVER_HEADER
#define COHERENT_SOLVER_HEADER "common/gasWall/WallCoherentSolver.H"
#endif
#include COHERENT_SOLVER_HEADER
#include "validation/tests/gas_wall/fixtures/coherent_directional_723.H"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace ugkwp::gaswall;
struct Stats {
  int iterations=0,residualEvaluations=0,factorizations=0;
  double physicalResidual=0,seconds=0;
};

// Fresh-cache replay of the recorded failed terminal iterate, not a claim to
// reconstruct the original pre-call warm-cache chronology.
int main() {
  CoherentDirectional723 fixture;
  // Alternative-probe helpers are not input-domain preflight. In particular,
  // the existing signed sqrt(k) coordinate remains valid on both sides of zero.
  assert(coherent::fdProbeAdmissible(fixture.input, 4, -1.));
  assert(coherent::fdProbeAdmissible(fixture.input, 4, 0.));
  assert(!coherent::fdProbeAdmissible(fixture.input, 3, -1e-9));
  assert(!coherent::fdProbeAdmissible(fixture.input, 3, 1. + 1e-9));
  assert(!coherent::fdProbeAdmissible(fixture.input, 5, -1e-9));
  int probes = 0;
  assert(coherent::fdSelectDirection([&](int sign) { ++probes; return sign < 0; }) == -1);
  assert(probes == 2);
  auto cfg = fixture.config;
  cfg.coherentDiscretization = true;
  for (int mode = 0; mode < 3; ++mode) {
    WallWorkspace<double, 2, 24> work{};
    coherent::Cache<24> cache;
    if (mode == 0) {
      fixture.acceptedSeed(work);
      cache.haveState = true;
    } else if (mode == 1) {
      fixture.failedSeed(work);
      cache.haveState = true;
    }
    auto scratch = std::make_unique<coherent::Scratch<24>>();
    coherent::Engine<24> engine(*scratch);
    WallOutput<double, 2> output;
    Stats stats;
    const bool published = engine.solve(fixture.input, cfg, work, cache, output, stats);
    int expectedMode = mode;
#ifdef COHERENT_SCALAR_RECOVERY_EXERCISE
    if (mode == 1) expectedMode = 3;
#endif
    const auto &expected = fixture.expected[expectedMode];
    std::cout << std::setprecision(17) << mode << ',' << published << ','
              << stats.physicalResidual << ',' << stats.iterations << ','
              << stats.residualEvaluations << ',' << stats.factorizations << ','
              << output.conductiveHeatFlux << ',' << output.traction[0] << '\n';
    if (!published || !(stats.physicalResidual <= cfg.absoluteTolerance + cfg.relativeTolerance)) {
      std::cerr << "captured root did not converge at original tolerance\n";
      return 1;
    }
    assert(stats.iterations == expected.iterations);
    assert(stats.residualEvaluations == expected.residualEvaluations);
    assert(stats.factorizations == expected.factorizations);
    assert(stats.physicalResidual == expected.residual);
    assert(output.conductiveHeatFlux == expected.heat);
    assert(output.traction[0] == expected.shear);
    WallWorkspace<double, 2, 24> reference{};
    fixture.expectedRoot(expectedMode, reference);
    for (int i = 0; i < cfg.nodes; ++i) {
      assert(work.y[i] == reference.y[i]);
      for (int j = 0; j < 6; ++j) {
        assert(work.state[i][j] == reference.state[i][j]);
        assert(work.residual[i][j] == reference.residual[i][j]);
      }
    }
  }
}
