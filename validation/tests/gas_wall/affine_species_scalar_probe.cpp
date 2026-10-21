#include "common/gasWall/WallCoherentSolver.H"
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "validation/tests/gas_wall/fixtures/coherent_warm_regression.H"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace ugkwp::gaswall;
using namespace ugkwp::gaswall::reconstruction;

void checkNonlinearInterfacePublication() {
  CoherentWarmRegression fixture;
  auto cfg = fixture.config;
  cfg.coherentDiscretization = true;
  WallWorkspace<double, 2, 24> work{};
  fixture.acceptedSeed(work);
  Profile p(fixture.input, cfg, work);
  WallOutput<double, 2> output{};
  output.conductiveHeatFlux = 1000;
  output.traction[0] = 80;
  p.setWallFlux(output);
  const double middle = .00025477711111950772;
  for (int cell = 0; cell < 2; ++cell) {
    Segment segment{};
    segment.lo = cell ? middle : 0;
    segment.hi = cell ? fixture.input.matchingDistance : middle;
    segment.normalLo = segment.normalHi = 0;
    segment.algebraicOmega = cell == 0;
    segment.target = p.baseAverage(segment);
    segment.target[1] *= cell ? 1.005 : .995;
    segment.target[7] = segment.target[0] * (cell ? .80 : .79);
    segment.target[8] = segment.target[0] - segment.target[7];
    p.segments.push_back(segment);
  }
  p.reconstructInternalTraces();
  static ProfileScratch scratch;
  assert(p.fitTeam(scratch, detail::SerialWallTeam{}));
  auto left = p.physicalPrimitive(middle, -1);
  auto right = p.physicalPrimitive(middle, 1);
  physical::Face<double, 2> leftFlux, rightFlux, averagedFlux;
  physical::Source<double, 2> leftSource, rightSource, averagedSource;
  const auto &in = fixture.input;
  assert(physical::evaluatePrimitive(in, cfg, left, middle, in.normal, 0., leftFlux, leftSource));
  assert(physical::evaluatePrimitive(in, cfg, right, middle, in.normal, 0., rightFlux, rightSource));
  const double common = (leftFlux.species[0] + rightFlux.species[0]) / 2;
  const auto published = p.flux(middle);
  assert(std::abs(leftFlux.species[0] - rightFlux.species[0]) < 1e-12);
  assert(std::abs(published[7] - common) < 1e-12);
  assert(std::abs(leftSource.nut - rightSource.nut) > 1e-10);
  for (int a = 0; a < 3; ++a) {
    left.gradT[a] = (left.gradT[a] + right.gradT[a]) / 2;
    left.gradK[a] = (left.gradK[a] + right.gradK[a]) / 2;
    left.gradOmega[a] = (left.gradOmega[a] + right.gradOmega[a]) / 2;
    for (int j = 0; j < 2; ++j) left.gradY[j][a] = (left.gradY[j][a] + right.gradY[j][a]) / 2;
    for (int b = 0; b < 3; ++b) left.gradU[a][b] = (left.gradU[a][b] + right.gradU[a][b]) / 2;
  }
  assert(physical::evaluatePrimitive(in, cfg, left, middle, in.normal, 0., averagedFlux, averagedSource));
  double correctedEnergy = averagedFlux.energy;
  for (int j = 0; j < 2; ++j)
    correctedEnergy += ugkwp::speciesH(j, left.T, in.model.thermo) *
                       (published[7 + j] - averagedFlux.species[j]);
  assert(std::abs(published[4] - correctedEnergy) < 1e-9);
  std::cout << "nonlinear-interface," << published[7] << ',' << published[4]
            << ',' << p.maxResidual << '\n';
}

// Scalar-only production-entrypoint check. The exact constant base is supplied
// directly, so this probes reconstruction independently of a nonlinear BVP solve.
template<int Capacity = 24>
void checkProfile(double composition, double amplitude, bool coherent = true,
                  bool equalDiffusivity = true, bool observeExhaustion = false) {
  CapturedSstWall material;
  material.species[1].molarMass = material.species[0].molarMass;
  auto in = material.input;
  auto cfg = material.config;
  cfg.nodes = Capacity;
  cfg.enableSst = false;
  cfg.coherentDiscretization = coherent;
  // Test-only exhaustion observes publication semantics, not a production limit.
  if (observeExhaustion) cfg.maxIterations = 0;
  const double height = .001, middle = height / 2, temperature = 300;
  in.temperature = in.matching.temperature = temperature;
  in.pressure = ugkwp::universalGasConstant<double>() /
                material.species[0].molarMass * temperature;
  in.matchingDistance = height;
  in.ownerDistance = middle / 2;
  in.matching.k = 0;
  in.matching.omega = 1;
  in.matching.massFraction[0] = composition;
  in.matching.massFraction[1] = 1 - composition;
  in.model.molecularPrandtl = 0;
  in.model.conductivity = .024;
  if (!equalDiffusivity) in.model.diffusivity[1] *= 2;
  in.volumeSstClosure = in.integrateOwnerKSource = false;
  for (int a = 0; a < 3; ++a) in.velocity[a] = in.matching.velocity[a] = 0;
  detail::LayerContext<double, 2> context;
  assert(detail::makeContext(in, cfg, context));
  WallWorkspace<double, 2, Capacity> work{};
  for (int i = 0; i < cfg.nodes; ++i) {
    work.y[i] = height * std::pow(double(i) / (cfg.nodes - 1), cfg.stretch);
    work.state[i][3] = in.matching.massFraction[context.independent[0]];
    work.state[i][5] = 1;
  }
  Profile p(in, cfg, work);
  WallOutput<double, 2> output{};
  p.setWallFlux(output);
  for (int cell = 0; cell < 2; ++cell) {
    Segment segment{};
    segment.lo = cell ? middle : 0;
    segment.hi = cell ? height : middle;
    segment.normalLo = segment.normalHi = 0;
    segment.algebraicOmega = true;
    Primitive average{};
    average.rho = 1;
    average.T = temperature;
    average.omega = 1;
    const double lo = segment.lo / height, hi = segment.hi / height;
    average.Y[0] = composition + amplitude *
        (1 - (hi * hi + hi * lo + lo * lo) / 3);
    average.Y[1] = 1 - average.Y[0];
    segment.target = p.conserved(average);
    p.segments.push_back(segment);
  }
  p.reconstructInternalTraces();
  static ProfileScratch scratch;
  assert(p.fitTeam(scratch, detail::SerialWallTeam{}));
  const auto wall = p.point(0), top = p.point(height);
  const auto flux = p.flux(middle), wallFlux = p.flux(0);
  if (observeExhaustion) {
    assert(p.affineSpeciesMode == 1 && p.affineIterations == 0);
    assert(p.affineHelpersOk && p.affineRawFinite);
    assert(std::isfinite(p.maxResidual) && p.maxResidual > 1e-4);
    std::cout << "exhaustion-still-publishes," << p.maxResidual << '\n';
    return;
  }
  assert(p.affineHelpersOk && p.affineRawFinite);
  assert(p.maxResidual < 5e-13);
  assert(std::abs(top.Y[0] - composition) < 1e-14);
  assert(std::abs(wallFlux[7]) < 1e-13);
  assert(std::abs(flux[7] + flux[8] - flux[0]) < 1e-14);
  if (coherent && equalDiffusivity && amplitude > 0) {
    // Removing the spurious wall Dirichlet anchor is the regression under test.
    if (!(wall.Y[0] > composition + amplitude / 2)) {
      std::cerr << "wall composition is still anchored: " << wall.Y[0] << '\n';
      std::exit(1);
    }
    physical::Face<double, 2> left, right;
    physical::Source<double, 2> leftSource, rightSource;
    assert(physical::evaluatePrimitive(in, cfg, p.physicalPrimitive(middle, -1),
                                     middle, in.normal, 0., left, leftSource));
    assert(physical::evaluatePrimitive(in, cfg, p.physicalPrimitive(middle, 1),
                                     middle, in.normal, 0., right, rightSource));
    assert(std::abs(left.species[0] - right.species[0]) < 1e-13);
    assert(std::abs(flux[7] - left.species[0]) < 1e-13);
  } else if (!coherent || !equalDiffusivity) {
    assert(wall.Y[0] == composition);
  } else {
    assert(wall.Y[0] == composition && top.Y[0] == composition);
    assert(flux[7] == 0 && flux[8] == 0);
  }
  std::cout << std::setprecision(17) << Capacity << ',' << composition << ',' << amplitude << ','
            << coherent << ',' << equalDiffusivity << ',' << p.maxResidual
            << ',' << wall.Y[0] << ',' << flux[7] << ',' << flux[4] << '\n';
}
int main() {
  checkProfile(.8, .02);
  checkProfile(0, 0);
  checkProfile(1, 0);
  checkProfile(.8, .02, false);
  checkProfile(.8, .02, true, false);
  checkProfile<48>(.8, .02);
  checkProfile<128>(.8, .02);
  checkNonlinearInterfacePublication();
  checkProfile(.8, .02, true, true, true);
}
