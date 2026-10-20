#pragma once
// Opt-in readback at an existing output boundary; no extra work during
// stepping.
inline int writeConservativeWallAudit(DeviceState *s) {
  const char *directory = std::getenv("UGKP_WALL_AUDIT_DIR");
  if (!directory || !*directory)
    return 0;
  auto &m = s->gasBoundaryLayerModel;
  auto &d = m.conservative;
  const int nw = s->gasBoundaryLayer.count, nc = s->nCells, nf = s->nFaces;
  if (!d.enabled)
    return 0;
  std::vector<ugkwp::ConservativeWallColumn> columns(nw);
  std::vector<ugkwp::ConservativeWallFace> faces(3 * nw);
  std::vector<ugkwp::gaswall::WallReferenceCounters> counts(nw);
  std::vector<double> moment(nw), sk(nc), so(nc);
  std::vector<int> active(nw), fallbacks(nw);
  int rc =
      copyToHost(active.data(), d.active, nw, "audit stage active") |
      copyToHost(fallbacks.data(), d.fallbacks, nw, "audit fallback count") |
      copyToHost(columns.data(), d.columns, nw, "audit columns") |
      copyToHost(faces.data(), d.face, 3 * nw, "audit reconstructed faces") |
      copyToHost(counts.data(), m.referenceCounters, nw, "audit Newton work") |
      copyToHost(moment.data(), d.momentResidual, nw,
                 "audit inventory residual") |
      copyToHost(sk.data(), d.sourceK, nc, "audit k source") |
      copyToHost(so.data(), d.sourceOmega, nc, "audit omega source");
  std::vector<double> flux[9];
  const double *ptr[9] = {
      s->gasPhiRho,      s->gasPhiRhoUx,     s->gasPhiRhoUy,
      s->gasPhiRhoUz,    s->gasPhiRhoE,      s->sstPhiRhoK,
      s->sstPhiRhoOmega, s->gasSpecies.flux, s->gasSpecies.flux + nf};
  for (int j = 0; j < 9; ++j) {
    flux[j].resize(nf);
    rc |= copyToHost(flux[j].data(), ptr[j], nf, "audit final face flux");
  }
  if (rc || s->diagnosticOwner->completeReadback(0, true))
    return 1;
  auto open = [&](const char *name) {
    std::string path = std::string(directory) + "/" + name;
    return std::fopen(path.c_str(), "w");
  };
  FILE *f = open("wall-work.csv");
  if (!f) {
    setLastErrorText("cannot open requested wall audit output");
    return 1;
  }
  std::fprintf(
      f,
      "slot,owner,donor,solves,failures,iterations,residuals,jacobians,central,"
      "history,maxResidual,momentResidual,ownerSourceK,donorSourceK,"
      "donorSourceOmega,"
      "tractionX,tractionY,tractionZ,diffusiveEnergy,active,fallbacks\n");
  for (int i = 0; i < nw; ++i) {
    const auto &c = columns[i];
    if (c.cells[0] < 0)
      continue;
    const auto &t = counts[i];
    const auto &v = faces[3 * i];
    std::fprintf(f,
                 "%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.17g,%.17g,%."
                 "17g,%.17g,%.17g,%."
                 "17g,%.17g,%.17g,%.17g,%d,%d\n",
                 i, c.cells[0], c.cells[1], t.solves, t.failures, t.iterations,
                 t.residuals, t.jacobians, t.central, t.history, t.maxResidual,
                 moment[i], sk[c.cells[0]], sk[c.cells[1]], so[c.cells[1]],
                 -v.diffusion[1], -v.diffusion[2], -v.diffusion[3],
                 v.diffusion[4], active[i], fallbacks[i]);
  }
  std::fclose(f);
  f = open("wall-faces.csv");
  if (!f)
    return 1;
  std::fprintf(
      f, "slot,index,face,mass,mx,my,mz,energy,k,omega,species0,species1\n");
  for (int i = 0; i < nw; ++i)
    if (columns[i].cells[0] >= 0)
      for (int j = 0; j < 3; ++j) {
        int q = columns[i].faces[j];
        std::fprintf(f, "%d,%d,%d", i, j, q);
        for (int k = 0; k < 9; ++k)
          std::fprintf(f, ",%.17g", flux[k][q]);
        std::fprintf(f, "\n");
      }
  std::fclose(f);
  f = open("stage-flux.bin");
  if (!f)
    return 1;
  for (int j = 0; j < 9; ++j)
    std::fwrite(flux[j].data(), sizeof(double), nf, f);
  std::fclose(f);
  f = open("stage-sources.bin");
  if (!f)
    return 1;
  std::fwrite(sk.data(), sizeof(double), nc, f);
  std::fwrite(so.data(), sizeof(double), nc, f);
  std::fclose(f);
  return 0;
}
