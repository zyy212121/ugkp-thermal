#pragma once
inline int configureConservativeWallStorage(DeviceState *s,
                                            BoundaryLayerAllocation &candidate,
                                            const int *walls,
                                            const double *geometry) {
  if constexpr (ugkwp::compiledGasSpecies != 2)
    return 0;
  auto &m = candidate.gasBoundaryLayerModel;
  auto &d = m.conservative;
  const int count = candidate.gasBoundaryLayer.count;
  if (m.config.model != ugkwp::gaswall::BoundaryLayerModel::ReactingSst ||
      !m.config.enableSst)
    return 0;
  const int nf = s->nFaces, nc = s->nCells;
  std::vector<int> own(nf), nei(nf);
  std::vector<double> area(nf), vol(nc), cx(nf), cy(nf), cz(nf), sx(nf), sy(nf),
      sz(nf);
  int rc = copyToHost(own.data(), s->faceOwner, nf, "column owners") |
           copyToHost(nei.data(), s->faceNeighbour, nf, "column neighbours") |
           copyToHost(area.data(), s->magSf, nf, "column areas") |
           copyToHost(vol.data(), s->V, nc, "column volumes");
  rc |= copyToHost(cx.data(), s->faceCx, nf, "column face x") |
        copyToHost(cy.data(), s->faceCy, nf, "column face y") |
        copyToHost(cz.data(), s->faceCz, nf, "column face z") |
        copyToHost(sx.data(), s->Sfx, nf, "column area x") |
        copyToHost(sy.data(), s->Sfy, nf, "column area y") |
        copyToHost(sz.data(), s->Sfz, nf, "column area z");
  if (rc || s->diagnosticOwner->completeReadback(0, true))
    return 1;
  std::vector<std::vector<int>> cellFaces(nc);
  for (int f = 0; f < nf; ++f) {
    cellFaces[own[f]].push_back(f);
    if (nei[f] >= 0)
      cellFaces[nei[f]].push_back(f);
  }
  std::vector<int> wallOwner(nc, 0);
  for (int slot = 0; slot < count; ++slot)
    ++wallOwner[own[walls[slot]]];
  std::vector<ugkwp::ConservativeWallColumn> col(count);
  std::vector<int> cellSlot(nc, -1), faceSlot(nf, -1), faceIndex(nf, -1);
  int covered = 0;
  for (int slot = 0; slot < count; ++slot) {
    int wall = walls[slot], owner = own[wall];
    const double *n = geometry + 7 * slot;
    auto height = [&](int f) {
      return (cx[f] - cx[wall]) * n[0] + (cy[f] - cy[wall]) * n[1] +
             (cz[f] - cz[wall]) * n[2];
    };
    auto normal = [&](int f) {
      return sx[f] * n[0] + sy[f] * n[1] + sz[f] * n[2];
    };
    auto upper = [&](int c, double lo) {
      int result = -1;
      for (int f : cellFaces[c]) {
        double dot = std::abs(normal(f));
        if (dot < 1e-10 * area[f])
          continue;
        if (std::abs(dot - area[f]) > 1e-10 * area[f])
          return -1;
        if (height(f) > lo + 1e-12 && f < s->nInternalFaces) {
          if (result >= 0)
            return -1;
          result = f;
        }
      }
      return result;
    };
    int middle = upper(owner, 0);
    if (middle < 0)
      continue;
    int donor = own[middle] == owner ? nei[middle] : own[middle];
    if (donor < 0 || wallOwner[donor] || wallOwner[owner] != 1)
      continue;
    double lo = height(middle);
    int top = upper(donor, lo);
    if (top < 0)
      continue;
    double hi = height(top);
    if (!(hi > lo && lo > 0) ||
        std::abs(vol[owner] - area[wall] * lo) > 1e-8 * vol[owner] ||
        std::abs(vol[donor] - area[wall] * (hi - lo)) > 1e-8 * vol[donor])
      continue;
    if (std::abs(area[middle] - area[wall]) > 1e-10 * area[wall] ||
        std::abs(area[top] - area[wall]) > 1e-10 * area[wall])
      continue;
    if (cellSlot[owner] >= 0 || cellSlot[donor] >= 0 || faceSlot[wall] >= 0 ||
        faceSlot[middle] >= 0 || faceSlot[top] >= 0)
      continue;
    col[slot] = {{owner, donor}, {wall, middle, top}, {0, lo, hi}};
    cellSlot[owner] = cellSlot[donor] = slot;
    for (int j = 0; j < 3; ++j) {
      int f = col[slot].faces[j];
      faceSlot[f] = slot;
      faceIndex[f] = j;
    }
    ++covered;
  }
  if (!covered)
    return 0;
  rc |= uploadSharedGasTable(d.columns, col.data(), col.size());
  rc |= uploadSharedGasTable(d.cellSlot, cellSlot.data(), cellSlot.size());
  rc |= uploadSharedGasTable(d.faceSlot, faceSlot.data(), faceSlot.size());
  rc |= uploadSharedGasTable(d.faceIndex, faceIndex.data(), faceIndex.size());
  rc |= allocateSharedGasZero(d.baseInventory, 2 * count);
  rc |= allocateSharedGasZero(d.face, 3 * count);
  rc |= allocateSharedGasZero(d.sourceK, nc);
  rc |= allocateSharedGasZero(d.kFluxScale, nc);
  rc |= allocateSharedGasZero(d.sourceActivity, nc);
  rc |= allocateSharedGasZero(d.sourceOmega, nc);
  rc |= allocateSharedGasZero(d.momentResidual, count);
  rc |= allocateSharedGasZero(d.active, count);
  rc |= allocateSharedGasZero(d.fallbacks, count);
  rc |= allocateSharedGasZero(d.scratch, m.workspaceCount);
  rc |= allocateSharedGasZero(d.matching, count);
  rc |= allocateSharedGasZero(d.pressure, count);
  rc |= allocateSharedGasZero(d.projectedCells, std::size_t(50) * nc);
  rc |= allocateSharedGasZero(d.projectedFaces, std::size_t(8) * nf);
  rc |= allocateSharedGasZero(d.projectedStatus, nc + nf);
  unsigned char *view = nullptr;
  rc |= allocate(view, sizeof(WallProjectionView), "allocate PB operator view");
  d.projectedState = view;
  if (rc)
    return 1;
  d.enabled = true;
  std::fprintf(
      stderr,
      "conservative wall coverage: %d/%d columns, %d cells, %d shared faces\n",
      covered, count, 2 * covered, 3 * covered);
  return 0;
}
