# Native coupled runtime acceptance

## Evidence boundary

Native CUDA execution in the rebuild environment: **NOT_RUN** (no `nvcc` or
usable CUDA device). Real Foundation OpenFOAM 10 mesh generation and input import,
C++ output/checkpoint tests, and synthetic checker tests are separate evidence.
None can produce a native acceptance claim. The scripts below are the pending
GPU acceptance gate; all must pass on the target CUDA machine.

## Run on a native CUDA machine

Source Foundation OpenFOAM 10 first. Set `CHMT_CUDA_ARCH` to an architecture
supported by the actual GPU and compiler, for example `sm_80`. From repository
root, choose fresh output directories:

```bash
export CHMT_CUDA_ARCH=sm_80
bash applications/CHMT/devtools/check_native_coupled.sh /tmp/chmt-native-single --single
bash applications/CHMT/devtools/check_native_coupled.sh /tmp/chmt-native-thermal
bash applications/CHMT/devtools/check_native_coupled.sh /tmp/chmt-native-diffusion --diffusion
bash applications/CHMT/devtools/check_native_coupled.sh /tmp/chmt-native-chemistry --chemistry
```

The launcher requires `nvcc`, an actual CUDA device/kernel roundtrip, Foundation
OpenFOAM 10, and `blockMesh`. It builds the real application itself with Ns=2
(thermal/diffusion) or Ns=10 (chemistry), records the native build manifest and
binary hash, and retains all meshes, inputs, runtime logs, CSVs and checkpoints.
No host emulation, frontend-only executable or pre-existing replacement solver
is accepted. Failure exits nonzero and leaves diagnostic logs; only a complete
success writes `native-acceptance.json` with `native-cuda-coupled-runtime` evidence.

## Physics and independent acceptance checks

All cases use a stationary, insulated pair of unit cubes: eight gas cells,
eight dense-solid cells, and four conformal contact faces. The solid initially
has density 1000 kg/m3, heat capacity 1000 J/(kg K), and temperature 300 K.
Thermal/diffusion gas starts at 600 K. Each side has conductivity 1 W/(m K).

The two cell-centre/contact distances are 0.25 m. Series resistance therefore
implies UA = 1 / (0.25/1 + 0.25/1) = 2 W/K over the unit interface. Initially the
thermal heat rate is 600 W into the material, about 0.3 J per 0.0005 s macro
window. The checker bounds actual heat with endpoint temperature extrema and
5% temporal slack. It rejects reversed heat or a balanced but incorrect
conductance. The refinement companion halves both macro interval H and gas dt,
keeps the final time fixed, and requires transferred heat agreement within 0.5%
plus documented binary64 inventory-subtraction allowance. This is a refinement
stability test, not a measured convergence-order claim.

Independent Python checks, without importing solver thermodynamic or ledger
code, require:

- Closed gas/material mass and frozen-species conservation, positive inventories
- Linear-cp or NASA7 formation-inclusive caloric energy and ideal-mixture EOS
  pressure for every gas cell; independent material caloric temperatures
- Closed total energy, resolved gas energy loss and material energy gain
- Signed conductive packet energy equal to both participants' inventory changes,
  correct consumer masks, no advective/radiative/mechanical sources, and only
  zero-valued pore bookkeeping packets in this dry case
- Consecutive accepted/commit counters, complete paired field sets, no orphan
  trial fields, increasing time, and the requested final time
- Schema-5 checkpoint header, fixed field-schema fingerprint, FNV-1a checksum,
  and independently decoded saved time/accepted/rejected/commit counters
- A real split/restart continuation: bitwise initial checkpoint reserialization
  and final fields/summary matching the uninterrupted native run

The absolute energy allowance is at least 2e-5 J, or 2e-13 of the initial total
energy when larger. For the 3e8 J solid this is approximately 6e-5 J, much smaller
than the intended transfer. It is not used to repair or rebalance inventories.

The single-gas gate uses `gasMode single`, `singleGasSpecies S0`, and full real
S0/S1 thermodynamic metadata in `materialGasProperties`. Gas species input
fields are absent. The launcher also runs the frozen-mixture companion using
the identical Ns=2 executable without an intervening rebuild. Both retain the
same material metadata and must pass the coupled physical/restart checks.

The diffusion case starts with alternating 0.8/0.2 species fractions. Both
species have identical molecular/caloric properties. Constant diffusivity is
0.02 m2/s, and a matched zero-D run must show less physical mixing: the active
case must reduce mass-weighted composition variance beyond the zero-D result.
Thus ordinary numerical convective mixing alone cannot satisfy this gate.

The chemical case uses the pinned 10-species H2/O2/N2 model and the established
Cantera nitrogen_1200K_1atm initial density, composition and energy. It checks
atomic conservation and a resolved H2/O2 decrease with H2O formation, exceeding
a matched chemistry-disabled frozen run's species drift by at least 100 times.
It still includes the solid contact, energy/exchange and restart gates. End time
is 2e-5 s, H is 1e-5 s, and gas dt is at most 1e-6 s. This is a coupled runtime
acceptance test, not a replacement for the separate mechanism-rate/reactor
accuracy tests.

## Host checks only

```bash
python3 -m pytest -q \
  applications/CHMT/tests/test_native_coupled_checker.py \
  applications/CHMT/tests/test_coupled_output_writer.py
```

These tests intentionally construct synthetic CSVs to prove the checker rejects
corrupted evidence; they also decode an actual CPU production-writer checkpoint.
They do not run CUDA. Direct use of `check_case.py` reports
`independent-output-checks-only`, regardless of its input's origin.
