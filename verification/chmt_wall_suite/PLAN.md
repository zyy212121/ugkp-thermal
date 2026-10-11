# CHMT wall validation suite implementation plan

Goal: provide native-executable cases, independent metrics and reproducible parameter/source cards for three validation classes without changing production code or any README.

Architecture: a Python standard-library case catalog drives disposable input generation, native mesh/solver execution and independent result readers. Existing production case generators are reused where they already define the right problem. Real experiment downloads remain opt-in and outside the checkout. CPU reference calculations and CUDA runs carry separate status.

1. Tests first: catalog coverage, case isolation, input generation, analytic cell averages, falsified output rejection, precision/status handling, licensing and provenance.
2. Implement generators for legacy Couette/Sod; gas species transport and reactor; two-region CHMT thermal/reactive/moving controls; fixed and moving flat-plate/MSS7 wall-family comparisons.
3. Implement native runner, dependency checks, exact source/binary/input fingerprints, output metrics and paired comparison. Preparation and unavailable hardware stay NOT_RUN.
4. Add complete synthetic material cards; versioned external source cards/download manifest with verified checksums and limitations; standalone VALIDATION.md.
5. Run Python tests, source guard, fresh-case generation, OpenFOAM mesh/input checks where available; commit only suite files. Native GPU cannot be claimed without actual execution.

Critical edge cases: pre-existing output directories, Git-LFS pointers, nonfinite/missing fields, incomplete native runs, comparing unlike physical models, unknown actual y+, missing or time-shifted experimental data, and unsupported full material thermo must fail closed or report unavailable rather than produce a PASS.

## Bounded repair plan, 2026-10-11

Scope: suite files only; no README changes, production edits, acceptance relaxation or long native runs. Preserve original-head evidence separately. Run pytest with `-p no:cacheprovider` and disable bytecode output.

1. Add failing material-volume tests with multiple condensed densities, actual per-cell porosity, permuted cell rows, and malformed inventories. Replace the first-component-only calculation with a named-component inventory calculation. Label endpoint mass/porosity volume consistency separately from independently integrated surface recession; missing accepted packet evidence remains unavailable.
2. Add failing eligibility tests for measured lowRe y+ around 51, missing y+, missing/invalid target and observations outside the final accepted endpoint. Add explicit baseline eligibility and target metadata without turning report-only comparisons into physical PASS claims or adjusting user meshes.
3. Add failing performance-evidence tests for concurrent/instrumented/short/invalid samples. Report linear estimates as diagnostic with their validity limits; never interpret process wall time as isolated GPU throughput.
4. Keep actual accepted cumulative budget availability explicit. The concurrent production fixes do not add a cumulative gas ledger or new CSV schema, so instantaneous wall rows remain insufficient. Add tests pinning this distinction.
5. Retain and run existing controlDict replacement, numerical-capability, endpoint scheduling and backend-identity tests; add any newly reproduced configuration failure before repairing it. Rerun the entire suite, record unavailable OpenFOAM/CUDA checks, and commit only independently verified suite changes.

New SST coupling, geometry algorithm and authoritative FP64 lifecycle tests await their defined production interfaces. This suite repair does not imply those production changes are accepted.
