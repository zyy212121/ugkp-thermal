# CHMT wall validation suite implementation plan

Goal: provide native-executable cases, independent metrics and reproducible parameter/source cards for three validation classes without changing production code or any README.

Architecture: a Python standard-library case catalog drives disposable input generation, native mesh/solver execution and independent result readers. Existing production case generators are reused where they already define the right problem. Real experiment downloads remain opt-in and outside the checkout. CPU reference calculations and CUDA runs carry separate status.

1. Tests first: catalog coverage, case isolation, input generation, analytic cell averages, falsified output rejection, precision/status handling, licensing and provenance.
2. Implement generators for legacy Couette/Sod; gas species transport and reactor; two-region CHMT thermal/reactive/moving controls; fixed and moving flat-plate/MSS7 wall-family comparisons.
3. Implement native runner, dependency checks, exact source/binary/input fingerprints, output metrics and paired comparison. Preparation and unavailable hardware stay NOT_RUN.
4. Add complete synthetic material cards; versioned external source cards/download manifest with verified checksums and limitations; standalone VALIDATION.md.
5. Run Python tests, source guard, fresh-case generation, OpenFOAM mesh/input checks where available; commit only suite files. Native GPU cannot be claimed without actual execution.

Critical edge cases: pre-existing output directories, Git-LFS pointers, nonfinite/missing fields, incomplete native runs, comparing unlike physical models, unknown actual y+, missing or time-shifted experimental data, and unsupported full material thermo must fail closed or report unavailable rather than produce a PASS.
