# Dusty-wave consistency case

This case verifies the coupled gas-particle acoustic response against the linear dusty-wave solution. It uses a 4 m one-dimensional domain with 400 cells, 400,000 weighted parcels, a particle response time of 0.2 s, and an analysis window of 1--3 m.

The production solver intentionally does not include the validation-only constant-response-time drag law. `./Allrun` stages the current `gasUGKP` sources, applies the adapter under `assets/solver_source`, builds private frontend and backend executables, runs the case, validates the result, and removes the temporary build products. The production source tree is not modified.

Run this case from a loaded OpenFOAM environment with `./Allrun`. Use `./Allclean` to restore the minimal case. The retained lightweight comparison data, metrics, and figure are written to `../result/dusty_wave.csv`, `../result/dusty_wave_metrics.json`, and `../result/dusty_wave.png`.
