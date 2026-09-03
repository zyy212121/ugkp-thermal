# MSS7 two-phase post-processing

Run `./draw.py` from the case directory after at least one result time has been written. The command reads written fields and particle checkpoints without modifying the CHT solution.

The reusable Bartz implementation is `tools/postprocessing/bartz.py`. Case geometry and gas-property inputs are in `bartz.json`. The Bartz curve uses the inlet chamber-pressure schedule, the nozzle area ratio, and the gas properties used by the case. It is a comparison curve only and is not coupled back into CHT.

Outputs are written below `examples/thermal/results/MSS7_twoPhase_dense`:

- `data/temperature_comparison.csv` and `figures/temperature_comparison.png`
- `data/wall_heat_flux_profiles` and `figures/wall_heat_flux_profiles`
- `data/effective_radiating_area_profiles` and `figures/effective_radiating_area_profiles`

The temperature comparison uses the written two-phase time range and interpolates the pure-gas reference onto the same times. The radiating-area ordinate is the exposed face-area fraction `A_rad/A_f`.
