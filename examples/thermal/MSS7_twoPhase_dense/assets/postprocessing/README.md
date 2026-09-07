# MSS7 two-phase post-processing

Run `./draw.py` from the case directory after at least one result time has been written. The command reads written fields and particle checkpoints without modifying the CHT solution.

The reusable Bartz implementation is `tools/postprocessing/bartz.py`. Case geometry and gas-property inputs are in `bartz.json`. The Bartz curve uses the inlet chamber-pressure schedule, the nozzle area ratio, and the gas properties used by the case. It is a comparison curve only and is not coupled back into CHT.

Outputs are written below `examples/thermal/results/MSS7_twoPhase_dense`:

- `data/temperature_comparison.csv` and `figures/temperature_comparison.png`
- `data/wall_heat_flux_profiles`: convection, radiation, reflection, deposition, their calculated total, Bartz, and the Yang et al. (2023) equivalent two-phase convection estimate
- `figures/wall_heat_flux_profiles`: one four-component heat-flux figure for every completed radiation-coupling time
- `figures/total_wall_heat_flux_comparison_profiles`: one calculated-total/Bartz/Yang-et-al. comparison for every completed radiation-coupling time
- `data/effective_radiating_area_profiles` and `figures/effective_radiating_area_profiles`

The temperature comparison uses the written two-phase time range and interpolates the pure-gas reference onto the same times. The radiating-area ordinate is the exposed face-area fraction `A_rad/A_f`.

The third total-heat-flux curve implements Eq. (28) from Liu Yang, Dong Zhichao et al., *International Communications in Heat and Mass Transfer* 145 (2023) 106845, DOI `10.1016/j.icheatmasstransfer.2023.106845`. The fitted constants are stored in `bartz.json`. The published condensed-product mass fraction `0.34` and burning-face diameter `0.18 m` are retained. Local Reynolds number and gas temperature use the same isentropic state used by the Bartz comparison, while the local geometric wall angle supplies the scour angle. Rows outside the paper's calibrated burning-area ratio range `5.063-36.0` are marked in the CSV because this nozzle application is an extrapolation of the combustor-scour correlation.
