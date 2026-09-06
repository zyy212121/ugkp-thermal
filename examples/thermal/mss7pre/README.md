# MSS7 pre-developed initial states

This directory owns the time `1` initial states used by four MSS7 CHT cases:

| Case | Gas pre-development model | Stored checkpoint |
|---|---|---|
| `MSS7_laminar` | Laminar | `checkpoints/MSS7_laminar/1` |
| `MSS7_turbulent_wallModel` | k-omega SST with wall functions | `checkpoints/MSS7_turbulent_wallModel/1` |
| `MSS7_twoPhase_sparse` | k-omega SST with wall functions | `checkpoints/MSS7_twoPhase_sparse/1` |
| `MSS7_twoPhase_dense` | k-omega SST with wall functions | `checkpoints/MSS7_twoPhase_dense/1` |

The gas field is pre-developed from `t=1` to `t=1.005 s` with the production mesh, inlet history, no-slip nozzle wall, and the selected laminar or SST model. Solid thermal coupling and particle injection are disabled during this short calculation. The temporary gas-side wall temperature is set to the 3200 K inlet temperature so the pre-development establishes a hydrodynamic boundary layer without imposing an artificial cold-wall contraction layer. Only the final internal gas fields are transferred; the production boundary conditions remain those of each target case.

The graphite field is independently reconstructed from the prescribed experimental depth-temperature profile. Each stored checkpoint also retains the target case's own zero-sequence thermal state and particle fields, so it can be restored without cross-case thermal-state hashes.

The stored checkpoints were regenerated with the current solver on 2026-09-06. The written pre-development time was `1.0050000000000439 s`. At the first axial cell station, the laminar field has 22.3141 m/s at the symmetry-side cell and 20.3654 m/s at the wall-side cell; the SST field has 22.2893 m/s and 19.9219 m/s, respectively. The former inverse near-wall profile is therefore absent. The common fluid and graphite mesh hashes are `533bb0c8a46eb13b91413b9b39db21542a8d8e3e98ec5dc53b74abc5bfebe2f3` and `0cfdc21a935451fc1e38f15b4c7ecf30ff33461d75e49479565df87367180299`. The reconstructed graphite field contains 6888 cells and spans 300.000-888.965 K.

Each target case provides a local `Allrun` and `Allclean`. `Allrun` first removes previous calculated times, restores its matching time `1` from this directory, prepares the radiation table when required, and then starts `CHT`. `Allclean` removes every numeric time directory, including time `1`; the authoritative initial state remains here and is restored by the next `Allrun`. To resume an existing calculation, run `CHT` directly instead of `Allrun`.

`build_predeveloped_checkpoints.sh` regenerates all four checkpoints with the currently built `CHT` executable. This directory intentionally has no `Allrun` or `Allclean` entry point.
