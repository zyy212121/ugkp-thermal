# UGKP Thermal GPU Solvers

GPU-resident UGKP solvers for OpenFOAM 10, including gas-only, finite-contact
particle heat-transfer, and conjugate heat-transfer configurations.

## Build

Load the OpenFOAM 10 environment, make CUDA available through `CUDA_HOME`,
and run:

```bash
./Allwmake
```

Reproducible examples are provided under `examples/`. Each example group
contains its own run and clean entry points.

## License

Copyright (C) 2026 Shashi Liu.

The source code in this repository is licensed under the GNU General Public
License, version 3 or, at your option, any later version
(`GPL-3.0-or-later`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
