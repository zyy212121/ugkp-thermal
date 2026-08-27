# Wind-Sand Shock Tube

This case uses a case-local solver adapter because its prescribed particle response time and two-dimensional inlet fluctuation are specific to this validation problem. The production `gasUGKP` source remains the canonical solver source.

## Private assets

- `assets/initialization/prepare_axial_particles.py` prepares and validates the axial particle initial condition.
- `assets/solver_source/GpuResidentStrict.constant-response-time.patch` adds the constant particle-response-time behavior required by this case.
- `assets/solver_source/GpuResidentStrict.axial-fluctuation.patch` restricts the inlet fluctuation to the physical two-dimensional plane.
- `assets/solver_source/private_backend/` contains the case-specific drag implementation.
- `assets/build/build_validation_solver.sh` stages the canonical `gasUGKP` source, applies the case adapters, and builds `windSandUGKP` plus `windSandUGKPCudaBackend` inside the case directory.
- `assets/build/clean_validation_solver.sh` removes the temporary build tree and case-local executables.

## Run

Load the OpenFOAM environment, enter this directory, and run:

```bash
./Allrun
```

`Allrun` performs the following operations in order:

1. validates the particle initial condition;
2. compiles the case-local frontend and CUDA backend;
3. creates the mesh when it is absent;
4. runs the validation solver;
5. removes the temporary build tree and case-local executables when the run exits.

The parent consistency runner also invokes this entry point automatically:

```bash
cd ..
./Allrun windSandShockTube
```

## Build only

To verify or inspect the private solver without running the case:

```bash
./assets/build/build_validation_solver.sh
```

The executables are created in `.validation-bin/`. Remove all generated private-build files with:

```bash
./assets/build/clean_validation_solver.sh
```
