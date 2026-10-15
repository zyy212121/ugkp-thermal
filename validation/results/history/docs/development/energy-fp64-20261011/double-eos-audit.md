# Existing FP64 legacy EOS identity repair

The authoritative-energy audit found a preexisting homogeneous FP64 counterexample:

- rho = 2, rhoUx = 0.00048828125, rhoUy = rhoUz = 0
- rhoE = nextafter(1e9, 2e9) = 1000000000.0000001
- Unconditional `kinetic + (rhoE - kinetic)` writes 1000000000 instead, losing one binary64 ULP despite a valid state.

The focused test first failed on this exact production recovery path. Recovery now writes double rhoE only if its existing internal-energy validity/floor test requires repair. Its density, momentum, pressure, temperature, and lower-bound calculation order is retained. The legacy homogeneous float path is unchanged. A second test confirms negative/nonfinite energy still takes the old floor repair and produces the expected 300 K temperature and unchanged valid momentum.

This is a behavior fix to an existing FP64 path; it is not evidence that CHMT or gasUGKP formerly stored float energy. No tolerance or mathematical model changed.

## Frozen-oracle check

Before changing any frozen oracle, an isolated header overlay restored only the old unconditional EOS rhoE assignment in the current code. Its 324 scheme/reconstruction/limiter/turbulence configurations reproduced the original FP64 aggregate hashes exactly. Each of the 123 hashed array fields was also emitted individually (all eight fixture entries), and the restored-assignment overlay matched all values from exact base ab900f54.

The conditional-repair version was then compared with that overlay, field by field. There were zero changed field values in these existing 324 configurations, so the frozen FP64 oracle does not need modification. The new nextafter fixture exercises the missing input class. The legacy FP32 oracle is unchanged too.

Verification commands:

- `python -m pytest -p no:cacheprovider tests/gas_transport/test_double_energy_recovery.py tests/gas_transport/test_gas_state_view.py -q`
- The per-field overlay check compiles the existing gas_state_probe.cpp with g++ C++17/O0/Real64, emits each existing hashBytes input using hexadecimal floating representation, and compares base, restored-assignment overlay, and conditional-repair output. The probe body and field set are unchanged; only diagnostic emissions are added to the disposable probe.

## Valid momentum recovery repair

A subsequent repeated-recovery test also exposed the existing double momentum round trip: rho=1.1 and rhoUx=19999000.875 produce rho*(rhoUx/rho)=19999000.875000004. With kinetic energy plus 1e6 internal energy, valid recovery must retain the original conserved rho, all momenta, and rhoE bits. The focused repair retains valid double momenta, while preserving the old density-floor and invalid-velocity replacement. It does not change the original pressure/temperature calculation order or add energy to compensate for momentum movement. Repeated recovery checks the bits and temperature; an additional test checks true density, momentum, and internal-energy repair. The homogeneous float branch remains unchanged.

Before updating the FP64 oracle, an isolated current-header overlay restored only the old unconditional homogeneous EOS momentum and energy assignments. It reproduced all 324 original aggregate hashes and all 123 array fields (eight entries each) from ab900f54 exactly. The fixed comparison changed only 30 rhoUy values, with maximum absolute difference 3.469446951953614e-18. No rho, rhoE, pressure, temperature, or other field changed in these fixtures. The FP64 oracle was updated only after that exact overlay and per-field proof; FP32 golden output is unchanged. This is a narrow existing-FP64 behavior fix and cannot be described as bitwise preservation of every old double output.
