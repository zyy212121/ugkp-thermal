#!/usr/bin/env bash
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -eo pipefail

case_dir=${1:?usage: initialise_uniform_gas_3200.sh CASE_DIR}

p0=2700000
T0=3200
rho0=2.939895470383275
rhoE0=13500000
k0=1.52222988431
omega0=247.535848046
nut0=0.006149533073

fd()
{
    foamDictionary "$case_dir/$1" -entry "$2" -set "$3"
}

drop_dotted()
{
    foamDictionary "$case_dir/$1" -entry "$2" -remove >/dev/null 2>&1 || true
}

for spec in \
    '1/fluid/T boundaryField.inlet.value' \
    '1/fluid/Tp boundaryField.inlet.value' \
    '1/fluid/U boundaryField.outlet.value' \
    '1/fluid/p boundaryField.inlet.value' \
    '1/fluid/p boundaryField.outlet.value' \
    '1/fluid/rho boundaryField.inlet.value' \
    '1/fluid/rhoE boundaryField.inlet.value' \
    '1/fluid/rhoE boundaryField.outlet.value' \
    '1/fluid/rhoE boundaryField.fluid_to_graphite.value' \
    '1/fluid/rhoU boundaryField.inlet.value' \
    '1/fluid/rhoU boundaryField.outlet.value' \
    '1/fluid/rhoU boundaryField.fluid_to_graphite.value' \
    '1/fluid/k boundaryField.inlet.value' \
    '1/fluid/k boundaryField.outlet.value' \
    '1/fluid/k boundaryField.fluid_to_graphite.value' \
    '1/fluid/omega boundaryField.inlet.value' \
    '1/fluid/omega boundaryField.outlet.value' \
    '1/fluid/omega boundaryField.fluid_to_graphite.value'
do
    drop_dotted ${spec}
done

fd constant/ugkwpProperties gpuResidentInletTemperature "$T0"
fd constant/ugkwpProperties gpuResidentInjectionTp "$T0"

fd 1/fluid/T internalField "uniform $T0"
fd 1/fluid/T boundaryField/inlet/value "uniform $T0"

fd 1/fluid/Tp internalField "uniform $T0"
fd 1/fluid/Tp boundaryField/inlet/value "uniform $T0"

fd 1/fluid/U internalField "uniform (0 0 0)"
fd 1/fluid/U boundaryField/outlet/value "uniform (0 0 0)"

fd 1/fluid/p internalField "uniform $p0"
fd 1/fluid/p boundaryField/inlet/value "uniform $p0"
fd 1/fluid/p boundaryField/outlet/value "uniform $p0"

fd 1/fluid/rho internalField "uniform $rho0"
fd 1/fluid/rho boundaryField/inlet/value "uniform $rho0"

fd 1/fluid/rhoE internalField "uniform $rhoE0"
fd 1/fluid/rhoE boundaryField/inlet/value "uniform $rhoE0"
fd 1/fluid/rhoE boundaryField/outlet/value "uniform $rhoE0"
fd 1/fluid/rhoE boundaryField/fluid_to_graphite/value "uniform $rhoE0"

fd 1/fluid/rhoU internalField "uniform (0 0 0)"
fd 1/fluid/rhoU boundaryField/inlet/value "uniform (0 0 0)"
fd 1/fluid/rhoU boundaryField/outlet/value "uniform (0 0 0)"
fd 1/fluid/rhoU boundaryField/fluid_to_graphite/value "uniform (0 0 0)"

fd 1/fluid/k internalField "uniform $k0"
fd 1/fluid/k boundaryField/inlet/value "uniform $k0"
fd 1/fluid/k boundaryField/outlet/value "uniform $k0"
fd 1/fluid/k boundaryField/fluid_to_graphite/value "uniform $k0"

fd 1/fluid/omega internalField "uniform $omega0"
fd 1/fluid/omega boundaryField/inlet/value "uniform $omega0"
fd 1/fluid/omega boundaryField/outlet/value "uniform $omega0"
fd 1/fluid/omega boundaryField/fluid_to_graphite/value "uniform $omega0"

fd 1/fluid/nut internalField "uniform $nut0"

for field in dMeanCell epsilonS rhoDs rhoEs rhoHp theta; do
    fd "1/fluid/$field" internalField "uniform 0"
done

for field in Us rhoUs; do
    fd "1/fluid/$field" internalField "uniform (0 0 0)"
done

echo "Initialized $case_dir at t=1: p=$p0 Pa T=$T0 K U=(0 0 0) rho=$rho0 rhoE=$rhoE0"
