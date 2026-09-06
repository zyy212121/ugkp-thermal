#!/usr/bin/env bash

set -euo pipefail

[[ -n ${WM_PROJECT_DIR:-} ]] || { echo "OpenFOAM environment is not loaded" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
thermal_root=$(realpath -e -- "${root}/..")
work=$(mktemp -d /tmp/mss7pre-build.XXXXXX)
trap 'rm -rf -- "${work}"' EXIT

prepare_run()
{
    local case_name=$1
    local run_dir="${work}/${case_name}"
    cp -a -- "${thermal_root}/${case_name}/constant" "${thermal_root}/${case_name}/system" "${thermal_root}/${case_name}/tools" "${root}/checkpoints/${case_name}/1" "${run_dir}/"
    cp -a -- "${thermal_root}/${case_name}/data/uniform_gas_start/." "${run_dir}/1/fluid/"
    python3 - "${run_dir}" <<'PY'
from pathlib import Path
import re
import sys
case = Path(sys.argv[1])
p = case / "1/fluid/T"
s = p.read_text()
s, n = re.subn(r"(fluid_to_graphite\s*\{.*?\bvalue\s+uniform\s+)[^;]+(;)", r"\g<1>3200\2", s, flags=re.S)
if n != 1:
    raise RuntimeError("cannot set the temporary isothermal gas wall")
p.write_text(s)
p = case / "constant/particleProperties"
s = p.read_text()
s, n = re.subn(r"(gpuResidentSolidThermalCoupling\s+)(true|false)(\s*;)", r"\g<1>false\3", s)
if n != 1:
    raise RuntimeError("cannot disable temporary solid coupling")
p.write_text(s)
p = case / "system/controlDict"
s = p.read_text()
for key, value in (("startFrom", "startTime"), ("startTime", "1"), ("stopAt", "endTime"), ("endTime", "1.005"), ("writeControl", "adjustableRunTime"), ("writeInterval", "0.005")):
    s, n = re.subn(rf"(?m)^\s*{key}\s+[^;]+;", f"{key} {value};", s)
    if n != 1:
        raise RuntimeError(f"cannot set {key}")
p.write_text(s)
PY
    rm -f -- "${run_dir}/1/thermalExchangeManifest" "${run_dir}/1/thermalExchangeState"
    (
        cd "${run_dir}"
        CHT > predevelop.log 2>&1
    )
}

mkdir -p -- "${work}/MSS7_laminar" "${work}/MSS7_turbulent_wallModel"
prepare_run MSS7_laminar
prepare_run MSS7_turbulent_wallModel

(
    cd "${work}/MSS7_laminar"
    postProcess -func writeCellCentres -time 1 -region graphite > graphite-centres.log 2>&1
    python3 tools/initialise_graphite_temperature.py
    rm -f -- 1/graphite/C 1/graphite/Cx 1/graphite/Cy 1/graphite/Cz
)

python3 - "${thermal_root}" "${root}" "${work}" <<'PY'
from pathlib import Path
import re
import sys

thermal_root = Path(sys.argv[1])
root = Path(sys.argv[2])
work = Path(sys.argv[3])
pattern = re.compile(r"internalField\s+(?:uniform\s+[^;]+|nonuniform\s+List<[^>]+>\s+\d+\s*\(.*?\))\s*;", re.S)

def latest_fluid(case_name):
    times = []
    for path in (work / case_name).iterdir():
        try:
            times.append((float(path.name), path))
        except ValueError:
            pass
    return max(times)[1] / "fluid"

def merge_internal(destination, source):
    target_text = destination.read_text()
    source_match = pattern.search(source.read_text())
    if source_match is None:
        raise RuntimeError(f"missing source internalField in {source}")
    output, count = pattern.subn(source_match.group(0), target_text, count=1)
    if count != 1:
        raise RuntimeError(f"missing destination internalField in {destination}")
    destination.write_text(output)

sources = {
    "MSS7_laminar": latest_fluid("MSS7_laminar"),
    "MSS7_turbulent_wallModel": latest_fluid("MSS7_turbulent_wallModel"),
    "MSS7_twoPhase_sparse": latest_fluid("MSS7_turbulent_wallModel"),
    "MSS7_twoPhase_dense": latest_fluid("MSS7_turbulent_wallModel"),
}
for case_name, source in sources.items():
    checkpoint = root / "checkpoints" / case_name / "1"
    for field in ("U", "T", "p", "rho", "rhoU", "rhoE", "k", "omega", "nut"):
        merge_internal(checkpoint / "fluid" / field, source / field)
    merge_internal(checkpoint / "graphite/T", work / "MSS7_laminar/1/graphite/T")
    print(f"updated={checkpoint}")
PY

python3 "${root}/refresh_initial_states.py"
