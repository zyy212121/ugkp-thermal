#!/usr/bin/env bash

if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 <binary> <case> <label> <log-root> [--performance]" >&2
    exit 2
fi

binary="$(readlink -f "$1")"
case_dir="$(readlink -f "$2")"
label="$3"
log_root="$(readlink -m "$4")"
contract_mode="${5:-}"
log_dir="${log_root}/${label}"
solver_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -x "$binary" ]] || { echo "missing executable: $binary" >&2; exit 2; }
[[ -d "$case_dir" ]] || { echo "missing case: $case_dir" >&2; exit 2; }
case "$case_dir" in
    "$solver_root"/experiments/runs/*) ;;
    *) echo "case is outside laboratory runs: $case_dir" >&2; exit 2 ;;
esac
case "$log_dir" in
    "$solver_root"/experiments/logs/*) ;;
    *) echo "log directory is outside laboratory logs: $log_dir" >&2; exit 2 ;;
esac
[[ ! -e "$log_dir" ]] || { echo "refusing to overwrite log directory: $log_dir" >&2; exit 2; }

mkdir -p "$log_dir"

start_from="$(awk '$1 == "startFrom" { value=$2; sub(/;$/, "", value); print value }' "$case_dir/system/controlDict")"
if [[ "$start_from" == "latestTime" ]]; then
    start_time="$(find "$case_dir" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$' | sort -g | tail -1)"
else
    start_time="$(awk '$1 == "startTime" { value=$2; sub(/;$/, "", value); print value }' "$case_dir/system/controlDict")"
fi
[[ -n "$start_time" ]] || { echo "cannot parse startTime" >&2; exit 2; }
start_dir="$case_dir/$start_time"
[[ -d "$start_dir" ]] || { echo "missing start-time directory: $start_dir" >&2; exit 2; }

contract_args=(--require-pool --require-relax --case "$case_dir")
if [[ "$contract_mode" == "--performance" ]]; then
    contract_args+=(--require-performance-seed --require-prepared --expected-kappa fallback)
elif [[ -n "$contract_mode" ]]; then
    echo "unknown contract mode: $contract_mode" >&2
    exit 2
fi
python3 "$solver_root/devtools/cht_v2_contract.py" \
    "${contract_args[@]}" >"$log_dir/preflight-contract.log" 2>&1

{
    echo "label=$label"
    echo "binary=$binary"
    echo "case=$case_dir"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$contract_mode" == "--performance" ]]; then
        echo "expected_steps=128"
        echo "delta_t=1e-8"
        printf '1e-8\n%.0s' {1..128} | sha256sum | awk '{print "dt_sequence_sha256=" $1}'
    fi
    sha256sum "$binary"
    sha256sum "$solver_root/gpu/GpuResidentStrict.cu"
    sha256sum "$case_dir/system/controlDict"
    sha256sum "$case_dir/constant/solidThermalCouplingProperties"
    sha256sum "$start_dir/thermalExchangeState"
    sha256sum "$start_dir/gpuResidentStrictParticles.dat"
    nvidia-smi \
        --query-gpu=timestamp,name,temperature.gpu,clocks.sm,power.draw,power.limit,memory.used,memory.total,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits | sed 's/^/gpu_before=/'
} >"$log_dir/manifest.txt"

nvidia-smi \
    --query-gpu=timestamp,index,name,temperature.gpu,clocks.sm,power.draw,power.limit,memory.used,memory.total,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits \
    --loop-ms=200 >"$log_dir/gpu-telemetry.csv" 2>"$log_dir/gpu-telemetry.stderr" &
telemetry_pid=$!

set +e
set -o pipefail
/usr/bin/time -v -o "$log_dir/runtime.txt" \
    "$binary" -case "$case_dir" 2>&1 | tee "$log_dir/solver.log"
solver_rc=${PIPESTATUS[0]}
set -e

kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "solver_exit_code=$solver_rc"
    echo "simulation_step_lines=$(grep -c 'simulationTime =' "$log_dir/solver.log" || true)"
    echo "cht_exchange_lines=$(grep -c ' CHTdeltaT = ' "$log_dir/solver.log" || true)"
    echo "fatal_lines=$(grep -Eic 'FOAM FATAL|CUDA error|segmentation fault|(^|[^[:alpha:]])(nan|[+-]?inf(inity)?)([^[:alpha:]]|$)' "$log_dir/solver.log" || true)"
    if [[ "$contract_mode" == "--performance" && "$solver_rc" -eq 0 ]]; then
        echo "executed_steps=128"
        echo "executed_steps_basis=fixed 128-step schedule and solver exit 0"
    fi
} >>"$log_dir/manifest.txt"

exit "$solver_rc"
