#!/usr/bin/env bash

if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <binary> <case> <label> <log-root> <report-root>" >&2
    exit 2
fi

binary="$(readlink -f "$1")"
case_dir="$(readlink -f "$2")"
label="$3"
log_root="$(readlink -m "$4")"
report_root="$(readlink -m "$5")"
log_dir="${log_root}/${label}"
report_prefix="${report_root}/${label}"
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
case "$report_prefix" in
    "$solver_root"/experiments/artifacts/*) ;;
    *) echo "report is outside laboratory artifacts: $report_prefix" >&2; exit 2 ;;
esac
[[ ! -e "$log_dir" ]] || { echo "refusing to overwrite log directory: $log_dir" >&2; exit 2; }
[[ ! -e "${report_prefix}.nsys-rep" ]] || {
    echo "refusing to overwrite report: ${report_prefix}.nsys-rep" >&2
    exit 2
}

mkdir -p "$log_dir" "$report_root"
python3 "$solver_root/devtools/cht_v2_contract.py" \
    --require-pool --require-relax --require-performance-seed --require-prepared \
    --expected-kappa fallback --case "$case_dir" \
    >"$log_dir/preflight-contract.log" 2>&1

start_time="$(awk '$1 == "startTime" { value=$2; sub(/;$/, "", value); print value }' "$case_dir/system/controlDict")"
start_dir="$case_dir/$start_time"
[[ -f "$start_dir/gpuResidentStrictParticles.dat" ]] || {
    echo "missing start particle restart: $start_dir" >&2
    exit 2
}

{
    echo "label=$label"
    echo "binary=$binary"
    echo "case=$case_dir"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "expected_steps=128"
    echo "delta_t=1e-8"
    printf '1e-8\n%.0s' {1..128} | sha256sum | awk '{print "dt_sequence_sha256=" $1}'
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

set +e
/usr/bin/time -v -o "$log_dir/profiler-runtime.txt" \
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        -o "$report_prefix" \
        "$binary" -case "$case_dir" \
        >"$log_dir/solver-and-profiler.log" 2>&1
profile_rc=$?
set -e

if [[ "$profile_rc" -eq 0 ]]; then
    nsys stats --report cuda_gpu_kern_sum --format csv \
        "${report_prefix}.nsys-rep" >"$log_dir/cuda-gpu-kern-sum.csv"
fi

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "profile_exit_code=$profile_rc"
    nvidia-smi \
        --query-gpu=timestamp,name,temperature.gpu,clocks.sm,power.draw,power.limit,memory.used,memory.total,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits | sed 's/^/gpu_after=/'
    if [[ -f "$log_dir/cuda-gpu-kern-sum.csv" ]]; then
        awk -F, '/accumulatePoissonPoolParticlesByCellKernel/ {print "pool_instances=" $3}' "$log_dir/cuda-gpu-kern-sum.csv"
        awk -F, '/relaxParticlesToResidentGasKernel/ {print "relax_instances=" $3}' "$log_dir/cuda-gpu-kern-sum.csv"
        awk -F, '/trackParticlesLocalFaceWalkKernel/ {print "track_instances=" $3}' "$log_dir/cuda-gpu-kern-sum.csv"
        if awk -F, '
            /accumulatePoissonPoolParticlesByCellKernel|relaxParticlesToResidentGasKernel|trackParticlesLocalFaceWalkKernel/ {
                if (($3 + 0) != 128) exit 1;
                seen += 1;
            }
            END { if (seen != 3) exit 1; }
        ' "$log_dir/cuda-gpu-kern-sum.csv"
        then
            echo "executed_steps=128"
            echo "executed_steps_basis=128 pool/relax/track instances in the Nsight report"
        fi
    fi
    echo "cht_exchange_lines=$(grep -c ' CHTdeltaT = ' "$log_dir/solver-and-profiler.log" || true)"
    echo "fatal_lines=$(grep -Eic 'FOAM FATAL|CUDA error|segmentation fault|(^|[^[:alpha:]])(nan|[+-]?inf(inity)?)([^[:alpha:]]|$)' "$log_dir/solver-and-profiler.log" || true)"
} >>"$log_dir/manifest.txt"

exit "$profile_rc"
