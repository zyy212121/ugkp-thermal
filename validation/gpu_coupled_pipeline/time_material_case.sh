#!/usr/bin/env bash
# Durable complete-process timing. Numerical auditing is a separate command;
# its failure cannot discard an already saved solver exit code or wall time.
set -euo pipefail
if [[ $# != 5 ]]; then echo 'usage: time_material_case.sh INPUT BASELINE CANDIDATE OUT REPEATS' >&2; exit 2; fi
input=$(realpath "$1"); baseline=$(realpath "$2"); candidate=$(realpath "$3"); out=$(realpath -m "$4"); repeats=$5
[[ ! -e "$out" ]]
mkdir -p "$out"
printf 'kind\trepeat\tcase\n' > "$out/order.tsv"
printf '%s\n' "$input" > "$out/input-path.txt"
sha256sum "$baseline" "$candidate" > "$out/binaries.sha256"
printf '%s\n' 'GNU time elapsed wall seconds (0.01 s resolution), then process exit status; process startup and normal output included, input copying and audits excluded.' > "$out/timing-method.txt"
export CHMT_CUDA_STACK_BYTES=65536 OMP_NUM_THREADS=4
for ((repeat=0;repeat<=repeats;repeat++)); do
    if ((repeat%2==0)); then order=(baseline candidate); else order=(candidate baseline); fi
    for kind in "${order[@]}"; do
        name="$kind-$repeat"; case_dir="$out/$name"
        mkdir "$case_dir"
        cp -a "$input/0" "$input/constant" "$input/system" "$case_dir/"
        printf '%s\t%s\t%s\n' "$kind" "$repeat" "$name" >> "$out/order.tsv"
        binary=$baseline; if [[ $kind == candidate ]]; then binary=$candidate; fi
        # A solver failure stops the cohort. There are no automatic reruns.
        /usr/bin/time -f '%e\n%x' -o "$case_dir/timing.txt" "$binary" -case "$case_dir" > "$case_dir/run.log" 2>&1
        printf '%s ' "$name"; head -1 "$case_dir/timing.txt"
    done
done
printf '%s\n' complete > "$out/solver-runs-complete"
