
ugkp_cuda_stack_soft=$(ulimit -S -s)
ugkp_cuda_stack_hard=$(ulimit -H -s)
if [[ "$ugkp_cuda_stack_soft" != unlimited && "$ugkp_cuda_stack_soft" -lt 262144 ]]; then
    ugkp_cuda_stack_target=262144
    if [[ "$ugkp_cuda_stack_hard" != unlimited && "$ugkp_cuda_stack_hard" -lt 262144 ]]; then
        ugkp_cuda_stack_target=$ugkp_cuda_stack_hard
    fi
    ulimit -S -s "$ugkp_cuda_stack_target"
fi
unset ugkp_cuda_stack_soft ugkp_cuda_stack_hard ugkp_cuda_stack_target







ugkp_cuda_stack_soft=$(ulimit -S -s)
ugkp_cuda_stack_hard=$(ulimit -H -s)
if [[ "$ugkp_cuda_stack_soft" != unlimited && "$ugkp_cuda_stack_soft" -lt 65536 ]]; then
    ugkp_cuda_stack_target=65536
    if [[ "$ugkp_cuda_stack_hard" != unlimited && "$ugkp_cuda_stack_hard" -lt 65536 ]]; then
        ugkp_cuda_stack_target=$ugkp_cuda_stack_hard
    fi
    ulimit -S -s "$ugkp_cuda_stack_target"
fi
unset ugkp_cuda_stack_soft ugkp_cuda_stack_hard ugkp_cuda_stack_target
ugkp_cuda_flags=()
case "${UGKWP_CUDA_FAST_COMPILE:-min}" in
    0|min|mid|max)
        ugkp_cuda_flags+=(--Ofast-compile="${UGKWP_CUDA_FAST_COMPILE:-min}")
        echo "[cuda-build] fast-compile=${UGKWP_CUDA_FAST_COMPILE:-min}; optimization level and FP flags preserved"
        ;;
    *) echo 'ERROR: UGKWP_CUDA_FAST_COMPILE must be 0, min, mid or max' >&2; exit 2 ;;
esac





ugkp_nvcc() {
    local compiler="$1"; shift
    local cache="${UGKP_CUDA_CACHE:-}"
    if [[ -z "$cache" ]] && command -v ccache >/dev/null 2>&1; then cache=ccache; fi
    if [[ "$cache" == 0 || -z "$cache" ]]; then
        "$compiler" "$@"
    else
        command -v "$cache" >/dev/null || { echo "ERROR: CUDA cache command not found: $cache" >&2; return 2; }
        CCACHE_COMPILERCHECK=content CCACHE_SLOPPINESS= CCACHE_NODIRECT=true \
            "$cache" "$compiler" "$@"
    fi
}
