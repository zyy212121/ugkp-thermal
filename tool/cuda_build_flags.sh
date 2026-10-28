# Shared CUDA compilation policy. Requires a toolkit supporting --Ofast-compile.
# The minimal level curbs expensive compiler passes while retaining the selected
# optimization level and all precision/FMA options. No automatic retries.
ugkp_cuda_flags=()
case "${UGKWP_CUDA_FAST_COMPILE:-min}" in
    0|min|mid|max)
        ugkp_cuda_flags+=(--Ofast-compile="${UGKWP_CUDA_FAST_COMPILE:-min}")
        echo "[cuda-build] fast-compile=${UGKWP_CUDA_FAST_COMPILE:-min}; optimization level and FP flags preserved"
        ;;
    *) echo 'ERROR: UGKWP_CUDA_FAST_COMPILE must be 0, min, mid or max' >&2; exit 2 ;;
esac
