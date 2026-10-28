# Source this file to use the applications built inside this checkout.
_ugkp_source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${WM_PROJECT_DIR:-} ]]; then
    source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
fi
export FOAM_USER_APPBIN="$_ugkp_source_root/bin"
export FOAM_USER_LIBBIN="$_ugkp_source_root/lib"
export UGKP_CASES_ROOT="$_ugkp_source_root/../case"
export FSH_CUDA_BACKEND="$FOAM_USER_APPBIN/FSHCudaBackend"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$FOAM_USER_APPBIN:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset _ugkp_source_root
