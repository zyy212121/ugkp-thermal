source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
mkdir -p "$O/review-fixed-steady-bin" "$O/review-fixed-steady-build"
export FOAM_USER_APPBIN=$O/review-fixed-steady-bin GAS_UGKP_STEADY_BUILD=$O/review-fixed-steady-build
export NVCC=$P/gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$O/repo/tool/gasUGKPSteady/build_cuda_solver.sh" > "$O/review-fixed-steady-build.log" 2>&1
sha256sum "$O/review-fixed-steady-bin/gasUGKPSteady" > "$O/review-fixed-steady-binary.sha256"