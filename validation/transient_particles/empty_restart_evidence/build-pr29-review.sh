source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
mkdir -p "$O/review-fixed-bin" "$O/review-fixed-build"
export FOAM_USER_APPBIN=$O/review-fixed-bin GAS_UGKP_BUILD=$O/review-fixed-build
export NVCC=$P/gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$O/repo/tool/gasUGKP/build_cuda_solver.sh" > "$O/review-fixed-build.log" 2>&1
sha256sum "$O/review-fixed-bin/gasUGKP" > "$O/review-fixed-binary.sha256"