source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
mkdir -p "$O/candidate-r2-bin" "$O/candidate-r2-build"
export FOAM_USER_APPBIN=$O/candidate-r2-bin GAS_UGKP_BUILD=$O/candidate-r2-build
export NVCC=$P/gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$O/repo/tool/gasUGKP/build_cuda_solver.sh" > "$O/candidate-r2-build.log" 2>&1
sha256sum "$O/candidate-r2-bin/gasUGKP" > "$O/candidate-r2-binary.sha256"