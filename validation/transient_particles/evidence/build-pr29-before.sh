source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
mkdir -p "$O/before-bin" "$O/before-build"
export FOAM_USER_APPBIN=$O/before-bin GAS_UGKP_BUILD=$O/before-build
export NVCC=$P/gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$P/steady-pr-review/stacked/tool/gasUGKP/build_cuda_solver.sh" > "$O/before-build.log" 2>&1
sha256sum "$O/before-bin/gasUGKP" > "$O/before-binary.sha256"