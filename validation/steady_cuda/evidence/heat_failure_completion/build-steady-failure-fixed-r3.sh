source /opt/openfoam10/etc/bashrc
set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/steady-pr-review
mkdir -p "$O/failure-fixed-r3-bin" "$O/failure-fixed-r3-build"
export FOAM_USER_APPBIN=$O/failure-fixed-r3-bin GAS_UGKP_STEADY_BUILD=$O/failure-fixed-r3-build NVCC=$O/../gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$O/stacked/tool/gasUGKPSteady/build_cuda_solver.sh" > "$O/failure-fixed-r3-build.log" 2>&1
sha256sum "$O/failure-fixed-r3-bin/gasUGKPSteady" > "$O/failure-failure-fixed-r3-binary.sha256"