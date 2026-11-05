source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
mkdir -p "$O/steady-r2-bin" "$O/steady-r2-build"
export FOAM_USER_APPBIN=$O/steady-r2-bin GAS_UGKP_STEADY_BUILD=$O/steady-r2-build
export NVCC=$P/gpu-coupled-validation/candidate-layer5-gas-r1/nvcc-fast UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2 CUDA_HOME=/usr/local/cuda
/usr/bin/time -v bash "$O/repo/tool/gasUGKPSteady/build_cuda_solver.sh" > "$O/steady-r2-build.log" 2>&1
sha256sum "$O/steady-r2-bin/gasUGKPSteady" > "$O/steady-r2-binary.sha256"