set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
bash "$O/repo/validation/steady_cuda/check_heat_domain.sh" "$O/heat-controls" > "$O/heat-controls.log" 2>&1