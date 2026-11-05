set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
bash "$P/steady-pr-review/stacked/validation/steady_cuda/check_heat_domain.sh" "$P/pr28-heat-failure-controls" > "$P/pr28-heat-failure-controls.log" 2>&1