source /opt/openfoam10/etc/bashrc
set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
python3 "$O/repo/validation/steady_cuda/check_heat_failure.py" --binary "$O/steady-r2-bin/gasUGKPSteady" --output "$O/pr28-failure-regression" > "$O/pr28-failure-regression.log" 2>&1