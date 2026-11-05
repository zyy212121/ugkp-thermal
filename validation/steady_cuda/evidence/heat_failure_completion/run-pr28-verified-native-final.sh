source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
python3 "$P/steady-pr-review/stacked/validation/steady_cuda/check_heat_failure.py" --binary "$P/steady-pr-review/fixed-bin/gasUGKPSteady" --output "$P/pr28-failure-validation-before-final" --expect-before
python3 "$P/steady-pr-review/stacked/validation/steady_cuda/check_heat_failure.py" --binary "$P/steady-pr-review/failure-fixed-r3-bin/gasUGKPSteady" --output "$P/pr28-failure-validation-after-final"