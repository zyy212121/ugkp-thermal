set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/steady-pr-review
cd "$O/stacked"
python3 -m pytest validation/tests/gas_transport/test_mixture_state.py validation/tests/gas_transport/test_mixture_transport.py validation/tests/gas_transport/test_review_thermo_budget.py -q --junitxml="$O/failure-fixed-shared-thermo-tests.xml" --basetemp="$O/failure-fixed-shared-thermo-tmp" > "$O/failure-fixed-shared-thermo-tests.log" 2>&1