source /opt/openfoam10/etc/bashrc
set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
python3 "$O/repo/validation/transient_particles/check_empty_restart.py" --transient "$O/candidate-r2-bin/gasUGKP" --steady "$O/steady-r2-bin/gasUGKPSteady" --expect-before --output "$O/review-before-r2" > "$O/review-before-r2.log" 2>&1