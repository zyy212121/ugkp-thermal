source /opt/openfoam10/etc/bashrc
set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
python3 "$O/repo/validation/transient_particles/check_empty_restart.py" --transient "$O/review-fixed-bin/gasUGKP" --steady "$O/review-fixed-steady-bin/gasUGKPSteady" --output "$O/review-after-r2" > "$O/review-after-r2.log" 2>&1