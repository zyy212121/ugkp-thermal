source /opt/openfoam10/etc/bashrc
set -eo pipefail
P=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026
O=$P/transient-mixture-particles
python3 "$O/repo/validation/transient_particles/check_native.py" --candidate "$O/candidate-r2-bin/gasUGKP" --before "$O/before-bin/gasUGKP" --steady-before "$P/steady-pr-review/failure-fixed-r3-bin/gasUGKPSteady" --steady-after "$O/steady-r2-bin/gasUGKPSteady" --output "$O/native-r3" > "$O/native-r3.log" 2>&1