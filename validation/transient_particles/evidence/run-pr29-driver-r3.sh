source /opt/openfoam10/etc/bashrc
set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
python3 "$O/repo/validation/transient_particles/check_driver.py" --candidate "$O/candidate-r2-bin/gasUGKP" --before "$O/before-bin/gasUGKP" --output "$O/driver-r3" > "$O/driver-r3.log" 2>&1