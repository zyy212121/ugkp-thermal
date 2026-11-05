set -eo pipefail
O=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles
python3 "$O/repo/validation/tests/gas_transport/steady_cpu_particles.py" --selftest "$O/cpu-particles-selftest" > "$O/cpu-particles-selftest.log" 2>&1
python3 "$O/repo/validation/steady_cpu/build.py" --particles --output "$O/cpu-reference" > "$O/cpu-build.log" 2>&1