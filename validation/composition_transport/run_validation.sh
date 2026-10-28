#!/usr/bin/env bash
# Fresh output directory; every executable comes from this source checkout.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?Usage: bash validation/composition_transport/run_validation.sh /absolute/fresh/output}
mkdir "$out" || exit 1
out=$(cd "$out" && pwd)
bash "$root/tool/build_local.sh" > "$out/build.log" 2>&1 || exit $?
source "$root/env.sh"
set -euo pipefail
cd "$root"
python3 -m pytest -q validation/composition_transport/test_configuration.py validation/composition_transport/test_steady_transport.py validation/tests/gas_transport/test_molecular_transport_config.py validation/tests/gas_transport/test_molecular_viscosity_cuda.py validation/tests/gas_transport/test_steady_gas_residual.py validation/tests/gas_transport/test_mixture_transport.py > "$out/unit.log" 2>&1
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -std=c++17 --expt-relaxed-constexpr -O2 --fmad=false -arch=sm_89 -Icommon validation/composition_transport/local_transport.cu -o "$out/local-transport" > "$out/probe-build.log" 2>&1
"$out/local-transport" validation/composition_transport/transport.dict > "$out/correlation-rows.txt"
python3 validation/composition_transport/verify_correlations.py "$out/correlation-rows.txt" > "$out/correlations.json"
python3 validation/composition_transport/prepare_native.py "$out/native"
python3 validation/composition_transport/run_native.py "$out/native"
python3 validation/composition_transport/verify_native.py "$out/native" > "$out/native-verification.log"
python3 validation/composition_transport/check_conservation.py "$out/conservation"
python3 validation/composition_transport/check_other_apps.py --output "$out/other-apps"
python3 validation/build/check_local.py --bin-dir "$root/bin" --output "$out/application-regression"
sha256sum "$root/bin/"* "$out/local-transport" > "$out/binaries.sha256"
