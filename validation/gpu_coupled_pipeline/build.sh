#!/usr/bin/env bash
# Reproducible isolated native builds; source OpenFOAM before calling.
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "usage: $0 SOURCE_ROOT OUTPUT_DIRECTORY [CHMT CHT]" >&2; exit 2; fi
source_root=$(realpath "$1")
out=$(realpath -m "$2")
shift 2
solvers=("$@")
if [[ ${#solvers[@]} -eq 0 ]]; then solvers=(CHMT CHT); fi
mkdir -p "$out/bin" "$out/build"
export FOAM_USER_APPBIN="$out/bin" CHMT_BUILD="$out/build/chmt"
export CHMT_CUDA_ARCH=sm_89 UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Set GPU_COUPLED_BUILD_MODE=standard to retain the repository optimization.
# Fast mode is confined to validation builds on memory-constrained hosts.
mode=${GPU_COUPLED_BUILD_MODE:-fast}
fast_level=${GPU_COUPLED_FAST_LEVEL:-max}
# The separate material TU can retain production optimization with fast gas builds.
material_mode=${GPU_COUPLED_MATERIAL_BUILD_MODE:-$mode}
thermal_mode=${GPU_COUPLED_THERMAL_BUILD_MODE:-standard}
case "$thermal_mode" in fast|standard) ;; *) echo "GPU_COUPLED_THERMAL_BUILD_MODE must be fast or standard" >&2; exit 2 ;; esac
if [[ "$mode" == standard && "$thermal_mode" != standard ]]; then echo "standard builds require standard thermal mode" >&2; exit 2; fi
case "$material_mode" in fast|standard) ;; *) echo "GPU_COUPLED_MATERIAL_BUILD_MODE must be fast or standard" >&2; exit 2 ;; esac
if [[ "$mode" == standard && "$material_mode" != standard ]]; then echo "standard builds require standard material mode" >&2; exit 2; fi
case "$mode" in
 fast)
  case "$fast_level" in min|mid|max) ;; *) echo "GPU_COUPLED_FAST_LEVEL must be min, mid or max" >&2; exit 2 ;; esac
  cat > "$out/nvcc-fast" <<NVCC
#!/usr/bin/env bash
if [[ "$material_mode" == standard ]]; then
 for source_file in "\$@"; do
  case "\$source_file" in */gpu/MaterialBackend.cu) exec "$CUDA_HOME/bin/nvcc" "\$@" ;; esac
 done
fi
if [[ "$thermal_mode" == standard ]]; then
 for source_file in "\$@"; do
  case "\$source_file" in */gpu/SparseConjugateGradient.cu|*/gpu/SolidThermalProperties.cu|*/gpu/ImplicitDiffusion.cu|*/gpu/ThermalFieldOperators.cu|*/gpu/MaterialThermalBackend.cu) exec "$CUDA_HOME/bin/nvcc" "\$@" ;; esac
 done
fi
exec "$CUDA_HOME/bin/nvcc" --Ofast-compile=$fast_level -Xcompiler=-fno-tree-fre "\$@"
NVCC
  chmod +x "$out/nvcc-fast"
  export NVCC="$out/nvcc-fast"
  ;;
 standard) export NVCC="$CUDA_HOME/bin/nvcc" ;;
 *) echo "GPU_COUPLED_BUILD_MODE must be fast or standard" >&2; exit 2 ;;
esac
python3 - "$source_root" "$out" "$mode" "$fast_level" "$material_mode" "$thermal_mode" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);mode=sys.argv[3];level=sys.argv[4];material_mode=sys.argv[5];thermal_mode=sys.argv[6]
def command(*args):return subprocess.check_output(args,cwd=root,text=True).strip()
tracked=command('git','ls-files','--cached','--others','--exclude-standard').splitlines()
h=hashlib.sha256()
for name in sorted(tracked):
    path=root/name
    if path.is_file():h.update(name.encode()+b'\0'+path.read_bytes()+b'\0')
metadata=dict(source_commit=command('git','rev-parse','HEAD'),source_status=command('git','status','--porcelain'),
    source_sha256=h.hexdigest(),source_root=str(root),arch='sm_89',Ns=2,precision='FP64',
    nvcc=command('/usr/local/cuda/bin/nvcc','--version'),
    thermal_build_mode=thermal_mode,thermal_additional_nvcc_flags=[] if thermal_mode=="standard" else [f"--Ofast-compile={level}","-Xcompiler=-fno-tree-fre"],
    material_build_mode=material_mode,material_additional_nvcc_flags=[] if material_mode=="standard" else [f"--Ofast-compile={level}","-Xcompiler=-fno-tree-fre"],
    build_mode=mode,fast_compile_level=level if mode=='fast' else None,additional_nvcc_flags=[f'--Ofast-compile={level}','-Xcompiler=-fno-tree-fre'] if mode=='fast' else [],
    performance_scope='fast compile native executable; not production optimization performance' if mode=='fast' else 'repository production optimization flags')
(out/'build-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
PY
cd "$source_root"
for solver in "${solvers[@]}"; do
 cp "$out/build-provenance.json" "$out/provenance-$solver.json"
 case "$solver" in
  CHMT) /usr/bin/time -v bash applications/CHMT/Allwmake > "$out/build-chmt.log" 2>&1 ;;
  CHT) /usr/bin/time -v bash tool/CHT/build_cuda_solver.sh > "$out/build-cht.log" 2>&1 ;;
  *) echo "Unknown solver: $solver" >&2; exit 2 ;;
 esac
 sha256sum "$out/bin/$solver" > "$out/$solver.sha256"
done
