#!/usr/bin/env bash
# Independent frozen CPU oracles and CUDA memory/synchronization checks.
set -euo pipefail
if [[ $# != 3 ]]; then echo "usage: $0 SOURCE_ROOT FROZEN_BASELINE_ROOT NEW_OUTPUT_DIRECTORY" >&2; exit 2; fi
root=$(realpath "$1")
baseline=$(realpath "$2")
out=$(realpath -m "$3")
mkdir "$out"
cuda=${CUDA_HOME:-/usr/local/cuda}
arch=${CHMT_CUDA_ARCH:-sm_89}
python3 - "$root" "$baseline" "$out" "$arch" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
root,baseline,out=map(Path,sys.argv[1:4]);arch=sys.argv[4]
name='applications/CHMT/mesh/Geometry.C'
frozen=subprocess.check_output(['git','show','88bc5434ee7199ec0bd7b986bbf121d693022245:'+name],cwd=baseline)
assert (baseline/name).read_bytes()==frozen,'Geometry.C oracle is not the frozen baseline'
paths=list((root/'common/gpu').glob('*Geometry*'))+[root/'applications/CHMT/mesh/Geometry.C',root/'applications/CHMT/mesh/GeometryGpu.H',root/'applications/CHMT/mesh/SweepWorkspace.H']
report={'frozen_commit':'88bc5434ee7199ec0bd7b986bbf121d693022245','arch':arch,'arithmetic':'FP64, contraction disabled; scaled double-double for original extended-range triangle distances','source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},'oracle_sha256':hashlib.sha256(frozen).hexdigest()}
(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
PY
"$cuda/bin/nvcc" -std=c++17 -O2 --fmad=false -lineinfo -arch="$arch" -c "$root/common/gpu/PolyhedralGeometry.cu" -o "$out/geometry.o" > "$out/build-cuda.log" 2>&1
for name in minimum_norm_cuda polyhedral_geometry_cuda harmonic_control_cuda sweep_linearization_cuda surface_edges_cuda boundary_motion_cuda geometry_rebuild_cuda; do
 "${CXX:-g++}" -std=c++17 -O2 -ffp-contract=off -I"$baseline/applications/CHMT" -I"$root/common" "$root/validation/gpu_coupled_pipeline/$name.cpp" "$out/geometry.o" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart -o "$out/$name" > "$out/build-$name.log" 2>&1
 "$out/$name" > "$out/$name.log" 2>&1
 "$cuda/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 "$out/$name" > "$out/$name-memcheck.log" 2>&1
 case "$name" in minimum_norm_cuda|polyhedral_geometry_cuda|harmonic_control_cuda)
  timeout 300 "$cuda/bin/compute-sanitizer" --tool synccheck --error-exitcode 99 "$out/$name" > "$out/$name-synccheck.log" 2>&1 ;;
 esac
 tail -n 1 "$out/$name.log"
done
python3 - "$out" <<'PY'
from pathlib import Path
import hashlib,json,sys
out=Path(sys.argv[1]);logs={}
for p in sorted(out.glob('*.log')):
 if p.name.startswith('build-'):continue
 text=p.read_text()
 if '-memcheck.' in p.name or '-synccheck.' in p.name:assert 'ERROR SUMMARY: 0 errors' in text,p
 else:assert 'PASS' in text,p
 logs[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':text}
(out/'results.json').write_text(json.dumps({'status':'PASS','checks':logs},indent=2)+'\n')
PY
