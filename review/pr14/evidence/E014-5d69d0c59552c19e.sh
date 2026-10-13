#!/bin/bash
set -o pipefail
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
root=/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013
out=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/resident-terminal/v2
status=0
apps=("$@")
if ((${#apps[@]}==0)); then apps=(CHT FSH); fi
for app in "${apps[@]}"; do
  command=(g++ -std=c++17 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor -Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O1 -DNoRepository -ftemplate-depth-100 -DUGKWP_USE_CUDA -fPIC -fopenmp)
  command+=( -I"$root/applications/$app" -I"$root/applications/$app/gpu" -I"$root/applications/$app/thermal" -I"$root/common" -I"$FOAM_SRC/finiteVolume/lnInclude" -I"$FOAM_SRC/meshTools/lnInclude" -I"$FOAM_SRC/OpenFOAM/lnInclude" -I"$FOAM_SRC/OSspecific/POSIX/lnInclude" -MMD -MF "$out/$app.d" -c "$root/applications/$app/diluteUgkwpFoam.C" -o "$out/$app.o" )
  python - "$root" "$out/$app-before-hashes.json" <<'PYHASH'
from pathlib import Path
import sys,json,hashlib
root=Path(sys.argv[1]); records={}
for folder in ['applications','common']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.H','.C','.h','.hpp','.cuh','.inl','.cu']:
   records[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(records,sort_keys=True))
PYHASH
  printf '%q ' "${command[@]}" > "$out/$app-command.txt"; printf '\n' >> "$out/$app-command.txt"
  "${command[@]}" > "$out/$app-compile.log" 2>&1
  rc=$?; echo "$rc" > "$out/$app-exit.txt"; echo "$app compile-only exit $rc"; ((rc==0)) || status=1
 done
exit "$status"
