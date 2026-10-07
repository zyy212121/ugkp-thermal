#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
base=${CHMT_SCOPE_BASE:-${1:-db455604156419a9e20b13f1b43694fde33be6c8}}
git rev-parse --verify "$base^{commit}" >/dev/null
scratch=$(mktemp -d "${TMPDIR:-/tmp}/chmt-core.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# Existing pre-CHMT files are immutable. Default to the actual upstream parent
# of the CHMT feature, not HEAD (where this new application already exists).
# CHMT_SCOPE_BASE or an explicit argument can select another reviewed baseline.
git diff --name-status "$base" -- > "$scratch/tracked"
git ls-files --others --exclude-standard > "$scratch/untracked"
python3 - "$scratch/tracked" "$scratch/untracked" <<'PY'
import pathlib, sys
errors=[]
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    status,*paths=line.split('\t')
    if status!='A': errors.append('existing-file change: '+line)
    for path in paths:
        if not path.startswith('applications/CHMT/'): errors.append('outside new-app scope: '+path)
        if pathlib.PurePosixPath(path).name.lower().startswith('readme'): errors.append('forbidden README: '+path)
for path in pathlib.Path(sys.argv[2]).read_text().splitlines():
    if not path.startswith('applications/CHMT/'): errors.append('untracked outside new-app scope: '+path)
    if pathlib.PurePosixPath(path).name.lower().startswith('readme'): errors.append('forbidden README: '+path)
if errors:
    raise SystemExit('\n'.join(errors))
print('CHMT additive-only scope guard passed')
PY
compiler=${CXX:-g++}
"$compiler" -std=c++14 -Wall -Wextra -Werror -pedantic -Iapplications/CHMT \
    applications/CHMT/tests/test_core.cpp -o "$scratch/test_core"
"$scratch/test_core"
