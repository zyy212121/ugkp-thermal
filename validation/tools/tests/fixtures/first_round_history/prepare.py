#!/usr/bin/env python3
"""Import the first-round parity baselines without changing the checked-out tree."""
from pathlib import Path as _ValidationPath
import sys as _validation_sys
_validation_sys.path.insert(0, str(next(p.parent for p in _ValidationPath(__file__).resolve().parents if p.name == "validation")))
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import os

HERE = (VALIDATION_ROOT / 'tests/fixtures/first_round_history')
ROOT = SOURCE_ROOT
SOURCE = 'refs/heads/perf/chmt-efficiency-refresh-20261011'
DESTINATION = 'refs/audit/first-round-original-history'
PIN = 'cab2910989abe37abd62586d2e16990c025f5766'
ENV = dict(os.environ, GIT_ALLOW_PROTOCOL='file', GIT_TERMINAL_PROMPT='0')


def main():
    available = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', PIN+'^{commit}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV)
    if available.returncode == 0:
        return
    subprocess.run(
        ['git', '-C', str(ROOT), '-c', 'gc.auto=0', 'fetch', '--no-tags',
         '--no-write-fetch-head', '--no-recurse-submodules', str((VALIDATION_ROOT / 'tests/fixtures/first_round_history/history.gitbundle')),
         SOURCE + ':' + DESTINATION],
        check=True, env=ENV,
    )
    print('Original parity history retained at ' + DESTINATION)


if __name__ == '__main__':
    main()
