#!/usr/bin/env python3
"""Import the first-round parity baselines without changing the checked-out tree."""
from pathlib import Path
import subprocess
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
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
         '--no-write-fetch-head', '--no-recurse-submodules', str(HERE / 'history.gitbundle'),
         SOURCE + ':' + DESTINATION],
        check=True, env=ENV,
    )
    print('Original parity history retained at ' + DESTINATION)


if __name__ == '__main__':
    main()
