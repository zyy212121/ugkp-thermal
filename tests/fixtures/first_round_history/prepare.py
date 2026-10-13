#!/usr/bin/env python3
"""Import the first-round parity baselines without changing the checked-out tree."""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = 'refs/heads/perf/chmt-efficiency-refresh-20261011'
DESTINATION = 'refs/audit/first-round-original-history'


def main():
    subprocess.run(
        ['git', '-C', str(ROOT), '-c', 'gc.auto=0', 'fetch', '--no-tags',
         '--no-write-fetch-head', str(HERE / 'history.gitbundle'),
         SOURCE + ':' + DESTINATION],
        check=True,
    )
    print('Original parity history retained at ' + DESTINATION)


if __name__ == '__main__':
    main()
