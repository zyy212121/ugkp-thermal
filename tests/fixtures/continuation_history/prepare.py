#!/usr/bin/env python3
"""Import offline continuation provenance into an audit ref only."""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REF = 'refs/audit/continuation-original-history'


def main():
    subprocess.run(
        ['git', '-C', str(ROOT), '-c', 'gc.auto=0', 'fetch', '--no-tags',
         '--no-write-fetch-head', str(HERE / 'history.gitbundle'),
         REF + ':' + REF],
        check=True,
    )
    print('Original continuation history retained at ' + REF)


if __name__ == '__main__':
    main()
