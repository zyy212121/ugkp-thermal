#!/usr/bin/env python3
"""Import offline continuation provenance into an audit ref only."""
from pathlib import Path
import subprocess
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REF = 'refs/audit/continuation-original-history'
PIN = 'dc79925991e8dd4f4fc8f339942a3b9e85e0c633'
ENV = dict(os.environ, GIT_ALLOW_PROTOCOL='file', GIT_TERMINAL_PROMPT='0')


def main():
    available = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', PIN+'^{commit}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV)
    if available.returncode == 0:
        return
    subprocess.run(
        ['git', '-C', str(ROOT), '-c', 'gc.auto=0', 'fetch', '--no-tags',
         '--no-write-fetch-head', '--no-recurse-submodules', str(HERE / 'history.gitbundle'),
         REF + ':' + REF],
        check=True, env=ENV,
    )
    print('Original continuation history retained at ' + REF)


if __name__ == '__main__':
    main()
