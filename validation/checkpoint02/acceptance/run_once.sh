#!/usr/bin/env bash
# Only after the parent explicitly declares the unified checkpoint-2 batch.
# Compilation and freeze are separate. Never rebuild or repair here.
source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$here/run_batch.py" run --batch "${1:?usage: run_once.sh FROZEN_BATCH_DIRECTORY}"
