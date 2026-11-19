#!/usr/bin/env bash

exec bash "$(dirname "${BASH_SOURCE[0]}")/build_cuda_solver.sh" "$@"
