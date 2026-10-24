#!/usr/bin/env bash
# Compatibility entrypoint: gasUGKP now links its resident CUDA owner directly.
exec bash "$(dirname "${BASH_SOURCE[0]}")/build_cuda_solver.sh" "$@"
