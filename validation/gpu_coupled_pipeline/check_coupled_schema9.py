#!/usr/bin/env python3
"""Register the frozen schema-9 layout for the unchanged external checker.

Only model configuration changed since its schema-8 adapter. This checker reads
HostState's unchanged clock/budget tail, verifies the FNV checksum, and performs
its original conservation and flux assertions. The schema hash is deliberately
fixed, not learned from an arbitrary input file. No tolerance is changed.
"""
import importlib.util
import os
from pathlib import Path

path = Path(os.environ['UGKP_CASES_ROOT']) / 'test/tools/CHMT/coupled_runtime/check_case.py'
spec = importlib.util.spec_from_file_location('independent_coupled_schema9', path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.FIELD_SCHEMA_HASHES[8] = 3358196285796704403
checker.FIELD_SCHEMA_HASHES[9] = 2786795666536209721
if __name__ == '__main__':
    checker.main()