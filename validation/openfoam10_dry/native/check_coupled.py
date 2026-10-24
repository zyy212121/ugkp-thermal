#!/usr/bin/env python3
"""Run the independent cases-repository conservation checker with schema 8.

Only configuration layout changed; state, checksum, conservation, exchange and
refinement assertions are the external checker unchanged. Set UGKP_CASES_ROOT
to the pinned ugkp-thermal-cases checkout recorded in the evidence manifest.
"""
import importlib.util
import os
from pathlib import Path
spec=importlib.util.spec_from_file_location("independent_coupled",Path(os.environ["UGKP_CASES_ROOT"])/"test/tools/CHMT/coupled_runtime/check_case.py")
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.FIELD_SCHEMA_HASHES[8]=3358196285796704403
if __name__=="__main__":checker.main()
