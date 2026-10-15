from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import sys


def pytest_sessionstart(session):
    fixtures = VALIDATION_ROOT / 'tools/tests/fixtures'
    for name in ('first_round_history', 'continuation_history'):
        subprocess.run([sys.executable, str(fixtures / name / 'prepare.py')], check=True)
