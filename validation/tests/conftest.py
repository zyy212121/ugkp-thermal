from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import sys


def pytest_addoption(parser):
    parser.addoption("--prepare-audit-history", action="store_true", default=False,
                     help="Import the bundled Git history for explicit historical parity tests")


def pytest_sessionstart(session):
    if not session.config.getoption("--prepare-audit-history"):
        return
    fixtures = VALIDATION_ROOT / 'tools/tests/fixtures'
    for name in ('first_round_history', 'continuation_history'):
        subprocess.run([sys.executable, str(fixtures / name / 'prepare.py')], check=True)
