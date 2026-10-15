from pathlib import Path
import subprocess
import sys


def pytest_sessionstart(session):
    fixtures = Path(__file__).resolve().parent / 'fixtures'
    for name in ('first_round_history', 'continuation_history'):
        subprocess.run([sys.executable, str(fixtures / name / 'prepare.py')], check=True)
