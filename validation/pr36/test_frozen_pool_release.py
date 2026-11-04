"""Host execution of the full shared pool chain; CUDA is an explicit companion run."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def test_frozen_pool_release_conserves_cancelled_candidate_energy(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'validation/pr36/run_frozen_pool_release.py'),
        '--output',str(tmp_path/'host')],check=True)
