import subprocess
import sys



for mode in ("registry-errors", "normal", "submission-error", "async-error", "static-lifetime"):
    subprocess.run([sys.argv[1], mode], check=True, timeout=30)
