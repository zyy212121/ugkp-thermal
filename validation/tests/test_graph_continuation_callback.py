import subprocess
import sys

# The error case poisons its CUDA context and must run in its own process.
# This is a diagnostic timeout, not a production solver stopping condition.
for mode in ("registry-errors", "normal", "submission-error", "async-error", "static-lifetime"):
    subprocess.run([sys.argv[1], mode], check=True, timeout=30)
