"""Read-only delivery smoke checks; outputs and extracted builds stay in work/tmp."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import tarfile
import tempfile

root = next(
    (p for p in Path(__file__).resolve().parents if (p / "common/gasWall/ReactingWallLayer.H").is_file()),
    Path("/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018"),
)
out = root / "work/nearwall-fix-20261020/delivery-verification"
out.mkdir(parents=True, exist_ok=True)
archive_path = root / "docs/diagnostics/nearwall-fix-20261020/performance-reproducer.tar.xz"
metadata = {
    "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "archive": str(archive_path.relative_to(root)),
    "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    "command": ["python3", "performance/cpu/run.py"],
    "gpu_executed": False,
}
with tempfile.TemporaryDirectory(prefix="pr22-delivery-review-") as td:
    scratch = Path(td)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts, member.name
            assert not member.issym() and not member.islnk(), member.name
        archive.extractall(scratch)
    required = scratch / "performance/cpu/common/gasNumerics/GpuSstAlgebra.cuh"
    assert required.is_file(), required
    with (out / "performance-clean-cpu.log").open("w") as log:
        log.write(f"scratch={scratch}\ncommand=python3 performance/cpu/run.py\n")
        log.flush()
        result = subprocess.run(metadata["command"], cwd=scratch, stdout=log, stderr=subprocess.STDOUT)
    metadata["cpu_exit_code"] = result.returncode
    assert result.returncode == 0

wall_path = root / "work/nearwall-fix-20261020/boundaryLayer-after/0.023300000000000001/uniform/gasBoundaryLayer.csv"
with wall_path.open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 336
nonfinite = {}
for column in rows[0]:
    bad = [i for i, row in enumerate(rows, 1) if not math.isfinite(float(row[column]))]
    if bad:
        nonfinite[column] = bad
metadata["bvp_wall_csv"] = str(wall_path.relative_to(root))
metadata["bvp_wall_csv_sha256"] = hashlib.sha256(wall_path.read_bytes()).hexdigest()
metadata["wall_rows"] = len(rows)
metadata["checked_numeric_columns"] = list(rows[0])
metadata["nonfinite_columns"] = nonfinite
assert not nonfinite
(out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
