#!/usr/bin/env python3
"""Compare repeated resident solid windows with the independent native owner."""
import argparse
import hashlib
import json
import itertools
import math
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("meshes", "binaries", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--varying-initial-state", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "cases": [], "binaries": {
        kind: hashlib.sha256((args.binaries / ("windows-" + kind)).read_bytes()).hexdigest()
        for kind in ("baseline", "candidate")}}

    def save():
        (args.work / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    modes = [("constant", []), ("variable", ["-variable-properties"]),
             ("variable-aitken", ["-variable-properties", "-aitken"])]
    if args.varying_initial_state:
        for _, flags in modes:
            flags.append("-varying-initial-state")
    seeds = sorted(p for p in args.meshes.iterdir() if p.is_dir())
    for seed, (mode, flags) in itertools.product(seeds, modes):
        case = args.work / (seed.name + "-" + mode)
        case.mkdir()
        for folder in ("system", "constant", "0"):
            shutil.copytree(seed / folder, case / folder)
        (case / "system/solid/fvSolution").write_text(
            "FoamFile {version 2.0; format ascii; class dictionary; object fvSolution;}\n"
            "solvers {TsolidCandidate {solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 2000;}}\n")
        row = {"case": seed.name, "mode": mode, "runs": {}}
        report["cases"].append(row)
        save()
        for kind in ("baseline", "candidate"):
            log = case / (kind + ".log")
            with log.open("w") as output:
                result = subprocess.run([str(args.binaries / ("windows-" + kind)), "-case", str(case), *flags],
                                        stdout=output, stderr=subprocess.STDOUT)
            values = {}
            for line in log.read_text().splitlines():
                if line.startswith(("WINDOW_CELL ", "WINDOW_FLUX ", "WINDOW_ENERGY ")):
                    words = line.split()
                    values[" ".join(words[:-1])] = float(words[-1])
            row["runs"][kind] = {"exit_code": result.returncode, "values": values}
            save()
            assert result.returncode == 0 and values, (case, kind)
            assert all(math.isfinite(value) for value in values.values()), (case, kind)
            assert {int(key.split()[1]) for key in values} == {1, 2, 3, 4}
        before = row["runs"]["baseline"]["values"]
        after = row["runs"]["candidate"]["values"]
        assert before.keys() == after.keys()
        for prefix, metric in (("WINDOW_CELL", "temperature_error"), ("WINDOW_FLUX", "boundary_flux_error"), ("WINDOW_ENERGY", "energy_error")):
            row[metric] = max(abs(before[key] - after[key]) for key in before if key.startswith(prefix))
            assert row[metric] < 2e-8, row
        row["status"] = "PASS"
        save()
        print(seed.name, mode, "PASS", flush=True)
    assert report["cases"]
    report["status"] = "PASS"
    save()


if __name__ == "__main__":
    main()
