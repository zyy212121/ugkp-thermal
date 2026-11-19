import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tool.precision.materialize_sources import materialize


def records(root, output, name, fixtures):
    tree = materialize(root, 32)
    binary = output / (name + "-probe")
    command = ["g++", "-std=c++17", "-O2", "-I" + str(tree / "common"),
               "-I" + str(tree / "common/gasNumerics"), "-I" + str(fixtures),
               str(output / "probe.cpp"), "-o", str(binary)]
    subprocess.run(command, check=True)
    data = subprocess.check_output([binary])
    (output / (name + ".txt")).write_bytes(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 stationary gas recovery and diagnostics for 72 states.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT / "validation/tests/gas_transport"
    preamble = (fixtures / "gas_state_probe.cpp").read_text().split("void hashBytes", 1)[0]
    (output / "probe.cpp").write_text(preamble + (fixtures / "stationary_primitive_probe.cpp").read_text())
    reference = records(args.reference.resolve(), output, "reference", fixtures)
    candidate = records(args.candidate.resolve(), output, "candidate", fixtures)
    lines = reference.splitlines()
    if len(lines) != 72 or len(candidate.splitlines()) != len(lines):
        raise ValueError("Incomplete primitive state probe")
    if reference != candidate:
        index = next(i for i, pair in enumerate(zip(lines, candidate.splitlines())) if pair[0] != pair[1])
        raise ValueError(f"Primitive record {index} differs. See {output}")
    print("PASS", len(lines), "FP32 state and diagnostic records", hashlib.sha256(candidate).hexdigest())


if __name__ == "__main__":
    main()
