#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


CASE = Path(__file__).resolve().parents[2]


def state_text(time_s: float, fluid_hash: str, solid_hash: str, config_hash: str) -> str:
    return f'''FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "{time_s:g}";
    object thermalExchangeState;
}}

formatVersion 1;
initialState true;
exchangeSequence "0";
completedTimeIndex 0;
completedSimulationTimeS {time_s:.17g};
previousExchangeSimulationTimeS {time_s:.17g};
fluidMeshTopologySha1 "{fluid_hash}";
solidMeshTopologySha1 "{solid_hash}";
couplingConfigurationSha1 "{config_hash}";
wallTemperatureSha1UsedForCompletedInterval "0000000000000000000000000000000000000000";
newlyUploadedWallTemperatureSha1 "0000000000000000000000000000000000000000";
gasWallLedgerConsumed false;
particleRadiationApplied false;
solidStateUpdated false;
wallTemperatureUploaded false;
particleMomentsRebuilt false;
particleContactEnergyJ 0;
'''


def zero_step(control: Path, time_s: float):
    source = control.read_text()
    temporary = re.sub(r"endTime\s+[-+0-9.eE]+\s*;", f"endTime {time_s:.17g};", source, count=1)
    control.write_text(temporary)
    try:
        return subprocess.run(["CHT", "-case", str(CASE)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finally:
        control.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=1.5)
    args = parser.parse_args()
    directory = CASE/f"{args.time:g}"
    state = directory/"thermalExchangeState"
    zeros = "0"*40
    state.write_text(state_text(args.time, zeros, zeros, zeros))
    first = zero_step(CASE/"system/controlDict", args.time)
    pattern = re.compile(r"fluidMesh=[0-9a-f]{40}/([0-9a-f]{40}).*?solidMesh=[0-9a-f]{40}/([0-9a-f]{40}).*?coupling=[0-9a-f]{40}/([0-9a-f]{40})", re.S)
    match = pattern.search(first.stdout)
    if not match:
        raise SystemExit("cannot extract expected hashes from CHT preflight:\n" + first.stdout[-4000:])
    state.write_text(state_text(args.time, *match.groups()))
    second = zero_step(CASE/"system/controlDict", args.time)
    if second.returncode != 0:
        raise SystemExit("corrected zero-step preflight failed:\n" + second.stdout[-4000:])
    archive = CASE/"assets/initialization/formal_1p5_checkpoint"
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(directory, archive)
    print(f"fluidMesh={match.group(1)}")
    print(f"solidMesh={match.group(2)}")
    print(f"configuration={match.group(3)}")
    print(f"archive={archive}")


if __name__ == "__main__":
    main()

