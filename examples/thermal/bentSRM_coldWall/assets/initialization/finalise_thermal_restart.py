#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CASE = Path(__file__).resolve().parents[2]


def state_text(time_s: float) -> str:
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
gasWallLedgerConsumed false;
particleRadiationApplied false;
solidStateUpdated false;
wallTemperatureUploaded false;
particleMomentsRebuilt false;
particleContactEnergyJ 0;
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=1.5)
    args = parser.parse_args()
    directory = CASE/f"{args.time:g}"
    state = directory/"thermalExchangeState"
    state.write_text(state_text(args.time))
    archive = CASE/"assets/initialization/formal_1p5_checkpoint"
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(directory, archive)
    print(f"archive={archive}")


if __name__ == "__main__":
    main()

