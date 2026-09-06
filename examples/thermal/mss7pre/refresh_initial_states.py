#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
THERMAL = ROOT.parent
CASES = (
    "MSS7_laminar",
    "MSS7_turbulent_wallModel",
    "MSS7_twoPhase_sparse",
    "MSS7_twoPhase_dense",
)


def state_text(fluid_hash: str, solid_hash: str, config_hash: str) -> str:
    return f'''FoamFile
{{
    format ascii;
    class dictionary;
    location "1";
    object thermalExchangeState;
}}

formatVersion 1;
initialState true;
exchangeSequence "0";
completedTimeIndex 0;
completedSimulationTimeS 1;
previousExchangeSimulationTimeS 1;
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


def main() -> None:
    pattern = re.compile(
        r"fluidMesh=[0-9a-f]{40}/([0-9a-f]{40}).*?"
        r"solidMesh=[0-9a-f]{40}/([0-9a-f]{40}).*?"
        r"coupling=[0-9a-f]{40}/([0-9a-f]{40})",
        re.S,
    )
    zeros = "0" * 40
    for case_name in CASES:
        with tempfile.TemporaryDirectory(prefix=f"mss7pre-state-{case_name}-") as temporary:
            case = Path(temporary)
            shutil.copytree(THERMAL / case_name / "constant", case / "constant")
            shutil.copytree(THERMAL / case_name / "system", case / "system")
            shutil.copytree(ROOT / "checkpoints" / case_name / "1", case / "1")
            (case / "1/thermalExchangeState").write_text(state_text(zeros, zeros, zeros))
            completed = subprocess.run(
                ["CHT", "-case", str(case)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            match = pattern.search(completed.stdout)
            if match is None:
                raise RuntimeError(
                    f"cannot extract current hashes for {case_name}:\n{completed.stdout[-4000:]}"
                )
            destination = ROOT / "checkpoints" / case_name / "1/thermalExchangeState"
            destination.write_text(state_text(*match.groups()))
            print(
                f"state={case_name} fluid={match.group(1)} solid={match.group(2)} "
                f"coupling={match.group(3)}"
            )


if __name__ == "__main__":
    main()
