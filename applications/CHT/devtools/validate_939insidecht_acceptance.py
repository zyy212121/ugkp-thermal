#!/usr/bin/env python3
                                                                

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "939insideCHT"
LOG = CASE / "log.ugkpcht_0p1_to_0p2"
RUNTIME = CASE / "runtime.ugkpcht_0p1_to_0p2"
REPORT = CASE / "acceptance.ugkpcht_0p1_to_0p2.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dictionary_value(path: Path, entry: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"(?m)^\s*{re.escape(entry)}\s+(\"[^\"]*\"|[^;]+)\s*;",
        text,
    )
    if not match:
        raise RuntimeError(f"{path}: missing {entry}")
    return match.group(1).strip().strip('"')


def final_time() -> tuple[float, Path]:
    times: list[tuple[float, Path]] = []
    for path in CASE.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if value > 0.1001:
            times.append((value, path))
    if not times:
        raise RuntimeError("no result time after 0.1")
    value, path = max(times)
    if abs(value - 0.2) > 2.0e-4:
        raise RuntimeError(f"latest result {value:.17g} is not 0.2")
    return value, path


def main() -> None:
    time_value, time_path = final_time()
    state = time_path / "thermalExchangeState"
    manifest = time_path / "thermalExchangeManifest"
    for required in (LOG, RUNTIME, state, manifest):
        if not required.is_file():
            raise RuntimeError(f"missing acceptance artifact: {required}")

    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = (
        r"FOAM FATAL",
        r"CUDA error",
        r"Segmentation fault",
        r"(?m)^\s*Floating point exception(?:\s+\(core dumped\))?\s*$",
        r"\b(?:nan|inf)\b",
    )
    for pattern in fatal_patterns:
        if re.search(pattern, log_text, re.IGNORECASE):
            raise RuntimeError(f"log contains forbidden pattern: {pattern}")

    runtime_match = re.search(
        r"elapsed_seconds=([0-9.eE+-]+)",
        RUNTIME.read_text(encoding="utf-8"),
    )
    if not runtime_match:
        raise RuntimeError("runtime file has no elapsed_seconds")
    elapsed = float(runtime_match.group(1))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"invalid elapsed time: {elapsed}")

    start_state = CASE / "0.1/thermalExchangeState"
    start_sequence = int(dictionary_value(start_state, "exchangeSequence"))
    final_sequence = int(dictionary_value(state, "exchangeSequence"))
    if final_sequence <= start_sequence:
        raise RuntimeError(
            f"coupling sequence did not advance: {start_sequence} -> "
            f"{final_sequence}"
        )

    completed_time = float(dictionary_value(state, "completedSimulationTimeS"))
    if completed_time <= 0.1:
        raise RuntimeError(
            f"completedSimulationTimeS did not advance: {completed_time}"
        )

    required_true = (
        "gasWallLedgerConsumed",
        "particleWallLedgerConsumed",
        "particleRadiationApplied",
        "solidStateUpdated",
        "wallTemperatureUploaded",
        "particleMomentsRebuilt",
    )
    for entry in required_true:
        if dictionary_value(state, entry) != "true":
            raise RuntimeError(f"{state}: {entry} is not true")

    report = (
        "# UGKP 939insideCHT acceptance\n\n"
        f"- final directory: `{time_path.name}`\n"
        f"- elapsed wall time: `{elapsed:.9g} s`\n"
        f"- coupling sequence: `{start_sequence} -> {final_sequence}`\n"
        f"- completed coupling time: `{completed_time:.17g} s`\n"
        "- thermal ledgers/radiation/solid update/upload/moment rebuild: "
        "`all true`\n"
        "- fatal/non-finite log scan: `clean`\n"
        f"- log SHA-256: `{sha256(LOG)}`\n"
        f"- runtime SHA-256: `{sha256(RUNTIME)}`\n"
        f"- thermalExchangeState SHA-256: `{sha256(state)}`\n"
        f"- thermalExchangeManifest SHA-256: `{sha256(manifest)}`\n"
    )
    REPORT.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
