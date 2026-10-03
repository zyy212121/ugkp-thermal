#!/usr/bin/env python3

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

KEYS = (
    "gpuResidentStrict",
    "gpuResidentPureGasOnly",
    "gpuResidentDynamicInlet",
    "gpuResidentParticleCapacity",
    "gpuResidentMaxFaceWalkHops",
    "gpuResidentCourantUpdateInterval",
    "gpuResidentMaxDeltaTGrowth",
    "gpuCsrCellLocalPath",
    "gpuCsrHeavyReduction",
    "gpuCsrHeavyReductionAutoInterval",
    "gpuParticleBlockThreads",
    "gpuReductionBlockThreads",
    "gpuCsrWarpAggregatedBinning",
    "gpuCsrSplitPreDirectory",
    "gpuCsrLevel",
    "bn",
)

DEFAULTS = {
    "gpuResidentPureGasOnly": "false",
    "gpuResidentDynamicInlet": "false",
    "gpuResidentParticleCapacity": "1000000",
    "gpuResidentMaxFaceWalkHops": "32",
    "gpuResidentCourantUpdateInterval": "1000",
    "gpuResidentMaxDeltaTGrowth": "1.05",
    "gpuCsrCellLocalPath": "false",
    "gpuCsrHeavyReduction": "false",
    "gpuCsrHeavyReductionAutoInterval": "100",
    "gpuParticleBlockThreads": "128",
    "gpuReductionBlockThreads": "128",
    "gpuCsrWarpAggregatedBinning": "false",
    "gpuCsrSplitPreDirectory": "true",
}


def scalar_value(text: str, key: str):
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s+([^;\s]+)\s*;\s*$",
        text,
    )
    return match.group(1) if match else None


def migrate(path: Path) -> None:
    original = path.read_text()
    schedule = path.with_name("schedulingProperties")
    existing_schedule = schedule.read_text() if schedule.is_file() else ""
    for text in (original, existing_schedule):
        if re.search(r"(?m)^\s*gpuResearchVariant\b", text):
            raise ValueError("thermal rejects gpuResearchVariant; select gpuCsrLevel L0/L1/L2/auto")
        level = scalar_value(text, "gpuCsrLevel")
        if level is not None and level not in ("L0", "L1", "L2", "auto"):
            raise ValueError("thermal requires gpuCsrLevel L0/L1/L2/auto; choose a level before migration")
    values = dict(DEFAULTS)
    for key in KEYS:
        value = scalar_value(existing_schedule, key)
        if value is None:
            value = scalar_value(original, key)
        if value is not None and key not in ("bn", "gpuResidentStrict"):
            values[key] = value

    csr_level = values.get("gpuCsrLevel")
    if csr_level not in ("L0", "L1", "L2", "auto"):
        heavy_mode = values["gpuCsrHeavyReduction"]
        if heavy_mode == "auto":
            csr_level = "auto"
        elif heavy_mode == "true":
            csr_level = "L2"
        elif values["gpuCsrCellLocalPath"] == "true":
            csr_level = "L1"
        else:
            csr_level = "L0"

    if csr_level == "auto" and int(values["gpuCsrHeavyReductionAutoInterval"]) < 1:
        raise ValueError("gpuCsrHeavyReductionAutoInterval must be positive in auto mode")

    bn = scalar_value(original, "bn")
    if bn is not None:
        old_threads = str(1 << int(bn))
        if scalar_value(original, "gpuParticleBlockThreads") is None:
            values["gpuParticleBlockThreads"] = old_threads
        if scalar_value(original, "gpuReductionBlockThreads") is None:
            values["gpuReductionBlockThreads"] = old_threads

    cleaned = original
    for key in KEYS:
        cleaned = re.sub(
            rf"(?m)^\s*{re.escape(key)}\s+[^;\n]+;\s*\n?",
            "",
            cleaned,
        )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    path.write_text(cleaned)

    automatic_interval_line = (
        f"gpuCsrHeavyReductionAutoInterval {values['gpuCsrHeavyReductionAutoInterval']};\n"
        if csr_level == "auto" else ""
    )
    schedule.write_text(
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"constant\";\n"
        "    object      schedulingProperties;\n"
        "}\n\n"
        "schemaVersion 1;\n\n"
        f"gpuResidentPureGasOnly          {values['gpuResidentPureGasOnly']};\n"
        f"gpuResidentDynamicInlet         {values['gpuResidentDynamicInlet']};\n"
        f"gpuResidentParticleCapacity     {values['gpuResidentParticleCapacity']};\n"
        f"gpuResidentMaxFaceWalkHops      {values['gpuResidentMaxFaceWalkHops']};\n"
        f"gpuResidentCourantUpdateInterval {values['gpuResidentCourantUpdateInterval']};\n"
        f"gpuResidentMaxDeltaTGrowth      {values['gpuResidentMaxDeltaTGrowth']};\n\n"
        f"gpuCsrLevel                     {csr_level};\n"
        f"{automatic_interval_line}"
        "\n"
        f"gpuParticleBlockThreads         {values['gpuParticleBlockThreads']};\n"
        f"gpuReductionBlockThreads        {values['gpuReductionBlockThreads']};\n"
    )


def main() -> None:
    paths = sorted(EXAMPLES.rglob("constant/particleProperties"))
    for path in paths:
        migrate(path)
    print(f"migrated={len(paths)}")


if __name__ == "__main__":
    main()
