#!/usr/bin/env python3
                                                                    

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "gpu" / "GpuResidentStrict.cu"
CUDA_HEADER = ROOT / "gpu" / "GpuResidentStrict.H"
PARCEL_MASS_LIMIT = 5.0e-8
SOURCE_FINGERPRINTS = {
    "v1": (12751, "b2d99662deccb1ed600c6399ba927d0dd508faed5a1a28d2371e0128588c4278"),
    "v2": (12751, "b2d99662deccb1ed600c6399ba927d0dd508faed5a1a28d2371e0128588c4278"),
}
CUDA_HEADER_SHA256 = "aadc2f2dcb37355ff8e1f4d479089a9db1750ac6d2f904afda50900764c85263"
PERFORMANCE_SEED_SHA256 = {
    "gpuResidentStrictParticles.dat": "680b74506411228c969466ccfa6639cd9a799ffe3212d851b60d17dc60bbf268",
    "gpuResidentStrictEpsGPrev.dat": "fd319d816a26dc05c50096b68f4349473220f85b13987260bcb60956b6521202",
    "gpuResidentStrictSourceResidual.dat": "b571e9832aad7cacdb749c9b2e3d5371362863701db6eaba2293f91f861a6287",
}
PERFORMANCE_MIE_SHA256 = "5613b9f3baaf4bfe19a9e71e9d293adaedfd6ba930946c0cbe141bab761a9619"


def extract_braced(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing source marker: {marker}")
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"missing opening brace after: {marker}")
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated source block after: {marker}")


def strip_comments(text: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL)


def scalar_entries(path: Path, key: str) -> list[float]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    pattern = rf"\b{re.escape(key)}\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;"
    return [float(match) for match in re.findall(pattern, text)]


def parse_single_scalar(path: Path, key: str) -> float:
    text = strip_comments(path.read_text(encoding="utf-8"))
    entries = re.findall(rf"\b{re.escape(key)}\s+([^;]+)\s*;", text)
    if len(entries) != 1:
        raise ValueError(f"expected exactly one {key} entry in {path}, found {len(entries)}")
    numeric = re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", entries[0].strip()
    )
    if numeric is None:
        raise ValueError(f"invalid scalar {key}={entries[0]!r} in {path}")
    value = float(entries[0])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={value!r} in {path}")
    return value


def parse_single_word(path: Path, key: str, default: str | None = None) -> str:
    text = strip_comments(path.read_text(encoding="utf-8"))
    entries = re.findall(rf"\b{re.escape(key)}\s+([^;]+)\s*;", text)
    if not entries and default is not None:
        return default
    if len(entries) != 1:
        raise ValueError(f"expected exactly one {key} entry in {path}, found {len(entries)}")
    value = entries[0].strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value) is None:
        raise ValueError(f"invalid word {key}={entries[0]!r} in {path}")
    return value


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_single_mie_table(case_dir: Path, coupling: Path) -> Path:
    coupling_text = strip_comments(coupling.read_text(encoding="utf-8"))
    entries = re.findall(r"\bmieTable\s+[^;]*;", coupling_text)
    values = re.findall(r'\bmieTable\s+"([^"]+)"\s*;', coupling_text)
    if len(entries) != 1 or len(values) != 1:
        raise ValueError(
            f"expected exactly one valid mieTable path, "
            f"found {len(entries)} total/{len(values)} valid"
        )
    mie_path = Path(values[0])
    if not mie_path.is_absolute():
        mie_path = case_dir / mie_path
    if not mie_path.is_file():
        raise ValueError(f"mieTable path does not exist: {mie_path}")
    return mie_path


def expected_time_directory(case_dir: Path, start_time: float) -> Path:
    matches = []
    tolerance = max(1.0e-15, abs(start_time)*1.0e-12)
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if math.isclose(value, start_time, rel_tol=0.0, abs_tol=tolerance):
            matches.append(child)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one start-time directory for {start_time:.17g}, found {len(matches)}"
        )
    return matches[0]


def source_contracts(require_pool: bool, require_relax: bool) -> list[str]:
    text = CUDA_SOURCE.read_text(encoding="utf-8")
    header = CUDA_HEADER.read_text(encoding="utf-8")
    failures: list[str] = []

    if require_pool or require_relax:
        stage = "v2" if require_relax else "v1"
        expected_lines, expected_sha = SOURCE_FINGERPRINTS[stage]
        actual_lines = len(text.splitlines())
        actual_sha = source_sha256(CUDA_SOURCE)
        if (actual_lines, actual_sha) != (expected_lines, expected_sha):
            failures.append(
                f"{stage} CUDA source fingerprint mismatch: "
                f"expected {expected_lines} lines/{expected_sha}, "
                f"got {actual_lines} lines/{actual_sha}"
            )
    header_sha = source_sha256(CUDA_HEADER)
    if header_sha != CUDA_HEADER_SHA256:
        failures.append(
            f"CUDA header changed: expected {CUDA_HEADER_SHA256}, got {header_sha}"
        )

    state = extract_braced(text, "struct DeviceState")
    relax = extract_braced(text, "__device__ void relaxOneParticleToResidentGas")
    impact = extract_braced(text, "__device__ void applyParticleWallImpactHeat")
    pool = extract_braced(text, "__global__ void accumulatePoissonPoolParticlesByCellKernel")
    create = extract_braced(text, 'extern "C" int ugkwpGpuResidentStrictCreate')
    scrub = extract_braced(text, "void scrubHostCalculationScalars")
    initialise = extract_braced(header, "void initialiseBase")

    for token in (
        "const double ux = s.pux[i];",
        "s.solveParticleTemperature != 0",
        "s.pT[i] = tpNew;",
    ):
        if token not in relax:
            failures.append(f"missing protected CHT relaxation token: {token}")

    for token in (
        "impactParticleWallContact",
        "atomicAdd(&s.particleWallEnergy[globalFaceId], result.wallEnergyJ);",
    ):
        if token not in impact:
            failures.append(f"missing protected reflected-impact heat token: {token}")

    for forbidden in (
        "pStuckFaceId",
        "pAdep",
        "particleWallDepositionMask",
        "depositedParticleWallContact",
        "kind == 5",
    ):
        if forbidden in text + header:
            failures.append(f"forbidden retained-particle token: {forbidden}")

    for token in (
        "relaxAndTrackParticlesKernel",
        "UGKP_DEVELOPMENT_PROBES",
        "DevelopmentProbeState",
    ):
        if token in text:
            failures.append(f"forbidden production token: {token}")

    if "const double mu = clampMin(s.gasMu, 1.0e-30);" not in relax:
        failures.append("particle relaxation must read gasMu at runtime")
    if "s.rhoSolid*dPart + 1.0e-300" not in relax:
        failures.append("raw rhoSolid drag denominator changed")

    forbidden_cache_patterns = (
        r"\b(?:cached|constant|fixed)Mu\b",
        r"\b(?:inv|inverse)Mu\b",
        r"\bmu(?:Cached|Constant|Fixed|Inv|Inverse)\b",
        r"\bprecomputed(?:Heat|Thermal|Nusselt|Drag)Rate\b",
    )
    for pattern in forbidden_cache_patterns:
        if re.search(pattern, state, flags=re.IGNORECASE):
            failures.append(f"DeviceState contains forbidden mu-dependent cache: {pattern}")

    if initialise.count("uploadBoundarySources") != 1:
        failures.append("initialiseBase must call uploadBoundarySources exactly once")
    if initialise.count("uploadFields") != 1:
        failures.append("initialiseBase must call uploadFields exactly once")
    if initialise.find("uploadBoundarySources") >= initialise.find("uploadFields"):
        failures.append("uploadBoundarySources must precede uploadFields and host scrub")

    full_sync_count = len(re.findall(r"\bsyncDeviceState\s*\(\s*s\s*,", text))
    if full_sync_count != 3:
        failures.append(f"expected exactly three full DeviceState synchronizations, found {full_sync_count}")

    if require_pool:
        if pool.count("granularCollisionTauFromCellDevice") != 1:
            failures.append("Poisson pool must evaluate collision tau exactly once")
        thread_zero = pool.find("threadIdx.x == 0")
        tau_eval = pool.find("granularCollisionTauFromCellDevice")
        barrier = pool.find("__syncthreads()")
        consume = pool.find("const double prob = cellCollisionProbability;")
        if thread_zero < 0 or not (thread_zero < tau_eval < barrier < consume):
            failures.append("Poisson probability must be produced by thread 0 and consumed after a barrier")
        if not re.search(r"__shared__\s+double\s+cellCollisionProbability\s*;", pool):
            failures.append("missing dedicated shared collision probability")

    if require_relax:
        fields = (
            "gasCp",
            "invRhoSolid",
            "particleThermalCapacity",
            "gasPrClamped",
            "gasPrOneThird",
        )
        for field in fields:
            if field not in state:
                failures.append(f"missing viscosity-independent field: {field}")
            if f"s->{field} = 0.0;" not in scrub:
                failures.append(f"host mirror does not scrub: {field}")
        initialisers = (
            "s->gasCp = gammaGas*Rgas/gammaCpDenominator;",
            "s->invRhoSolid = 1.0/rhoSolidDenominator;",
            "s->particleThermalCapacity = particleThermalRho*particleCp;",
            "s->gasPrClamped = gasPr < 1.0e-12 ? 1.0e-12 : gasPr;",
            "s->gasPrOneThird = std::pow(s->gasPrClamped, 1.0/3.0);",
        )
        for token in initialisers:
            if token not in create:
                failures.append(f"missing cache initialiser: {token}")
        for field in fields:
            if re.search(rf"s->{field}\s*=.*\bgasMu\b", create):
                failures.append(f"cached field depends on gasMu: {field}")
                                                                   
                                                                         
                                                                           
                                                                            
        expected_counts = {
            "gasCp": 17,
            "invRhoSolid": 12,
            "particleThermalCapacity": 4,
            "gasPrClamped": 8,
            "gasPrOneThird": 4,
        }
        for field, expected_count in expected_counts.items():
            actual_count = len(re.findall(rf"\b{field}\b", text))
            if actual_count != expected_count:
                failures.append(
                    f"unexpected {field} occurrence count: expected {expected_count}, got {actual_count}"
                )
        required_relax = (
            "molecularGasConductivity(s)",
            "s.particleThermalCapacity*dPart*dPart + 1.0e-300",
            "-finiteOr(s.gradPx[c], 0.0)*s.invRhoSolid",
            "-finiteOr(s.gradPy[c], 0.0)*s.invRhoSolid",
            "-finiteOr(s.gradPz[c], 0.0)*s.invRhoSolid",
        )
        for token in required_relax:
            if token not in relax:
                failures.append(f"relaxation missing required expression: {token}")

    return failures


def case_contracts(
    case_dir: Path,
    expected_kappa: str | None,
    require_fresh: bool,
    require_prepared: bool,
    require_performance_seed: bool = False,
) -> list[str]:
    failures: list[str] = []
    case_dir = case_dir.resolve()
    root_properties = case_dir / "constant" / "ugkwpProperties"
    control = case_dir / "system" / "controlDict"
    coupling = case_dir / "constant" / "solidThermalCouplingProperties"
    fluid_region = "region0"

    try:
        fluid_region = parse_single_word(root_properties, "gpuResidentFluidRegion", "region0")
        region_properties = case_dir / "constant" / fluid_region / "ugkwpProperties"
        property_paths = [root_properties]
        if region_properties.is_file() and region_properties != root_properties:
            property_paths.append(region_properties)
        parcel_masses = [parse_single_scalar(path, "parcelMass") for path in property_paths]
        for parcel_mass in parcel_masses:
            if parcel_mass <= 0.0 or parcel_mass >= PARCEL_MASS_LIMIT:
                failures.append(
                    f"unsafe parcelMass={parcel_mass:.17g}; require 0 < parcelMass < {PARCEL_MASS_LIMIT:g}"
                )
        if any(value != parcel_masses[0] for value in parcel_masses[1:]):
            failures.append(f"root/region parcelMass mirrors disagree: {parcel_masses}")
        if require_performance_seed:
            if any(value != 5.0e-9 for value in parcel_masses):
                failures.append(
                    f"performance seed requires parcelMass=5e-9, got {parcel_masses}"
                )
            capacities = [
                parse_single_scalar(path, "gpuResidentParticleCapacity")
                for path in property_paths
            ]
            if any(value != 65536 for value in capacities):
                failures.append(
                    f"performance seed requires capacity 65536, got {capacities}"
                )
            dynamic_inlet = [
                parse_single_word(path, "gpuResidentDynamicInlet")
                for path in property_paths
            ]
            if any(value not in ("false", "no", "off") for value in dynamic_inlet):
                failures.append(
                    f"performance seed requires dynamic inlet off, got {dynamic_inlet}"
                )
    except (OSError, ValueError) as error:
        failures.append(str(error))

    try:
        root_gks = case_dir / "constant" / "gksProperties"
        region_gks = case_dir / "constant" / fluid_region / "gksProperties"
        gks_paths = [path for path in (root_gks, region_gks) if path.is_file()]
        if not gks_paths:
            raise ValueError(f"missing root/region gksProperties in {case_dir}")
        kappas = [parse_single_scalar(path, "gasKappa") for path in gks_paths]
        if any(value != kappas[0] for value in kappas[1:]):
            failures.append(f"root/region gasKappa mirrors disagree: {kappas}")
        kappa = kappas[0]
        if expected_kappa == "fallback" and kappa > 0.0:
            failures.append(f"fallback case has positive gasKappa={kappa:.17g}")
        if expected_kappa == "explicit" and not math.isclose(kappa, 0.024, rel_tol=0.0, abs_tol=0.0):
            failures.append(f"explicit case requires gasKappa=0.024, got {kappa:.17g}")
        if require_performance_seed and kappa != -1.0:
            failures.append(f"performance seed requires fallback gasKappa=-1, got {kappa:.17g}")
    except (OSError, ValueError) as error:
        failures.append(str(error))

    try:
        start_from = parse_single_word(control, "startFrom")
        if start_from not in ("startTime", "latestTime"):
            failures.append(f"unsupported startFrom mode: {start_from}")
        adjust_time_step = parse_single_word(control, "adjustTimeStep")
        if adjust_time_step not in ("false", "no", "off"):
            failures.append("fixed-step case requires adjustTimeStep false")
        start_time = parse_single_scalar(control, "startTime")
        if start_time < 0.0:
            failures.append(f"startTime must be non-negative, got {start_time:.17g}")
        write_precision = parse_single_scalar(control, "writePrecision")
        if write_precision < 17 or not write_precision.is_integer():
            failures.append(
                f"ASCII thermal restart requires integer writePrecision >= 17, got {write_precision:.17g}"
            )
        if (require_fresh or require_prepared or require_performance_seed) and start_from != "startTime":
            failures.append("fresh/prepared/performance case must use startFrom startTime")
        if require_performance_seed:
            delta_t = parse_single_scalar(control, "deltaT")
            end_time = parse_single_scalar(control, "endTime")
            write_interval = parse_single_scalar(control, "writeInterval")
            if delta_t != 1.0e-8:
                failures.append(f"performance seed requires deltaT=1e-8, got {delta_t:.17g}")
            expected_end = start_time + 128.0*delta_t
            if not math.isclose(end_time, expected_end, rel_tol=0.0, abs_tol=1.0e-15):
                failures.append(
                    f"performance seed must advance 128 fixed steps: "
                    f"expected endTime {expected_end:.17g}, got {end_time:.17g}"
                )
            if write_interval != 1_000_000:
                failures.append(
                    f"performance seed requires writeInterval=1000000, got {write_interval:.17g}"
                )
            runtime_modifiable = parse_single_word(control, "runTimeModifiable")
            if runtime_modifiable not in ("false", "no", "off"):
                failures.append("performance seed requires runTimeModifiable false")
    except OSError as error:
        failures.append(str(error))
        start_time = 0.0
    except ValueError as error:
        failures.append(str(error))
        start_time = 0.0

    try:
        interval = parse_single_scalar(coupling, "couplingInterval")
        if interval <= 0.0:
            failures.append("couplingInterval must be positive")
        if require_performance_seed and interval != 1.0e-6:
            failures.append(
                f"performance seed requires couplingInterval=1e-6, got {interval:.17g}"
            )
        mie_path = resolve_single_mie_table(case_dir, coupling)
        with mie_path.open("r", encoding="utf-8") as stream:
            magic = stream.readline().strip()
            version = stream.readline().strip()
        if magic != "MIE_TABLE" or version != "formatVersion 1":
            failures.append(
                f"mieTable is incompatible with current parser: "
                f"magic={magic!r}, version={version!r}, path={mie_path}"
            )
        if require_performance_seed:
            mie_sha = source_sha256(mie_path)
            if mie_sha != PERFORMANCE_MIE_SHA256:
                failures.append(
                    f"performance Mie table fingerprint mismatch: "
                    f"expected {PERFORMANCE_MIE_SHA256}, got {mie_sha}"
                )
    except (OSError, ValueError) as error:
        failures.append(str(error))

    if require_fresh or require_performance_seed:
        stale = []
        tolerance = max(1.0e-15, abs(start_time)*1.0e-12)
        for child in case_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                if float(child.name) > start_time + tolerance:
                    stale.append(child.name)
            except ValueError:
                pass
        if stale:
            failures.append(f"fresh case contains nonzero time directories: {sorted(stale)}")
        logs = sorted(path.name for path in case_dir.glob("log.*") if path.is_file())
        if logs:
            failures.append(f"fresh case contains archived logs: {logs}")

    if require_prepared or require_performance_seed:
        try:
            initial_dir = expected_time_directory(case_dir, start_time)
            state = initial_dir / "thermalExchangeState"
            state_text = strip_comments(state.read_text(encoding="utf-8"))
            required_state_patterns = (
                ("formatVersion", r"\bformatVersion\s+2\s*;"),
                ("initialState", r"\binitialState\s+true\s*;"),
                ("exchangeSequence", r'\bexchangeSequence\s+"?0"?\s*;'),
            )
            for key, pattern in required_state_patterns:
                entry_count = len(re.findall(rf"\b{key}\s+[^;]*;", state_text))
                valid_count = len(re.findall(pattern, state_text))
                if entry_count != 1 or valid_count != 1:
                    failures.append(
                        f"prepared state requires exactly one valid {key} entry, "
                        f"found {entry_count} total/{valid_count} valid"
                    )
            completed = parse_single_scalar(state, "completedSimulationTimeS")
            if not math.isclose(
                completed, start_time, rel_tol=0.0,
                abs_tol=max(1.0e-15, abs(start_time)*1.0e-12),
            ):
                failures.append(
                    f"prepared state completedSimulationTimeS must equal startTime {start_time:.17g}"
                )
            hash_entries = re.findall(r"\bcouplingConfigurationSha1\s+[^;]*;", state_text)
            hashes = re.findall(r'\bcouplingConfigurationSha1\s+"([0-9a-f]{40})"\s*;', state_text)
            if len(hash_entries) != 1 or len(hashes) != 1:
                failures.append(
                    f"prepared state requires one lowercase SHA-1, "
                    f"found {len(hash_entries)} total/{len(hashes)} valid"
                )
            for key in ("fluidMeshTopologySha1", "solidMeshTopologySha1"):
                entries = re.findall(rf"\b{key}\s+[^;]*;", state_text)
                values = re.findall(rf'\b{key}\s+"([0-9a-f]{{40}})"\s*;', state_text)
                if len(entries) != 1 or len(values) != 1:
                    failures.append(f"prepared state requires one valid {key}")
            coupling_text = strip_comments(coupling.read_text(encoding="utf-8"))
            if "solidSolidInterfaces" in coupling_text:
                auxiliary_entries = re.findall(
                    r"\bauxiliarySolidMeshTopologySha1\s+[^;]*;", state_text,
                )
                auxiliary = re.findall(
                    r'\bauxiliarySolidMeshTopologySha1\s+"([0-9a-f]{40})"\s*;',
                    state_text,
                )
                if len(auxiliary_entries) != 1 or len(auxiliary) != 1:
                    failures.append("three-region prepared state requires auxiliary-solid SHA-1")
            if (initial_dir / "thermalExchangeManifest").exists():
                failures.append("fresh prepared state must not contain a thermal exchange manifest")
        except OSError as error:
            failures.append(str(error))
        except ValueError as error:
            failures.append(str(error))

    if require_performance_seed:
        try:
            initial_dir = expected_time_directory(case_dir, start_time)
            for name, expected_sha in PERFORMANCE_SEED_SHA256.items():
                path = initial_dir / name
                actual_sha = source_sha256(path)
                if actual_sha != expected_sha:
                    failures.append(
                        f"performance seed {name} fingerprint mismatch: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
            particle_lines = (
                initial_dir / "gpuResidentStrictParticles.dat"
            ).read_text(encoding="utf-8").splitlines()
            particle_header = particle_lines[0]
            if particle_header != "UGKP_PARTICLES_SCHEMA3 28652":
                failures.append(
                    f"performance seed particle header mismatch: {particle_header!r}"
                )
            if len(particle_lines) != 28_653:
                failures.append(
                    f"performance seed requires 28652 particle rows, got {len(particle_lines) - 1}"
                )
            residual_lines = (
                initial_dir / "gpuResidentStrictSourceResidual.dat"
            ).read_text(encoding="utf-8").splitlines()
            if residual_lines[0] != "UGKP_SOURCE_RESIDUAL_SCHEMA1 48":
                failures.append("performance seed requires the configured 48 source slots")
            if len(residual_lines) != 49 or any(
                len(row.split()) != 2 or float(row.split()[1]) != 0.0
                for row in residual_lines[1:]
            ):
                failures.append("performance seed source-slot residual masses must all be zero")
        except (OSError, ValueError, IndexError) as error:
            failures.append(str(error))

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-pool", action="store_true")
    parser.add_argument("--require-relax", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--expected-kappa", choices=("fallback", "explicit"))
    parser.add_argument("--require-fresh", action="store_true")
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-performance-seed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.require_performance_seed and args.case is None:
        parser.error("--require-performance-seed requires --case")
    failures = source_contracts(args.require_pool, args.require_relax)
    if args.case is not None:
        failures.extend(
            case_contracts(
                args.case,
                args.expected_kappa,
                args.require_fresh,
                args.require_prepared,
                args.require_performance_seed,
            )
        )
    if failures:
        print("UGKP_CHT UGKP V2 contract failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: UGKP_CHT UGKP V2 contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
