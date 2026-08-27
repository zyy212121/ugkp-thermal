                                                              

                                                                             
                                                                      
                                             
   

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CREATE_FIELDS = (
    ("std::int32_t", "nCells"),
    ("std::int32_t", "nFaces"),
    ("std::int32_t", "nInternalFaces"),
    ("std::int32_t", "nCellPlanes"),
    ("std::int32_t", "particleCapacity"),
    ("std::int32_t", "maxFaceWalkHops"),
    ("double", "injectionParcelMass"),
    ("std::uint64_t", "rngSeed"),
    ("double", "gammaGas"),
    ("double", "Rgas"),
    ("double", "rhoSolid"),
    ("std::int32_t", "solveParticleTemperature"),
    ("double", "gasMu"),
    ("double", "gasPr"),
    ("std::int32_t", "dragModelId"),
    ("std::int32_t", "particleGasHeatTransferModelId"),
    ("double", "dragResidualRe"),
    ("double", "gravityX"),
    ("double", "gravityY"),
    ("double", "gravityZ"),
    ("double", "particleDiameterFallback"),
    ("double", "particleDiameterMin"),
    ("double", "particleDiameterMax"),
    ("double", "particleDiameterSigma"),
    ("double", "injectionTheta"),
    ("double", "rhoMin"),
    ("double", "TgasMin"),
    ("double", "epsSMin"),
    ("double", "thetaMin"),
    ("double", "TpMin"),
    ("double", "TpMax"),
    ("std::int32_t", "collisionalPressureEnabled"),
    ("double", "collisionalRestitution"),
    ("double", "pressureKickFraction"),
    ("std::int32_t", "jammingPressureEnabled"),
    ("double", "packingFraction"),
    ("std::int32_t", "packingProjectionIterations"),
    ("std::int32_t", "gasFluxScheme"),
    ("std::int32_t", "gasReconstruction"),
    ("std::int32_t", "gasLimiter"),
    ("std::int32_t", "gasTimeIntegrator"),
    ("std::int32_t", "gasRobustFallback"),
    ("std::int32_t", "turbulenceModel"),
    ("double", "lesDeltaCoeff"),
    ("double", "turbulentPrandtl"),
    ("double", "waleCw"),
    ("double", "smagorinskyCs"),
    ("double", "maxDiffusionNumber"),
    ("std::int32_t", "csrCellLocalPathEnabled"),
    ("std::int32_t", "csrHeavyReductionMode"),
    ("std::int32_t", "csrHeavyAutoInterval"),
    ("std::int32_t", "particleBlockThreads"),
    ("std::int32_t", "reductionBlockThreads"),
    ("std::int32_t", "csrWarpAggregatedBinning"),
)

CREATE_ABI_FIELDS = CREATE_FIELDS

GAS_NUMERICS_ARGUMENTS = (
    "gasFluxScheme",
    "gasReconstruction",
    "gasLimiter",
    "gasTimeIntegrator",
    "gasRobustFallback",
    "gasTurbulenceModel",
)


def _delimited_after(
    source: str,
    marker: str,
    opening: str,
    closing: str,
    start: int = 0,
) -> tuple[str, int]:
    marker_at = source.index(marker, start)
    begin = source.index(opening, marker_at + len(marker))
    depth = 0
    for index in range(begin, len(source)):
        char = source[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return source[begin + 1:index], index + 1
    raise AssertionError(f"unclosed {opening!r} after {marker!r}")


def _split_top_level(source: str) -> list[str]:
    values: list[str] = []
    begin = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(source):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif (
            char == ","
            and round_depth == 0
            and square_depth == 0
            and brace_depth == 0
        ):
            values.append(re.sub(r"\s+", " ", source[begin:index]).strip())
            begin = index + 1
    tail = re.sub(r"\s+", " ", source[begin:]).strip()
    if tail:
        values.append(tail)
    return values


def _create_signature(source: str) -> list[tuple[str, str]]:
    parameters, _ = _delimited_after(
        source,
        "ugkwpGpuResidentStrictCreate",
        "(",
        ")",
    )
    result: list[tuple[str, str]] = []
    for declaration in _split_top_level(parameters):
        match = re.fullmatch(r"(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)", declaration)
        if match is None:
            raise AssertionError(f"cannot parse create parameter: {declaration!r}")
        parameter_type = re.sub(r"\s+", " ", match.group("type")).strip()
        parameter_type = parameter_type.replace(" *", "*").replace("* ", "*")
        parameter_type = {
            "int": "std::int32_t",
            "unsigned long long": "std::uint64_t",
        }.get(parameter_type, parameter_type)
        result.append((parameter_type, match.group("name")))
    return result


def _assert_contiguous_subsequence(
    testcase: unittest.TestCase,
    actual: list[str],
    expected: tuple[str, ...],
) -> None:
    width = len(expected)
    testcase.assertTrue(
        any(tuple(actual[index:index + width]) == expected
            for index in range(len(actual) - width + 1)),
        f"{expected!r} is not contiguous in {actual!r}",
    )


class UGKPNumericsConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.create = (
            (ROOT / "readGpuGasConfiguration.H").read_text()
            + "\n"
            + (ROOT / "createFields.H").read_text()
        )
        cls.protocol = (ROOT / "gpu/GpuBackendProtocol.H").read_text()
        cls.api = (ROOT / "gpu/GpuBackendApi.H").read_text()
        cls.client = (ROOT / "gpu/GpuBackendClient.C").read_text()
        cls.wrapper = (ROOT / "gpu/GpuResidentStrict.H").read_text()
        cls.server = (ROOT / "private_backend/GpuBackendServer.C").read_text()
        cls.cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()

    def test_runtime_dictionary_and_supported_names(self) -> None:
        self.assertIn('fvSchemesDict.found("fluxScheme")', self.create)
        self.assertNotIn('subDict("gasNumerics")', self.create)
        for name in (
            "rusanovTadmor",
            "hllKurganov",
            "hlle",
            "hllc",
            "roe",
            "hllem",
            "hllcAdc",
            "slau2",
            "slau2_2",
            "firstOrder",
            "MUSCL",
            "energyLimitedLinear",
            "none",
            "barthJespersen",
            "venkatakrishnan",
            "Euler",
            "SSPRK2",
            "SSPRK3",
        ):
            self.assertIn(f'"{name}"', self.create)

    def test_runtime_names_map_to_protocol_enum_values(self) -> None:
        enum_expectations = {
            "GasFluxScheme": {
                "RusanovTadmor": 1,
                "HllKurganov": 2,
                "Hlle": 3,
                "Hllc": 4,
                "Roe": 5,
                "Hllem": 6,
                "HllcAdc": 7,
                "Slau2": 8,
                "Slau2_2": 9,
            },
            "GasReconstruction": {
                "firstOrder": 0,
                "MUSCL": 1,
                "OpenFoamEnergyLimitedLinear": 2,
            },
            "GasLimiter": {
                "none": 0,
                "barthJespersen": 1,
                "venkatakrishnan": 2,
            },
            "GasTimeIntegrator": {"Euler": 1, "SSPRK2": 2, "SSPRK3": 3},
        }
        for enum_name, expected in enum_expectations.items():
            match = re.search(
                rf"enum class {enum_name}\s*:[^{{]+"
                rf"\{{(?P<body>.*?)\}};",
                self.protocol,
                flags=re.S,
            )
            self.assertIsNotNone(match, enum_name)
            actual = {
                name: int(value)
                for name, value in re.findall(
                    r"([A-Za-z_]\w*)\s*=\s*(\d+)",
                    match.group("body"),
                )
            }
            self.assertEqual(actual, expected)

        mapping_contracts = (
            (r'gasFluxSchemeName == "rusanovTadmor".*?gasFluxScheme = 1;',),
            (r'gasFluxSchemeName == "hllKurganov".*?gasFluxScheme = 2;',),
            (r'gasFluxSchemeName == "hlle".*?gasFluxScheme = 3;',),
            (r'gasFluxSchemeName == "hllc".*?gasFluxScheme = 4;',),
            (r'gasFluxSchemeName == "roe".*?gasFluxScheme = 5;',),
            (r'gasFluxSchemeName == "hllem".*?gasFluxScheme = 6;',),
            (r'gasFluxSchemeName == "hllcAdc".*?gasFluxScheme = 7;',),
            (r'gasFluxSchemeName == "slau2".*?gasFluxScheme = 8;',),
            (r'gasReconstructionName == "firstOrder".*?gasReconstruction = 0;',),
            (r'gasReconstructionName == "MUSCL".*?gasReconstruction = 1;',),
            (
                r'gasReconstructionName == "energyLimitedLinear".*?'
                r'gasReconstruction = 2;',
            ),
            (r'gasLimiterName == "none".*?gasLimiter = 0;',),
            (r'gasLimiterName == "barthJespersen".*?gasLimiter = 1;',),
            (r'gasLimiterName == "venkatakrishnan".*?gasLimiter = 2;',),
            (r'gasTimeIntegratorName == "Euler".*?gasTimeIntegrator = 1;',),
            (r'gasTimeIntegratorName == "SSPRK2".*?gasTimeIntegrator = 2;',),
            (r'gasTimeIntegratorName == "SSPRK3".*?gasTimeIntegrator = 3;',),
        )
        for (pattern,) in mapping_contracts:
            with self.subTest(pattern=pattern):
                self.assertRegex(
                    self.create,
                    re.compile(pattern, flags=re.S),
                )

    def test_create_payload_field_order_and_size(self) -> None:
        match = re.search(
            r"struct CreateArgs\s*\{(?P<body>.*?)\n\};",
            self.protocol,
            flags=re.S,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        actual = re.findall(
            r"(std::int32_t|std::uint64_t|double)\s+([A-Za-z_]\w*)\s*;",
            body,
        )
        self.assertEqual(actual, list(CREATE_FIELDS))
        self.assertIn(
            'static_assert(sizeof(CreateArgs) == 352, "unexpected create payload ABI")',
            self.protocol,
        )

    def test_c_abi_create_signatures_match_protocol_exactly(self) -> None:
        expected = list(CREATE_ABI_FIELDS) + [("void**", "handle")]
        layers = {
            "public API": self.api,
            "OpenFOAM wrapper": self.wrapper,
            "IPC client definition": self.client,
            "CUDA definition": self.cuda,
        }
        for layer, source in layers.items():
            with self.subTest(layer=layer):
                self.assertEqual(_create_signature(source), expected)

    def test_client_payload_and_server_forwarding_are_exact(self) -> None:
        payload, _ = _delimited_after(
            self.client,
            "const CreateArgs args",
            "{",
            "}",
        )
        self.assertEqual(
            _split_top_level(payload),
            [name for _, name in CREATE_FIELDS],
        )

        forwarding, _ = _delimited_after(
            self.server,
            "const int status = ugkwpGpuResidentStrictCreate",
            "(",
            ")",
        )
        self.assertEqual(
            _split_top_level(forwarding),
            [f"a.{name}" for _, name in CREATE_ABI_FIELDS]
            + ["&state.backend"],
        )

    def test_openfoam_wrapper_forwards_gas_numerics_contiguously(self) -> None:
        wrapper_call_marker = "const int createRc = ugkwpGpuResidentStrictCreate"
        search_from = 0
        expected = (
            "int(gasFluxScheme)",
            "int(gasReconstruction)",
            "int(gasLimiter)",
            "int(gasTimeIntegrator)",
            "gasRobustFallback ? 1 : 0",
            "int(gasTurbulenceModel)",
        )
        for call_number in (1, 2):
            arguments, search_from = _delimited_after(
                self.wrapper,
                wrapper_call_marker,
                "(",
                ")",
                search_from,
            )
            with self.subTest(call=call_number):
                _assert_contiguous_subsequence(
                    self,
                    _split_top_level(arguments),
                    expected,
                )

        for target in ("initialiseGasBase", "initialiseBase"):
            arguments, _ = _delimited_after(
                self.wrapper,
                f"\n        {target}\n        ",
                "(",
                ")",
            )
            with self.subTest(wrapper_target=target):
                _assert_contiguous_subsequence(
                    self,
                    _split_top_level(arguments),
                    GAS_NUMERICS_ARGUMENTS,
                )

    def test_solver_forwards_gas_numerics_to_both_modes(self) -> None:
        search_from = 0
        for mode in ("pure gas", "particle"):
            arguments, search_from = _delimited_after(
                (ROOT / "diluteUgkwpFoam.C").read_text(),
                "resident.initialise",
                "(",
                ")",
                search_from,
            )
            with self.subTest(mode=mode):
                _assert_contiguous_subsequence(
                    self,
                    _split_top_level(arguments),
                    GAS_NUMERICS_ARGUMENTS,
                )

    def test_validation_is_present_at_every_untrusted_boundary(self) -> None:
        layers = {
            "OpenFOAM dictionary": self.create,
            "IPC client": self.client,
            "IPC server": self.server,
            "CUDA C ABI": self.cuda,
        }
        for field in (
            "gasFluxScheme",
            "gasReconstruction",
            "gasLimiter",
            "gasTimeIntegrator",
            "gasRobustFallback",
        ):
            for layer, source in layers.items():
                with self.subTest(field=field, layer=layer):
                    self.assertIn(field, source)
        for layer, source in (
            ("IPC client", self.client),
            ("IPC server", self.server),
            ("CUDA C ABI", self.cuda),
        ):
            with self.subTest(layer=layer):
                self.assertRegex(
                    source,
                    r"gasRobustFallback\s*!=\s*0"
                    r".*?gasRobustFallback\s*!=\s*1",
                )
        self.assertNotIn('"entropyFixCoefficient"', self.create)
        self.assertNotIn("gasEntropyFixCoefficient", self.create)

    def test_old_gks_model_is_not_an_active_cuda_selector(self) -> None:
        self.assertNotIn("gasKineticFluxModel", self.create)
        self.assertNotIn("shockRelaxationCoefficient", self.create)
        self.assertNotRegex(
            self.cuda,
            r"gasKineticFluxModel\s*!=\s*1",
        )

    def test_active_examples_use_runtime_riemann_configuration(self) -> None:
        if not (ROOT / "examples").is_dir():
            self.skipTest("clean UGKP source package has no legacy examples")
        roots = (
            ROOT / "examples/gks_flux_validation",
            ROOT / "examples/les_validation",
            ROOT / "examples/pressuretest/ugkp.5_purephase",
            ROOT / "examples/pressuretest/ugkp.6_purephase_0p01",
            ROOT / "examples/pressuretest/ugkp.6_purephase_0p01_euler",
            ROOT
            / "examples/pressuretest"
            / "ugkp.6_purephase_0p01_hllc_firstorder_euler",
            ROOT / "examples/twophaseflux",
        )
        dictionaries = [
            path
            for root in roots
            for path in root.rglob("gksProperties")
            if "source_switch_backups" not in path.parts
        ]
        if not dictionaries:
            self.skipTest(
                "clean UGKP package has no legacy numerical-validation cases"
            )
        for path in dictionaries:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8", errors="replace")
                self.assertIn("gasNumerics", source)
                self.assertNotIn("gasKineticFlux", source)

    def test_pure_gas_validation_launches_ugkp(self) -> None:
        validation_root = ROOT / "examples/gks_flux_validation"
        if not validation_root.is_dir():
            self.skipTest("clean UGKP source package has no legacy examples")
        launchers = list(validation_root.rglob("Allrun"))
        controls = list(validation_root.rglob("controlDict"))
        self.assertTrue(launchers)
        self.assertTrue(controls)
        for path in launchers + controls:
            source = path.read_text(encoding="utf-8", errors="replace")
            if "diluteUgkwpFoam_" in source:
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertIn("diluteUgkwpFoam_UGKP", source)
                    self.assertNotIn("legacySolverName", source)

    def test_short_pure_gas_gate_matches_first_cpu_write(self) -> None:
        short_case = ROOT / "examples/pressuretest/ugkp.6_purephase_0p01"
        if not short_case.is_dir():
            short_case = (
                ROOT
                / "examples/pressuretest/tempt/ugkp.6_purephase_0p01"
            )
        if not short_case.is_dir():
            self.skipTest("clean UGKP source package has no legacy examples")
        control = (short_case / "system/controlDict").read_text()
        self.assertRegex(control, r"\bendTime\s+0\.01\s*;")
        self.assertRegex(control, r"\bwriteInterval\s+0\.01\s*;")
        cpu_write = (
            ROOT / "examples/pressuretest/cpu_rhoPimpleFoam/0.0099999498"
        )
        self.assertTrue(cpu_write.is_dir())
        for field in ("p", "rho", "T", "U"):
            self.assertTrue((cpu_write / field).is_file())


if __name__ == "__main__":
    unittest.main()
