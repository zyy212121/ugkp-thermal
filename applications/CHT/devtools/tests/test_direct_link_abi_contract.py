                                                                 

                                                                            
                                                                             
                                                                          
   

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    ("void**", "handle"),
)


TWO_PHASE_CREATE_ARGUMENTS = (
    "int(nCells)",
    "int(nFaces)",
    "int(nInternalFaces)",
    "int(nCellPlanes)",
    "int(particleCapacity)",
    "int(maxFaceWalkHops)",
    "injectionParcelMass",
    "rngSeed",
    "gammaGas",
    "RgasValue",
    "rhoSolid",
    "solveParticleTemperature ? 1 : 0",
    "gasMu",
    "gasPr",
    "int(dragModelId)",
    "int(particleGasHeatTransferModelId)",
    "dragResidualRe",
    "gravityX",
    "gravityY",
    "gravityZ",
    "particleDiameterFallback",
    "particleDiameterMin",
    "particleDiameterMax",
    "particleDiameterSigma",
    "injectionTheta",
    "rhoMin",
    "TgasMin",
    "epsSMin",
    "thetaMin",
    "TpMin",
    "TpMax",
    "collisionalPressureEnabled ? 1 : 0",
    "collisionalRestitution",
    "pressureKickFraction",
    "jammingPressureEnabled ? 1 : 0",
    "packingFraction",
    "int(packingProjectionIterations)",
    "int(gasFluxScheme)",
    "int(gasReconstruction)",
    "int(gasLimiter)",
    "int(gasTimeIntegrator)",
    "gasRobustFallback ? 1 : 0",
    "int(gasTurbulenceModel)",
    "lesDeltaCoeff",
    "turbulentPrandtl",
    "waleCw",
    "smagorinskyCs",
    "maxDiffusionNumber",
    "csrCellLocalPathEnabled ? 1 : 0",
    "csrHeavyReductionMode",
    "int(scheduling.heavyReductionAutoInterval)",
    "int(scheduling.particleBlockThreads)",
    "int(scheduling.reductionBlockThreads)",
    "csrWarpAggregatedBinning ? 1 : 0",
    "&raw",
)


PURE_GAS_CREATE_ARGUMENTS = (
    "int(nCells)",
    "int(nFaces)",
    "int(nInternalFaces)",
    "int(nCellPlanes)",
    "0",
    "int(maxFaceWalkHops)",
    "scalar(1)",
    "rngSeed",
    "gammaGas",
    "RgasValue",
    "scalar(1)",
    "0",
    "gasMu",
    "gasPr",
    "int(dragModelId)",
    "0",
    "dragResidualRe",
    "gravityX",
    "gravityY",
    "gravityZ",
    "scalar(1)",
    "scalar(1)",
    "scalar(1)",
    "scalar(0)",
    "scalar(0)",
    "rhoMin",
    "TgasMin",
    "scalar(0)",
    "scalar(0)",
    "scalar(300)",
    "scalar(300)",
    "0",
    "scalar(0.9)",
    "scalar(0.25)",
    "0",
    "scalar(0.63)",
    "20",
    "int(gasFluxScheme)",
    "int(gasReconstruction)",
    "int(gasLimiter)",
    "int(gasTimeIntegrator)",
    "gasRobustFallback ? 1 : 0",
    "int(gasTurbulenceModel)",
    "lesDeltaCoeff",
    "turbulentPrandtl",
    "waleCw",
    "smagorinskyCs",
    "maxDiffusionNumber",
    "scheduling.csrCellLocalPath ? 1 : 0",
    "int(scheduling.heavyReductionMode)",
    "int(scheduling.heavyReductionAutoInterval)",
    "int(scheduling.particleBlockThreads)",
    "int(scheduling.reductionBlockThreads)",
    "scheduling.warpAggregatedBinning ? 1 : 0",
    "&raw",
)


PARTICLE_STUCK_FIELDS = (
    ("void*", "handle"),
    ("std::int32_t", "nFaces"),
    ("const unsigned char*", "candidateFaceMask"),
    ("double", "sommerfeldThreshold"),
    ("std::int32_t", "heatTransferEnabled"),
    ("double", "maximumCoverage"),
    ("double", "depositionHeatTransferEfficiency"),
    ("double", "reflectionHeatTransferEfficiency"),
    ("double", "adhesionEnergyScale"),
    ("double", "contactAngleDegree"),
    ("std::int32_t", "wallTransientResistance"),
    ("std::int32_t", "nonlinearIterations"),
    ("double", "meltingTemperatureK"),
    ("double", "mushyRangeK"),
    ("double", "latentHeatJkg"),
    ("double", "solidDensityKgM3"),
    ("double", "solidSpecificHeatJkgK"),
    ("double", "solidThermalConductivityWmK"),
    ("double", "pinningThicknessFraction"),
    ("double", "interfaceResistanceM2KW"),
)


PARTICLE_STUCK_ARGUMENTS = (
    "handle_",
    "int(mesh.nFaces())",
    "candidateFaceMask.begin()",
    "sommerfeldThreshold",
    "heatTransfer ? 1 : 0",
    "maximumCoverage",
    "depositionHeatTransferEfficiency",
    "reflectionHeatTransferEfficiency",
    "adhesionEnergyScale",
    "contactAngleDegree",
    "wallTransientResistance ? 1 : 0",
    "int(nonlinearIterations)",
    "meltingTemperatureK",
    "mushyRangeK",
    "latentHeatJkg",
    "solidDensityKgM3",
    "solidSpecificHeatJkgK",
    "solidThermalConductivityWmK",
    "pinningThicknessFraction",
    "interfaceResistanceM2KW",
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
        match = re.fullmatch(
            r"(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)",
            declaration,
        )
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


def _named_signature(source: str, function_name: str) -> list[tuple[str, str]]:
    parameters, _ = _delimited_after(source, function_name, "(", ")")
    result: list[tuple[str, str]] = []
    for declaration in _split_top_level(parameters):
        match = re.fullmatch(
            r"(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)",
            declaration,
        )
        if match is None:
            raise AssertionError(f"cannot parse parameter: {declaration!r}")
        parameter_type = re.sub(r"\s+", " ", match.group("type")).strip()
        parameter_type = parameter_type.replace(" *", "*").replace("* ", "*")
        parameter_type = {
            "int": "std::int32_t",
            "unsigned long long": "std::uint64_t",
        }.get(parameter_type, parameter_type)
        result.append((parameter_type, match.group("name")))
    return result


def _assert_fields_exact(
    testcase: unittest.TestCase,
    actual: list[tuple[str, str]],
) -> None:
    testcase.assertEqual(len(actual), len(CREATE_FIELDS))
    for index, (observed, expected) in enumerate(zip(actual, CREATE_FIELDS)):
        with testcase.subTest(index=index, field=expected[1]):
            testcase.assertEqual(observed, expected)


def _assert_call_arguments_exact(
    testcase: unittest.TestCase,
    actual: list[str],
    expected: tuple[str, ...],
) -> None:
    testcase.assertEqual(len(expected), len(CREATE_FIELDS))
    testcase.assertEqual(len(actual), len(CREATE_FIELDS))
    for index, ((_, field), observed, required) in enumerate(
        zip(CREATE_FIELDS, actual, expected)
    ):
        with testcase.subTest(index=index, field=field):
            testcase.assertEqual(observed, required)


class DirectLinkCreateAbiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "gpu/GpuResidentStrict.H").read_text(
            encoding="utf-8"
        )
        cls.cuda = (ROOT / "gpu/GpuResidentStrict.cu").read_text(
            encoding="utf-8"
        )

    def test_header_and_cuda_create_signatures_match_field_for_field(self) -> None:
        _assert_fields_exact(self, _create_signature(self.header))
        _assert_fields_exact(self, _create_signature(self.cuda))

    def test_two_phase_wrapper_create_call_matches_every_abi_field(self) -> None:
        start = self.header.index("void initialiseBase")
        arguments, _ = _delimited_after(
            self.header,
            "const int createRc = ugkwpGpuResidentStrictCreate",
            "(",
            ")",
            start,
        )
        actual = _split_top_level(arguments)
        _assert_call_arguments_exact(self, actual, TWO_PHASE_CREATE_ARGUMENTS)

        packing = [field for _, field in CREATE_FIELDS].index("packingFraction")
        self.assertEqual(
            actual[packing:packing + 2],
            ["packingFraction", "int(packingProjectionIterations)"],
        )

    def test_pure_gas_wrapper_create_call_uses_projection_default_twenty(self) -> None:
        start = self.header.index("void initialiseGasBase")
        arguments, _ = _delimited_after(
            self.header,
            "const int createRc = ugkwpGpuResidentStrictCreate",
            "(",
            ")",
            start,
        )
        actual = _split_top_level(arguments)
        _assert_call_arguments_exact(self, actual, PURE_GAS_CREATE_ARGUMENTS)

        packing = [field for _, field in CREATE_FIELDS].index("packingFraction")
        self.assertEqual(actual[packing:packing + 2], ["scalar(0.63)", "20"])

    def test_particle_stuck_direct_abi_carries_solidification_parameters(self) -> None:
        api = (ROOT / "gpu/GpuBackendApi.H").read_text(encoding="utf-8")
        function = "ugkwpGpuResidentStrictConfigureParticleStuckModel"
        self.assertEqual(_named_signature(api, function), list(PARTICLE_STUCK_FIELDS))
        self.assertEqual(_named_signature(self.header, function), list(PARTICLE_STUCK_FIELDS))
        self.assertEqual(_named_signature(self.cuda, function), list(PARTICLE_STUCK_FIELDS))

    def test_particle_stuck_wrapper_forwards_solidification_parameters(self) -> None:
        start = self.header.index("void configureParticleStuckModel")
        arguments, _ = _delimited_after(
            self.header,
            "ugkwpGpuResidentStrictConfigureParticleStuckModel",
            "(",
            ")",
            start,
        )
        self.assertEqual(_split_top_level(arguments), list(PARTICLE_STUCK_ARGUMENTS))

    def test_contact_angle_is_host_validated_and_device_uses_cached_cosine(self) -> None:
        self.assertIn('"contactAngleDegree", scalar(145)', self.header)
        self.assertIn("FatalIOErrorInFunction(ugkwpProps)", self.header)
        self.assertIn("contactAngleDegree <= scalar(0)", self.header)
        self.assertIn("contactAngleDegree >= scalar(180)", self.header)
        self.assertIn("double particleWallContactAngleCosine", self.cuda)
        self.assertIn(
            "::cos(contactAngleDegree*M_PI/180.0)",
            self.cuda,
        )
        self.assertEqual(self.cuda.count("s.particleWallContactAngleCosine"), 3)
        self.assertIn("s->particleWallContactAngleCosine", self.cuda)


if __name__ == "__main__":
    unittest.main(verbosity=2)
