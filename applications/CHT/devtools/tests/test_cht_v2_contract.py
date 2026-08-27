from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cht_v2_contract as contract


class SourceBaselineTests(unittest.TestCase):
    @unittest.skip(
        "legacy pre-gasUGKP5 wall-impact fingerprint is superseded by the "
        "shared finite-contact state machine"
    )
    def test_baseline_cht_guards_and_lifecycle_pass(self) -> None:
        self.assertEqual(contract.source_contracts(False, False), [])

    def test_extract_braced_rejects_missing_marker(self) -> None:
        with self.assertRaises(ValueError):
            contract.extract_braced("int value;", "missing")

    @unittest.skip(
        "legacy pre-gasUGKP5 wall-impact fingerprint is superseded by the "
        "shared finite-contact state machine"
    )
    def test_final_v2_fingerprint_and_structure_pass(self) -> None:
        self.assertEqual(contract.source_contracts(True, True), [])


class CaseContractTests(unittest.TestCase):
    def make_case(self, parcel_mass: str = "5e-9", kappa: str = "-1") -> Path:
        root = Path(self.addCleanupDirectory())
        (root / "constant").mkdir()
        (root / "system").mkdir()
        (root / "0").mkdir()
        mie = root / "constant" / "mie.dat"
        mie.write_text("MIE_TABLE\nformatVersion 1\n", encoding="utf-8")
        (root / "constant" / "ugkwpProperties").write_text(
            f"parcelMass {parcel_mass};\n", encoding="utf-8"
        )
        (root / "constant" / "gksProperties").write_text(
            f"gasKappa {kappa};\n", encoding="utf-8"
        )
        (root / "constant" / "solidThermalCouplingProperties").write_text(
            f'solidThermalCoupling {{ couplingInterval 1e-6; radiation {{ mieTable "{mie}"; }} }}\n',
            encoding="utf-8",
        )
        (root / "system" / "controlDict").write_text(
            "startFrom startTime; startTime 0; adjustTimeStep false; writePrecision 17;\n",
            encoding="utf-8",
        )
        (root / "0" / "thermalExchangeState").write_text(
            "formatVersion 2;\n"
            "initialState true;\n"
            'exchangeSequence "0";\n'
            "completedSimulationTimeS 0;\n"
            'fluidMeshTopologySha1 "0123456789abcdef0123456789abcdef01234567";\n'
            'solidMeshTopologySha1 "123456789abcdef0123456789abcdef012345678";\n'
            'couplingConfigurationSha1 "23456789abcdef0123456789abcdef0123456789";\n',
            encoding="utf-8",
        )
        return root

    def addCleanupDirectory(self) -> str:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary.name

    def test_safe_fallback_case_passes(self) -> None:
        case = self.make_case()
        self.assertEqual(contract.case_contracts(case, "fallback", True, True), [])

    def test_equal_limit_is_rejected(self) -> None:
        case = self.make_case(parcel_mass="5e-8")
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("unsafe parcelMass" in failure for failure in failures))

    def test_production_scale_value_is_rejected(self) -> None:
        case = self.make_case(parcel_mass="1e-6")
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("unsafe parcelMass" in failure for failure in failures))

    def test_duplicate_kappa_is_rejected(self) -> None:
        case = self.make_case()
        with (case / "constant" / "gksProperties").open("a", encoding="utf-8") as stream:
            stream.write("gasKappa 0.024;\n")
        failures = contract.case_contracts(case, None, True, True)
        self.assertTrue(any("exactly one gasKappa" in failure for failure in failures))

    def test_explicit_requires_exact_configured_value(self) -> None:
        case = self.make_case(kappa="0.024")
        self.assertEqual(contract.case_contracts(case, "explicit", True, True), [])

    def test_stale_time_is_rejected_for_fresh_case(self) -> None:
        case = self.make_case()
        (case / "2e-06").mkdir()
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("nonzero time directories" in failure for failure in failures))

    def test_nonzero_start_time_is_the_fresh_seed(self) -> None:
        case = self.make_case()
        (case / "system" / "controlDict").write_text(
            "startFrom startTime; startTime 0.1; adjustTimeStep false; writePrecision 17;\n",
            encoding="utf-8",
        )
        (case / "0").rename(case / "0.1")
        state = case / "0.1" / "thermalExchangeState"
        state.write_text(
            state.read_text(encoding="utf-8").replace(
                "completedSimulationTimeS 0;", "completedSimulationTimeS 0.1;"
            ),
            encoding="utf-8",
        )
        self.assertEqual(contract.case_contracts(case, "fallback", True, True), [])

    def test_performance_seed_rejects_latest_time_start(self) -> None:
        case = self.make_case()
        control = case / "system" / "controlDict"
        control.write_text(
            control.read_text(encoding="utf-8").replace(
                "startFrom startTime;", "startFrom latestTime;"
            ),
            encoding="utf-8",
        )
        failures = contract.case_contracts(
            case, "fallback", False, False, require_performance_seed=True
        )
        self.assertTrue(
            any("must use startFrom startTime" in failure for failure in failures)
        )

    def test_performance_seed_rejects_modified_mie_body(self) -> None:
        case = self.make_case()
        mie = case / "constant" / "mie.dat"
        frozen = (
            contract.ROOT
            / "experiments/assets/mie/alumina_mieTable_smoke_v1.dat"
        )
        if not frozen.is_file():
            self.skipTest("frozen Mie performance asset is not present")
        mie.write_bytes(frozen.read_bytes())
        before = contract.case_contracts(
            case, "fallback", False, False, require_performance_seed=True
        )
        self.assertFalse(any("Mie table fingerprint" in failure for failure in before))
        mie.write_bytes(mie.read_bytes() + b"\n# valid-header body tamper\n")
        after = contract.case_contracts(
            case, "fallback", False, False, require_performance_seed=True
        )
        self.assertTrue(any("Mie table fingerprint" in failure for failure in after))

    def test_duplicate_start_from_is_rejected(self) -> None:
        case = self.make_case()
        with (case / "system" / "controlDict").open("a", encoding="utf-8") as stream:
            stream.write("startFrom latestTime;\n")
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("exactly one startFrom" in failure for failure in failures))

    def test_commented_prepared_state_tokens_do_not_pass(self) -> None:
        case = self.make_case()
        state = case / "0" / "thermalExchangeState"
        state.write_text(
            state.read_text(encoding="utf-8")
            .replace("formatVersion 2;", "// formatVersion 2;")
            .replace("initialState true;", "// initialState true;")
            .replace('exchangeSequence "0";', '// exchangeSequence "0";'),
            encoding="utf-8",
        )
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("valid formatVersion" in failure for failure in failures))
        self.assertTrue(any("valid initialState" in failure for failure in failures))
        self.assertTrue(any("valid exchangeSequence" in failure for failure in failures))

    def test_duplicate_prepared_state_token_is_rejected(self) -> None:
        case = self.make_case()
        with (case / "0" / "thermalExchangeState").open("a", encoding="utf-8") as stream:
            stream.write('exchangeSequence "0";\n')
        failures = contract.case_contracts(case, "fallback", True, True)
        self.assertTrue(any("valid exchangeSequence" in failure for failure in failures))

    def test_conflicting_prepared_state_entries_are_rejected(self) -> None:
        case = self.make_case()
        with (case / "0" / "thermalExchangeState").open("a", encoding="utf-8") as stream:
            stream.write(
                "formatVersion 3;\n"
                "initialState false;\n"
                'exchangeSequence "1";\n'
                'couplingConfigurationSha1 "not-a-sha";\n'
            )
        failures = contract.case_contracts(case, "fallback", True, True)
        for key in ("formatVersion", "initialState", "exchangeSequence", "lowercase SHA-1"):
            self.assertTrue(any(key in failure for failure in failures), key)


if __name__ == "__main__":
    unittest.main()
