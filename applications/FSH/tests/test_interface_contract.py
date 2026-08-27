#!/usr/bin/env python3
                                                                         

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"required UGKP file is missing: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


ACTIVE_INTERFACE = (
    "createFields.H",
    "diluteUgkwpFoam.C",
    "gpu/GpuBackendApi.H",
    "gpu/GpuBackendClient.C",
    "gpu/GpuBackendProtocol.H",
    "gpu/GpuResidentStrict.H",
    "private_backend/GpuBackendServer.C",
    "private_backend/GpuResidentStrict.cu",
)


class UGKPInterfaceContract(unittest.TestCase):
    def test_ugkpfsh_executable_backend_and_protocol_identity(self) -> None:
        self.assertIn(
            "FSH",
            source("Make/files"),
        )
        private_build = source("private_backend/build_private_backend.sh")
        public_build = source("tools/build_public_frontend.sh")
        client = source("gpu/GpuBackendClient.C")
        protocol = source("gpu/GpuBackendProtocol.H")

        self.assertIn("FSHCudaBackend", private_build)
        self.assertIn("FSH", private_build)
        self.assertIn("FSH", public_build)
        self.assertIn('frontend="${FOAM_USER_APPBIN}/FSH"', public_build)
        self.assertIn("FSH_CUDA_BACKEND", client)
        self.assertIn('"FSHCudaBackend"', client)
        self.assertRegex(protocol, r"0x46534842U\s*;")

        identity_text = "\n".join(
            (
                source("Make/files"),
                private_build,
                public_build,
                client,
                protocol,
                source("createFields.H"),
                source("diluteUgkwpFoam.C"),
                source("gpu/GpuResidentStrict.H"),
            )
        )
        for legacy in (
            "legacySolverName",
            "ugkpCudaBackend",
            "GU25",
        ):
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, identity_text)

    def test_gas_kappa_is_absent_from_active_interface_and_backend(self) -> None:
        offenders = [name for name in ACTIVE_INTERFACE if "gasKappa" in source(name)]
        self.assertFalse(
            offenders,
            "gasKappa remains in the executable chain: " + ", ".join(offenders),
        )

    def test_mu_cp_over_pr_is_the_only_molecular_conductivity_path(self) -> None:
        fields = (
            source("readGpuGasConfiguration.H")
            + "\n"
            + source("createFields.H")
        )
        cuda = source("private_backend/GpuResidentStrict.cu")

        self.assertNotIn('lookupOrDefault<scalar>("gasKappa"', fields)
        self.assertRegex(
            cuda,
            r"return\s+s\.gasMu\s*\*\s*s\.gasCp\s*/\s*s\.gasPrClamped\s*;",
        )
        self.assertIn("molecularGasConductivity(s)", cuda)
        self.assertIn("s.gasPrOneThird", cuda)

    def test_thermodynamic_inputs_are_validated_at_each_entry_layer(self) -> None:
        fields = (
            source("readGpuGasConfiguration.H")
            + "\n"
            + source("createFields.H")
        )
        client = source("gpu/GpuBackendClient.C")
        server = source("private_backend/GpuBackendServer.C")
        cuda = source("private_backend/GpuResidentStrict.cu")

        self.assertIn("gammaG.value() > scalar(5.0/3.0)", fields)
        for text, prefix in ((client, ""), (server, "a."), (cuda, "")):
            with self.subTest(layer=prefix or "C ABI"):
                self.assertRegex(text, rf"{re.escape(prefix)}gammaGas\s*>\s*5\.0/3\.0")
                self.assertRegex(text, rf"{re.escape(prefix)}gasMu\s*<\s*0\.0")
                self.assertRegex(text, rf"{re.escape(prefix)}gasPr\s*<=\s*0\.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
