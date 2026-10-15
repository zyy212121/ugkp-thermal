#!/usr/bin/env python3
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

                                                                            

from source_contract_utils import read_source
from pathlib import Path
import re
import unittest


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')


def source(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"required gasUGKP file is missing: {relative}")
    return read_source(path)


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
    def test_ugkp_executable_backend_and_protocol_identity(self) -> None:
        self.assertIn(
            "$(FOAM_USER_APPBIN)/gasUGKP",
            source("Make/files"),
        )
        private_build = source('../../tool/gasUGKP/build_private_backend.sh')
        public_build = source('../../tool/gasUGKP/build_public_frontend.sh')
        client = source("gpu/GpuBackendClient.C")
        protocol = source("gpu/GpuBackendProtocol.H")

        self.assertIn("gasUGKPCudaBackend", private_build)
        self.assertIn("gasUGKP", private_build)
        self.assertIn("gasUGKP", public_build)
        self.assertIn('frontend="${FOAM_USER_APPBIN}/gasUGKP"', public_build)
        self.assertIn("GAS_UGKP_CUDA_BACKEND", client)
        self.assertIn('"gasUGKPCudaBackend"', client)
        self.assertRegex(protocol, r"0x47554750U\s*;")

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
            "GU26",
            "0x47553236",
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

    def test_frontend_forwards_thermodynamics_while_downstream_layers_remain_owned(self) -> None:
        fields = (
            source("readGpuGasConfiguration.H")
            + "\n"
            + source("createFields.H")
        )
        client = source("gpu/GpuBackendClient.C")
        server = source("private_backend/GpuBackendServer.C")
        cuda = source("private_backend/GpuResidentStrict.cu")

        self.assertNotIn("gammaG.value() > scalar(5.0/3.0)", fields)
        for text, prefix in ((client, ""), (server, "a."), (cuda, "")):
            with self.subTest(layer=prefix or "C ABI"):
                for field in ("gammaGas", "Rgas", "gasMu", "gasPr"):
                    self.assertIn(prefix + field, text)
                self.assertNotRegex(text, rf"{re.escape(prefix)}gammaGas\s*>\s*5\.0/3\.0")
                self.assertNotRegex(text, rf"{re.escape(prefix)}gasMu\s*<\s*0\.0")
                self.assertNotRegex(text, rf"{re.escape(prefix)}gasPr\s*<=\s*0\.0")
        self.assertIn("s->gammaGas = gammaGas;", cuda)
        self.assertIn("s->gasMu = gasMu;", cuda)
        self.assertIn("s->gasPrClamped = gasPr < 1.0e-12 ? 1.0e-12 : gasPr;", cuda)


if __name__ == "__main__":
    unittest.main(verbosity=2)
