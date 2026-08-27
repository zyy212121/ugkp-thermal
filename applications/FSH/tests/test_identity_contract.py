#!/usr/bin/env python3
                                                              

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class UGKPIdentityContractTests(unittest.TestCase):
    def test_public_frontend_identity(self) -> None:
        make_files = read("Make/files")
        build_script = read("tools/build_public_frontend.sh")
        solver = read("diluteUgkwpFoam.C")

        self.assertIn(
            "EXE = $(FOAM_USER_APPBIN)/FSH",
            make_files,
        )
        self.assertIn(
            'frontend="${FOAM_USER_APPBIN}/FSH"',
            build_script,
        )
        self.assertIn("ldd", build_script)
        self.assertIn(
            "#error FSH must be built with UGKWP_USE_CUDA",
            solver,
        )

    def test_private_backend_identity(self) -> None:
        build_script = read("private_backend/build_private_backend.sh")

        self.assertIn('fmad_mode="${FSH_FMAD:-1}"', build_script)
        self.assertIn(
            'backend_bin="${FOAM_USER_APPBIN}/FSHCudaBackend"',
            build_script,
        )
        self.assertIn(
            'frontend_bin="${FOAM_USER_APPBIN}/FSH"',
            build_script,
        )
        self.assertIn("ERROR: FSH_FMAD must be 0 or 1", build_script)

    def test_runtime_backend_selection_identity(self) -> None:
        client = read("gpu/GpuBackendClient.C")

        self.assertIn('std::getenv("FSH_CUDA_BACKEND")', client)
        self.assertEqual(client.count('"FSHCudaBackend"'), 2)

    def test_ipc_protocol_identity(self) -> None:
        protocol = read("gpu/GpuBackendProtocol.H")

        self.assertIn(
            "static constexpr std::uint32_t magic = 0x46534842U;",
            protocol,
        )
        self.assertIn(
            "static constexpr std::uint16_t protocolMajor = 6;",
            protocol,
        )
        self.assertIn(
            "static constexpr std::uint16_t protocolMinor = 0;",
            protocol,
        )
        for operation, number in (
            ("create", 1),
            ("uploadFields", 9),
            ("downloadAndResetParticleWallHeatLedgers", 20),
            ("release", 21),
            ("downloadSst", 23),
        ):
            self.assertIn(f"{operation} = {number}", protocol)
        self.assertIn(
            "response.minor != protocolMinor",
            read("gpu/GpuBackendClient.C"),
        )
        self.assertIn(
            "request.minor!=protocolMinor",
            read("private_backend/GpuBackendServer.C"),
        )

    def test_active_identity_files_do_not_reference_ugkp(self) -> None:
        active_files = (
            "Make/files",
            "tools/build_public_frontend.sh",
            "private_backend/build_private_backend.sh",
            "gpu/GpuBackendClient.C",
            "gpu/GpuBackendProtocol.H",
            "diluteUgkwpFoam.C",
        )
        legacy_tokens = (
            "UGKP_CUDA_BACKEND",
            "UGKP_FMAD",
            "ugkpCudaBackend",
            "legacySolverName",
            "0x47553235",
            '"GU25"',
        )

        for relative_path in active_files:
            contents = read(relative_path)
            for token in legacy_tokens:
                with self.subTest(file=relative_path, token=token):
                    self.assertNotIn(token, contents)


if __name__ == "__main__":
    unittest.main()
