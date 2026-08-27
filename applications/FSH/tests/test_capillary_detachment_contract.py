#!/usr/bin/env python3
                                                        

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="strict")


class UGKPCapillaryDetachmentContract(unittest.TestCase):
    def test_temperature_dependent_alumina_model_is_device_visible(self) -> None:
        header = read("thermal/GpuAluminaProperties.H")
        for token in (
            "liquidAluminaProperties",
            "5.77e-4*::exp(9743.0/temperatureK)",
            "0.632 - 2.310e-5",
            "1000.0*(2.917 - 1.228e-4",
            "aluminaSpecificHeatJkgK",
            "aluminaSpecificEnthalpyJkg",
            "aluminaTemperatureFromSpecificEnthalpyK",
            "aluminaThermalConductivityWmK = 2.70",
        ):
            with self.subTest(token=token):
                self.assertIn(token, header)

    def test_adhesion_scale_crosses_protocol_without_damping(self) -> None:
        active = "\n".join(
            read(path)
            for path in (
                "gpu/GpuBackendProtocol.H",
                "gpu/GpuBackendApi.H",
                "gpu/GpuBackendClient.C",
                "private_backend/GpuBackendServer.C",
                "gpu/GpuResidentStrict.H",
                "private_backend/GpuResidentStrict.cu",
            )
        )
        self.assertGreaterEqual(active.count("adhesionEnergyScale"), 8)
        self.assertNotIn("mechanicalDampingScale", active)
        self.assertNotIn("decayWallMechanicalEnergy", active)
        self.assertIn("sizeof(ParticleStuckConfigArgs) == 152", active)
        self.assertGreaterEqual(active.count("contactAngleDegree"), 8)
        self.assertNotIn("hcImpact", active)

    def test_contact_angle_crosses_separated_protocol_in_tail_position(self) -> None:
        protocol = read("gpu/GpuBackendProtocol.H")
        client = read("gpu/GpuBackendClient.C")
        server = read("private_backend/GpuBackendServer.C")
        api = read("gpu/GpuBackendApi.H")
        self.assertIn("protocolMinor = 0", protocol)
        self.assertIn("double contactAngleDegree;", protocol)
        self.assertLess(
            protocol.index("double wallConductivityWmK;"),
            protocol.index("double contactAngleDegree;"),
        )
        self.assertIn("a.contactAngleDegree", server)
        self.assertIn("contactAngleDegree", client)
        self.assertIn("double contactAngleDegree", api)
        self.assertIn("double meltingTemperatureK;", protocol)
        self.assertIn("double interfaceResistanceM2KW;", protocol)

    def test_all_wall_physics_use_the_unified_alumina_properties(self) -> None:
        sommerfeld = read("thermal/GpuSommerfeldSticking.H")
        heat = read("thermal/GpuParticleWallContactHeat.H")
        capillary = read("thermal/GpuCapillaryDetachment.H")
        for source in (sommerfeld, capillary):
            self.assertIn("liquidAluminaProperties", source)
        self.assertIn("aluminaSpecificEnthalpyJkg", heat)
        for removed in ("rhoP", "cpP", "muP", "sigmaP", "kP"):
            self.assertNotIn(removed, heat)

    def test_gas_particle_heat_uses_temperature_dependent_enthalpy(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        protocol = read("gpu/GpuBackendProtocol.H")
        fields = read("createFields.H")
        self.assertIn("particleSpecificHeatDevice(tpAvg)", cuda)
        self.assertIn("particleSpecificHeatDevice(tpOld)", cuda)
        self.assertGreaterEqual
        (
            cuda.count("particleSpecificEnthalpyDevice(tp)"),
            3,
        )
        self.assertIn("particleTemperatureFromSpecificEnthalpyDevice", cuda)
        self.assertNotIn("particleThermalRho", cuda)
        self.assertNotIn("double particleCp;", protocol)
        self.assertNotIn('"particleCp"', fields)

    def test_ptheta_has_one_granular_temperature_meaning(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        self.assertNotIn("const double theta = s.pStuck[i] != 0", cuda)
        self.assertNotIn("const double theta = stuck", cuda)
        self.assertNotIn("accumulatedSpecificEnergy", cuda)
        self.assertNotIn("storedSpecificEnergy", cuda)
        self.assertNotIn("forceParticleMassToParcelMassKernel", cuda)
        self.assertIn("s.pm[i]", cuda)
        self.assertIn("s.pTheta[i] = 0.0;", cuda)
        self.assertNotIn("decayWallMechanicalEnergy", cuda)
        self.assertNotIn("decayedSpecificEnergy", cuda)

    def test_collision_preserves_particle_temperature(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        start = cuda.index("__device__ void sampleOnePoissonPoolParticle")
        end = cuda.index("__device__ void finalizeOneThermalizedParticle", start)
        body = cuda[start:end]
        self.assertNotIn("s.pT[i] =", body)
        self.assertNotIn("poissonPoolHeat", cuda)
        self.assertIn("blockReduceComponentSums<8>", cuda)

    def test_detachment_launches_into_owner_cell(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        start = cuda.index(
            "__device__ void finalizeOneThermalizedStuckParticle"
        )
        end = cuda.index(
            "template<bool StuckPath>",
            start,
        )
        body = cuda[start:end]
        self.assertIn("const double fluctuationUx = candidateUx - poolMeanUx;", body)
        self.assertIn("0.5*sqr3(fluctuationUx, fluctuationUy, fluctuationUz)", body)
        self.assertIn("applyCapillaryContactDamage", body)
        self.assertIn("damage.residualSpecificEnergyJkg/sampledSpecificEnergy", body)
        self.assertIn("poolMeanUx + fluctuationScale*fluctuationUx", body)
        self.assertIn("relativeNormal > 0.0", body)
        self.assertNotIn("candidateUx - wallUx", body)
        self.assertIn("s.puxOld[i] = releaseUx;", body)
        self.assertIn("s.pStuck[i] = 0;", body)
        self.assertIn("s.pDepositionArea[i] =", body)

    def test_persistent_heat_uses_equilibrium_cap_contact_area(self) -> None:
        cuda = read("private_backend/GpuResidentStrict.cu")
        start = cuda.index("__device__ void relaxOneParticleToResidentGas")
        end = cuda.index("__global__ void relaxMobileParticlesToResidentGasKernelStatic", start)
        body = cuda[start:end]
        self.assertIn("capillary.equilibriumContactAreaM2", body)
        self.assertIn("storedDepositionArea", body)
        self.assertIn("finiteContactWallConductanceTimeIntegral", body)
        self.assertIn("lumpedParticleWallInterfaceContact", body)
        self.assertIn("particleWallDeposited", body)

    def test_capillary_geometry_uses_runtime_constant_angle(self) -> None:
        capillary = read("thermal/GpuCapillaryDetachment.H")
        self.assertIn("const double contactAngleCosine", capillary)
        self.assertIn("clampCapillary", capillary)
        self.assertNotIn("tungstenReferenceTemperatureK", capillary)
        self.assertNotIn("referenceContactAngleDegree", capillary)
        self.assertNotIn("solidSurfaceEnergyDifference", capillary)
        self.assertNotIn("::cos", capillary)
        self.assertNotIn("50.90", capillary)
        self.assertNotIn("capillaryReferenceContactAngleDegree", capillary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
