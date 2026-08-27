                                                                        

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OpenFoamCaseConfigurationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.create = (
            (ROOT / "readGpuGasConfiguration.H").read_text(encoding="utf-8")
            + "\n"
            + (ROOT / "createFields.H").read_text(encoding="utf-8")
        )
        cls.solver = (ROOT / "diluteUgkwpFoam.C").read_text(encoding="utf-8")

    def test_gas_thermodynamics_come_from_fluid_properties(self) -> None:
        self.assertIn('"fluidProperties"', self.create)
        self.assertIn("const dictionary& physicalProperties = fluidPropertiesFile", self.create)
        self.assertIn('subDict("thermoType")', self.create)
        self.assertIn('subDict("mixture")', self.create)
        self.assertIn('subDict("specie")', self.create)
        self.assertIn('subDict("thermodynamics")', self.create)
        self.assertIn('subDict("transport")', self.create)
        self.assertIn('"molWeight"', self.create)
        self.assertIn('"Cp"', self.create)
        self.assertIn('"mu"', self.create)
        self.assertIn('"Pr"', self.create)
        self.assertIn("physicoChemical::RR", self.create)
        self.assertIn('#include "physicoChemicalConstants.H"', self.solver)

    def test_flux_and_discretisation_come_from_fv_schemes(self) -> None:
        self.assertIn("mesh.schemes().dict()", self.create)
        self.assertIn('"fluxScheme"', self.create)
        self.assertIn('subDict("ddtSchemes")', self.create)
        self.assertIn('subDict("gradSchemes")', self.create)
        self.assertIn('subDict("divSchemes")', self.create)
        self.assertIn('subDict("laplacianSchemes")', self.create)
        self.assertIn('subDict("interpolationSchemes")', self.create)
        self.assertIn('subDict("snGradSchemes")', self.create)
        self.assertIn("'Gauss linear'", self.create)
        self.assertIn("'Gauss linear corrected'", self.create)
        for entry in (
            "div(phi,U)",
            "div(phi,e)",
            "div(phi,K)",
            "div(phi,(p|rho))",
            "div(((rho*nuEff)*dev2(T(grad(U)))))",
        ):
            self.assertIn(f'"{entry}"', self.create)

    def test_solver_controls_come_from_fv_solution(self) -> None:
        self.assertIn("mesh.solution().dict()", self.create)
        self.assertIn('subDict("UGKP")', self.create)
        self.assertIn('"rhoMin"', self.create)
        self.assertIn('"TMin"', self.create)

    def test_particle_settings_remain_separate_from_gas_properties(self) -> None:
        for entry in (
            "particleTemperatureTransport",
            "particleTMin",
            "particleTMax",
        ):
            self.assertIn(f'ugkwpProps.lookupOrDefault', self.create)
            self.assertIn(f'"{entry}"', self.create)
        self.assertNotIn('"particleCp"', self.create)

    def test_rho_is_optional_and_derived_from_the_openfoam_thermo_state(self) -> None:
        self.assertIn("IOobject::READ_IF_PRESENT", self.create)
        self.assertIn("p/(max(Tgas, TgasMinG)*Rgas)", self.create)

    def test_scheduled_inlet_requires_p_t_and_solid_fraction_not_rho(self) -> None:
        validation_start = self.solver.index("Scheduled inlet requires")
        validation = self.solver[validation_start - 800 : validation_start + 300]
        self.assertIn("p.boundaryField()", validation)
        self.assertIn("Tgas.boundaryField()", validation)
        self.assertIn("epsS.boundaryField()", validation)
        self.assertNotIn("rho.boundaryField()", validation)
        self.assertIn("fixed-value p, T, and", validation)
        self.assertIn('"epsilonS on patch "', validation)

    def test_scheduled_inlet_density_is_derived_from_p_and_t(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("scheduledPressureDevice(s, simulationTime)", cuda)
        self.assertIn(
            "pressure/clampMin(s.Rgas*temperature, OfSmall)",
            cuda,
        )


if __name__ == "__main__":
    unittest.main()
