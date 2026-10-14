// Native CPU owner regression. No CUDA backend or GPU surrogate is linked.
#include "fvCFD.H"
#include "io/MeshIO.H"
#include "materials/CpuMaterialDriver.H"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace chmt;

void check(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

bool sameBits(Real a, Real b)
{
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

int main(int argc, char** argv)
{
    Foam::argList::addOption("phaseCase", "word", "unselected, selected, or melting");
    Foam::argList args(argc, argv);
    Foam::Time runTime(Foam::Time::controlDictName, args);
    Foam::fvMesh mesh(Foam::IOobject(Foam::fvMesh::defaultRegion,
        runTime.timeName(), runTime, Foam::IOobject::MUST_READ));
    const Foam::word mode = args.optionLookupOrDefault<Foam::word>("phaseCase", "unselected");
    try {
        ModelConfig model;
        auto& p = model.physics;
        p.enableFilm = true;
        p.spatialOrder = 1;
        p.gasConductivity = 2;
        p.gasViscosity = .01;
        p.liquidViscosity = 1;
        for (auto& s : p.species) {
            s.R = 287; s.cp0 = 1000; s.Tmin = 100; s.Tmax = 3000;
        }
        for (auto& s : p.condensed) {
            s.rho = 1000; s.cp0 = 1000; s.conductivity = 2;
            s.Tmin = 100; s.Tmax = 3000;
        }
        p.liquid = p.condensed[0];
        p.material.phaseFilmY[0] = 1;
        if (mode != "unselected") p.material.phaseCondensed = 0;
        if (mode == "melting") {
            p.material.enableMelting = true;
            p.meshMotion.policy = MeshMotionPolicy::CoupledRecession;
            p.meltTemperature = 600;
            p.liquid.e0 = 200000;
        }
        check(p.filmThermalMode == FilmThermalMode::ThicknessAveraged,
            "fixture must use ThicknessAveraged");
        check(mode != "unselected" || (p.material.phaseCondensed == -1
            && !p.material.enableMelting), "fixture must retain the unselected default");

        HostState base;
        base.solidMesh = io::importMesh(mesh, Foam::dictionary());
        base.gasMesh = base.solidMesh;
        for (auto& point : base.gasMesh.points) point.x += 1.001;
        std::string error;
        check(rebuildGeometry(base.gasMesh, error), error);
        base.gasMesh.oldPoints = base.gasMesh.referencePoints = base.gasMesh.points;
        base.gasMesh.oldVolumes = base.gasMesh.volumes;
        const int sf = mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("right")].start();
        const int gf = mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("left")].start();
        base.solidMesh.boundaryKind[sf] = BoundaryKind::Interface;
        base.gasMesh.boundaryKind[gf] = BoundaryKind::Interface;
        check(rebuildGeometry(base.solidMesh, error), error);
        check(rebuildGeometry(base.gasMesh, error), error);
        base.solid.resize(1);
        base.solid[0].condensed[0] = p.condensed[0].rho * base.solidMesh.volumes[0];
        base.solid[0].energy = base.solid[0].condensed[0] * condensedE(p.condensed[0], 700);
        GasPrimitive gas;
        gas.rho = 1; gas.temperature = 800; gas.pressure = gas.rho * 287 * gas.temperature;
        gas.Y[0] = 1;
        base.gas = {conservativeGas(gas, base.gasMesh.volumes[0], p)};
        auto& surface = base.surface;
        surface.area = surface.oldArea = {1};
        surface.normal = {{1, 0, 0}};
        surface.centre = {{1.001, .5, .5}};
        surface.baseVelocity = surface.meshVelocity = {{0, 0, 0}};
        surface.gasFace = {gf}; surface.solidFace = {sf}; surface.solidCell = {0};
        surface.gasDistance = surface.solidDistance = {.5};
        surface.persistentId = {base.gasMesh.faceIds[gf]};
        base.film.resize(1);
        base.film[0].mass = 1;
        base.film[0].species[0] = 1;
        base.film[0].enthalpy = liquidH(p, 800, gas.pressure);
        base.filmAux.resize(1);
        check(recoverFilm(base.film[0], 1, gas.pressure, p, base.filmAux[0]), "film recovery");
        base.filmAux[0].normal = {1, 0, 0};

        CouplingInterval interval;
        interval.identity.sequence = 1;
        interval.end = 1e-6;
        GasIntervalRecord record;
        record.end = interval.end; record.microSequence = 1;
        record.gasGeometry = base.gasMesh.geometryVersion;
        record.solidGeometry = base.solidMesh.geometryVersion;
        record.gasTrace = {gas}; record.gasTraction = {{0, 0, 0}};
        IntervalHistory history;
        check(history.begin(interval, error) && history.appendAccepted(record, error), error);
        base.gasChemistryAudit.resize(1);
        base.gasChemistryAudit[0].integratedTime = 42;
        base.gasChemistryAudit[0].energyResidual = NAN;
        base.gasSstAudit.sourceK = NAN;
        base.budget.turbulenceInventory = NAN;
        HostState endpoint = base;
        endpoint.gasChemistryAudit[0].integratedTime = 99; // Legacy endpoint diagnostics are not propagated.
        endpoint.gasSstAudit.sourceK = 123;
        endpoint.time = interval.end;
        CpuMaterialDriver driver(model, &mesh);
        HostState result;
        WallProgram corrected;
        CpuMaterialReport report;
        const bool advanced = driver.advanceCandidate(base, endpoint, history,
            CpuMaterialControls{}, result, corrected, report, error);
        check(advanced, "actual CPU material candidate: " + error);
        check(result.gasChemistryAudit.size() == 1 && result.gasChemistryAudit[0].integratedTime == 42
            && std::isnan(result.gasChemistryAudit[0].energyResidual) && std::isnan(result.gasSstAudit.sourceK),
            "legacy diagnostics remain base payload without per-window endpoint copy");
        check(report.materialSteps == 1 && report.filmSteps == 1, "complete material/film step");
        check(result.ledger.size() == 1, "phase packet must be retained");
        const auto& phase = result.ledger[0];
        check(phase.kind == ExchangeKind::SolidFilm, "phase packet kind");
        check(phase.energy != 0 && phase.conductive != 0 && mag(phase.momentum) == 0,
            "phase packet retains heat without unused support impulse");
        check(report.budgetDelta.consumedPackets == 1, "phase accounting count");
        check(report.budgetDelta.exchangeEnergy[SolidParticipant] == -phase.energy
            && report.budgetDelta.exchangeEnergy[FilmParticipant] == phase.energy,
            "solid/film phase heat accounting");
        check(report.budgetDelta.exchangeMomentum[SolidParticipant].x == -phase.momentum.x
            && report.budgetDelta.exchangeMomentum[FilmParticipant].x == phase.momentum.x,
            "solid/film phase momentum diagnostic is inactive");
        check(report.budgetDelta.supportImpulse.x == -phase.momentum.x,
            "legacy support slot is not accumulated");
        check(result.film[0].enthalpy != base.film[0].enthalpy,
            "phase heat must still reach film forcing");
        if (mode == "melting") {
            check(phase.mass != 0, "nonzero-phase control must transfer mass");
            MaterialPrimitive material;
            check(recoverMaterial(base.solid[0], base.solidMesh.volumes[0], p, material), "material recovery");
            Real expected = 0;
            expected -= phase.mass / p.condensed[p.material.phaseCondensed].rho / (1 - material.porosity);
            check(sameBits(report.solidTargetSweeps.at(0), expected), "nonzero-phase FP64 arithmetic changed");
        } else {
            check(phase.mass == 0, "noMelting must have zero phase mass");
            check(report.solidTargetSweeps.at(0) == 0, "zero phase mass must have zero volume");
            check(result.film[0].mass == base.film[0].mass, "zero phase mass changed film inventory");
            check(result.solid[0].condensed[0] == base.solid[0].condensed[0], "zero phase mass changed solid inventory");
        }
        std::cout << "PASS native CPU phase " << mode << " mass=" << std::hexfloat << phase.mass
            << " heat=" << phase.energy << " force=" << phase.momentum.x
            << " sweep=" << report.solidTargetSweeps[0] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
