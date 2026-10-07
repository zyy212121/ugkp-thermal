#include "fvCFD.H"
#include "configuration/ModelIO.H"
#include "configuration/InitialConditions.H"
#include "configuration/MultirateIO.H"
#include "io/Output.H"
#include "io/MultirateOutput.H"
#include "particles/ParticleMath.H"
#include "gpu/Backend.H"
#include "film/FilmContracts.H"
#include <memory>
namespace {
    using namespace chmt;
    using namespace chmt::io;
    void localOperator(const Foam::dictionary& d, const ModelConfig& model, Output& out,
        const std::string& operation) {
        std::string error;
        if (operation == "filmProfileContract") {
            const auto& p = d.subDict("profile");
            FilmProfileRequest request;
            request.delta = readNumber(p, "delta");
            request.mu = readNumber(p, "mu");
            request.rho = readNumber(p, "rho");
            request.h = readNumber(p, "h");
            request.baseVelocity = readVector(p, "baseVelocity");
            request.topShear = readVector(p, "topShear");
            request.pressureBodyGradient = readVector(p, "pressureBodyGradient");
            const auto z = scalars(p, "z");
            require(z.size() <= FilmContractMaxSamples, "too many film profile samples");
            request.nSamples = z.size();
            std::copy(z.begin(), z.end(), request.z);
            FilmProfileResult result;
            require(evaluateFilmProfileContract(request, result, error), error);
            out.profile(result);
        } else if (operation == "filmPressureContract") {
            const auto& p = d.subDict("pressureContract");
            const auto pressures = scalars(p, "pressures");
            require(!pressures.empty(), "pressure contract requires at least one pressure");
            FilmPressureRequest request;
            request.physics = model.physics;
            request.area = readNumber(p, "area");
            request.oldPressure = pressures.front();
            request.state.mass = model.physics.liquid.rho*readNumber(p, "delta")*request.area;
            request.state.enthalpy = request.state.mass*liquidH(model.physics, readNumber(p, "temperature"),
                request.oldPressure);
            Real Y[Ns]{};
            array(p, "Y", Y);
            for (int s = 0; s<Ns; ++s)request.state.species[s] = request.state.mass*Y[s];
            for (std::size_t i = 0; i<pressures.size(); ++i) {
                request.newPressure = pressures[i];
                FilmPressureResult result;
                require(evaluateFilmPressureContract(request, result, error), error);
                out.pressure(i, result);
                request.state = result.state;
                request.oldPressure = request.newPressure;
            }
        } else if (operation == "filmPhaseContract") {
            const auto& p = d.subDict("phaseContract");
            FilmPhaseRequest request;
            request.massFlux = readNumber(p, "massFlux");
            request.liquidEnthalpy = readNumber(p, "liquidEnthalpy");
            request.gasEnthalpy = readNumber(p, "gasEnthalpy");
            request.pressure = readNumber(p, "pressure");
            request.normalSpeed = readNumber(p, "normalSpeed");
            request.normal = readVector(p, "normal");
            request.liquidVelocity = readVector(p, "liquidVelocity");
            request.gasVelocity = readVector(p, "gasVelocity");
            request.liquidViscousWork = readNumber(p, "liquidViscousWork");
            request.gasViscousWork = readNumber(p, "gasViscousWork");
            request.gasConductiveFlux = readNumber(p, "gasConductiveFlux");
            FilmPhaseResult result;
            require(evaluateFilmPhaseContract(request, result, error), error);
            out.phase(result);
        } else throw std::runtime_error("unsupported local operator "+operation);
        out.complete(0, 0, 0, "CUDA_OPERATOR_CONTRACT", d);
    }
    void runRemap(const Foam::dictionary& d, const HostState& state, Output& output) {
        const auto& specification = d.subDict("remap");
        require(specification.lookup<Foam::label>("axis") == 0,
            "remap overlap operator currently supports x-directed 1D columns");
        const auto& m = state.gasMesh;
        std::vector<int> sorted(m.volumes.size());
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return m.cellCentres[a].x<m.cellCentres[b].x;
        });
        std::vector<Real> oldEdges, newEdges;
        std::vector<GasQ> donors, remapped;
        Real crossSection = 0;
        for (int c:sorted) {
            const auto bounds = cellBounds(m, c);
            rectangular(bounds, m.volumes[c]);
            const Real area = (bounds.hi.y-bounds.lo.y)*(bounds.hi.z-bounds.lo.z);
            if (oldEdges.empty()) {
                oldEdges.push_back(bounds.lo.x);
                crossSection = area;
            }
            require(closeEnough(crossSection, area, 1e-13, 1e-10) && closeEnough(oldEdges.back(), bounds.lo.x,
                1e-13, 1e-10), "remap donor mesh must be a contiguous constant-area 1D column");
            oldEdges.push_back(bounds.hi.x);
            donors.push_back(state.gas[c]);
        }
        for (Real x:oldEdges)newEdges.push_back(x+readNumber(specification,
            "amplitude")*std::sin(readNumber(specification, "waveNumber")*x));
        // Pin only the prescribed fixed endpoints, not any interior overlap result.
        require(std::abs(newEdges.front()-oldEdges.front())<1e-13
            && std::abs(newEdges.back()-oldEdges.back())<1e-13,
            "rezone prescription must preserve fixed endpoints");
        newEdges.front() = oldEdges.front();
        newEdges.back() = oldEdges.back();
        std::string error;
        require(remapGas1D(oldEdges, donors, newEdges, remapped, error), error);
        output.remap(newEdges, remapped);
        output.complete(0, 0, 0, "CUDA_REMAP_OPERATOR", d);
    }
    int execute(Foam::Time& time, const Foam::dictionary& d) {
        const std::string operation = text(d, "operation", "evolve");
        const ModelConfig model = readModel(d);
        Output output(time.path(), d);
        fileText(output.directory()/"build.json", backendBuildInfo()+"\n");
        if (operation == "filmProfileContract" || operation == "filmPressureContract"
            || operation == "filmPhaseContract") {
            localOperator(d, model, output, operation);
            return 0;
        }
        require(operation == "evolve" || operation == "remap1D", "unsupported operation");
        // The default region is also the geometric source for explicit gas-disabled
        // surface/normal configurations. It does not imply a gas inventory exists.
        Foam::fvMesh source(Foam::IOobject(Foam::polyMesh::defaultRegion, time.timeName(), time,
            Foam::IOobject::MUST_READ));
        HostMesh imported = importMesh(source, d);
        HostState state;
        state.time = time.value();
        if (model.physics.enableGas) {
            state.gasMesh = imported;
            initialGas(source, d, model, state);
        }
        std::unique_ptr<Foam::fvMesh> solidSource;
        if (d.found("solidRegion")) {
            const auto name = text(d, "solidRegion");
            solidSource.reset(new Foam::fvMesh(Foam::IOobject(name, time.timeName(), time,
                Foam::IOobject::MUST_READ)));
            state.solidMesh = importMesh(*solidSource, d.found("solidBoundaries")?d.subDict("solidBoundaries"):d);
            importSolidFields(*solidSource, model, state);
        }
        if (d.found("normalColumn")) {
            require(!model.physics.enableGas && !solidSource,
                "first normal-only frontend forbids unrelated external gas/solid inventory");
            initialNormal(source, imported, d, model, state);
        } else if (d.found("surface")) {
            const bool standalone = flag(d.subDict("surface"), "standalone");
            require(standalone || bool(solidSource), "coupled surface requires an actual solid region");
            auto surface = importSurface(standalone?source:*solidSource, standalone?imported:state.solidMesh,
                d.subDict("surface"), standalone);
            if (model.physics.enableFilm)initialFilm(d, model, surface, state);
            else {
                state.surface = surface.mesh;
                initializeDryInterfaceAux(state);
            }
            if (!standalone && model.physics.enableGas)attachCoupledSurface(source, d, state);
        }
        importParticles(d, model, state);
        if (operation == "remap1D") {
            runRemap(d, state, output);
            return 0;
        }
        std::string error;
        if (d.found("restartDirectory"))require(readCheckpoint(text(d, "restartDirectory"), model, state,
            error), error);
        const Real end = Foam::readScalar(time.controlDict().lookup("endTime"));
        const Real configuredDt = Foam::readScalar(time.controlDict().lookup("deltaT"));
        require(chmt::finite(configuredDt) && configuredDt>0 && chmt::finite(end) && end >= state.time,
            "invalid controlDict interval");
        const ExecutionMode mode = readExecutionMode(d,state);
        const bool multirate = mode == ExecutionMode::Multirate;
        fileText(output.directory()/"execution-mode.json",std::string("{\"execution_mode\":\"")
            +(multirate?"Multirate":mode==ExecutionMode::Standalone?"Standalone":"LegacyExplicit")+"\"}\n");
        if (mode == ExecutionMode::Standalone) {
            require(state.solid.empty(), "Standalone cannot advance a coupled solid region; use Multirate");
            for (int face:state.surface.gasFace)
                require(face<0, "Standalone cannot advance gas/material interface exchange; use Multirate");
        }
        CouplingControls coupling;
        CpuMaterialControls material;
        if (multirate) {
            require(model.physics.enableGas && !state.gas.empty(), "Multirate requires actual gas inventory");
            require(state.normalEnthalpy.empty(), "ResolvedNormal is a separate Standalone verification model");
            require(!state.solid.empty() && bool(solidSource), "Multirate requires actual 3D solid inventory and its solid fvMesh");
            require(!model.physics.enableParticles || model.physics.particle.contactDuration<=0,
                "finite-duration particle contact is unsupported in multirate mode");
            if (model.physics.enableParticles) for (const auto& particle:state.particles)
                require(particle.state!=ParticleContact, "pending particle contact is unsupported in multirate mode");
            coupling = readCouplingControls(d,configuredDt);
            material = readCpuMaterialControls(d,coupling);
            if (solidSource) publishSolidMesh(*solidSource,state.solidMesh,model.physics.tolerances);
        }
        std::unique_ptr<Backend, void(*)(Backend*)> backend(createBackend(model, state, error), destroyBackend);
        require(bool(backend), error);
        require(downloadState(*backend, state, error), error);
        requireSynchronizedState(state);
        // Declared before the CPU driver, solidSource outlives every matrix solve
        // and every rejected candidate. Only macro commits publish native points.
        std::unique_ptr<CpuMaterialDriver> cpu;
        if (multirate) {
            cpu.reset(new CpuMaterialDriver(model,solidSource.get()));
            startWindowOutput(output,coupling,material,flag(d,"writeStageGeometry")); // windows.csv
        }
        auto times = d.found("outputTimes")?scalars(d, "outputTimes"):std::vector<Real>{state.time, end};
        require(!times.empty(), "outputTimes may not be empty");
        require(std::is_sorted(times.begin(), times.end()) && std::adjacent_find(times.begin(),
            times.end()) == times.end(), "outputTimes must strictly increase");
        for (Real target:times) {
            require(multirate?(target >= state.time && target <= end):
                (target >= state.time-1e-14 && target <= end+1e-14),
                "output time outside accepted start/controlDict endTime");
        }
        if (times.back()<end)times.push_back(end);
        for (Real target:times) {
            while (multirate?state.time<target:
                state.time<target-1e-14*std::max(1.0,std::abs(target))) {
                const Real previous = state.time;
                if (multirate) {
                    WindowEvolutionReport report;
                    // target is an output/end boundary. The controller chooses H
                    // independently of controlDict deltaT and state.nextDt.
                    if (!advanceCoupledWindow(*backend,*cpu,model,coupling,material,target,
                        flag(d,"writeStageGeometry"),state,report,error)) {
                        const std::string cause = error;
                        std::string savedError;
                        requireSynchronizedState(state);
                        const bool saved = writeCheckpoint(std::string(output.directory()/"last-accepted"),
                            model,state,savedError);
                        throw std::runtime_error("multirate advance failed: "+cause
                            +(saved?"; saved last synchronized checkpoint":"; checkpoint failed: "+savedError));
                    }
                    requireSynchronizedState(state);
                    require(state.time>previous && state.time<=target+1e-12,
                        "macro commit time outside requested interval");
                    if (solidSource) publishSolidMesh(*solidSource,state.solidMesh,model.physics.tolerances);
                    writeWindowOutput(output,state,report);
                } else {
                    // Explicit regression/independent verification path only.
                    const Real request = std::min(target-state.time, std::min(configuredDt,
                        state.nextDt>0?state.nextDt:model.physics.maxDt));
                    StepReport report;
                    if (!advance(*backend,request,report,error)) {
                        const std::string cause = error;
                        std::string savedError;
                        HostState accepted;
                        const bool downloaded = downloadState(*backend,accepted,savedError);
                        if (downloaded) {
                            requireSynchronizedState(accepted);
                            writeCheckpoint(std::string(output.directory()/"last-accepted"),model,accepted,savedError);
                        }
                        throw std::runtime_error("advance failed: "+cause
                            +(savedError.empty()?"":"; last-accepted checkpoint: "+savedError));
                    }
                    require(report.acceptedDt>0 && report.acceptedDt<=request*(1+1e-12),
                        "backend accepted invalid dt");
                    require(downloadState(*backend,state,error),error);
                    requireSynchronizedState(state);
                    require(state.time>previous && state.time<=target+1e-12,
                        "backend accepted time outside requested interval");
                    output.acceptedStep(state,report);
                    output.stages(state);
                }
            }
            require(multirate?state.time==target:
                std::abs(state.time-target)<=1e-14*std::max(1.0,std::abs(target)),
                "accepted state does not match requested sample time");
            requireSynchronizedState(state);
            output.state(state,model,d,target);
            if (multirate) writeSolidOutput(output,state,model.physics);
            require(writeCheckpoint(std::string(output.directory()/Foam::fileName("checkpoint-"
                +std::to_string(state.acceptedSteps))),model,state,error),error);
        }
        output.complete(state.acceptedSteps,state.lastAcceptedDt,state.time,
            multirate?"GPU_GAS_CPU_IMPLICIT_MATERIAL_MULTIRATE":"CUDA_TIME_EVOLUTION",d);
        return 0;
    }
}
int main(int argc, char** argv) {
    Foam::argList::noParallel();
    Foam::argList::addBoolOption("build-info",
        "Print compiled CHMT build and actual CUDA device identity; no simulation.");
    Foam::FatalError.throwExceptions();
    Foam::FatalIOError.throwExceptions();
    try {
        Foam::argList args(argc, argv);
        if (args.optionFound("build-info")) {
            std::cout<<chmt::backendBuildInfo()<<'\n';
            return 0;
        }
        if (!args.checkRootCase())return 2;
        Foam::Time time(Foam::Time::controlDictName, args);
        Foam::IOdictionary properties(Foam::IOobject("chmtProperties", time.constant(), time,
            Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE));
        return execute(time, properties);
    } catch (const Foam::error& error) {
        Foam::SeriousError<<"CHMT OpenFOAM input failure: "<<error<<Foam::endl;
        return 2;
    } catch (const std::exception& error) {
        std::cerr<<"CHMT failed: "<<error.what()<<'\n';
        return 2;
    }
}
