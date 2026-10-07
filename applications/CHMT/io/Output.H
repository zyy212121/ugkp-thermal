#ifndef CHMT_IO_OUTPUT_H
#define CHMT_IO_OUTPUT_H
#include "configuration/InitialConditions.H"
#include "film/FilmContracts.H"
#include <fstream>
#include <iomanip>
namespace chmt {
    namespace io {
        inline std::string jsonString(const std::string& input) {
            std::ostringstream o;
            o<<'"';
            for (unsigned char c:input) {
                if (c == '"' || c == '\\')o<<'\\'<<c;
                else if (c == '\n')o<<"\\n";
                else if (c == '\r')o<<"\\r";
                else if (c == '\t')o<<"\\t";
                else if (c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<int>(c)<<std::dec;
                else o<<c;
            }
            o<<'"';
            return o.str();
        }
        inline void fileText(const Foam::fileName& path, const std::string& value) {
            std::ofstream f(path.c_str());
            require(bool(f), "cannot write "+std::string(path));
            f<<value;
            f.flush();
            require(bool(f), "write failed "+std::string(path));
        }
        inline std::vector<Real> coordinates(const std::vector<Vec3>& points, int axis) {
            std::vector<Real> values;
            for (auto p:points)values.push_back(component(p, axis));
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end(), [](Real a, Real b) {
                return std::abs(a-b)<1e-10;
            }), values.end());
            return values;
        }
        inline int rank(const std::vector<Real>& positions, Real value) {
            auto it = std::min_element(positions.begin(), positions.end(), [&](Real a, Real b) {
                return std::abs(a-value)<std::abs(b-value);
            });
            return it-positions.begin();
        }
        class Output {
            Foam::fileName directory_;
            std::ofstream observations_;
            std::string case_, variant_;
            int resolution_;
            bool stages_;
            Real maximumAcceptedDt_ = 0;
            public: Output(const Foam::fileName& casePath, const Foam::dictionary& d):directory_(casePath/text(d, "outputDirectory", "chmtOutput")), case_(text(d, "caseLabel", "production")), variant_(text(d, "variantLabel", "default")), resolution_(d.lookupOrDefault<Foam::label>("resolution", 0)), stages_(flag(d, "writeStageGeometry")) {
                require(!Foam::isDir(directory_) && !Foam::isFile(directory_), "outputDirectory exists; refusing to overwrite raw outputs");
                require(Foam::mkDir(directory_), "cannot create output directory");
                observations_.open((directory_/"observations.csv").c_str());
                require(bool(observations_), "cannot open observations");
                observations_<<"case,variant,n,time,sample,quantity,value,kind\n"<<std::setprecision(17);
                require(case_.find_first_of(",\r\n\"") == std::string::npos && variant_.find_first_of(",\r\n\"") == std::string::npos, "unsafe CSV output label");
                fileText(directory_/"steps.csv", "step,time,accepted_dt,next_dt,rejected_trials,coupling_residual,gcl_residual,film_reduced_residual,film_kinetic_defect\n");
                if (stages_)fileText(directory_/"gcl.csv", "step,stage,cell,V_old,V_new,sweep_left,sweep_right,sweep_bottom,sweep_top\n");
            }
            const Foam::fileName& directory()const {
                return directory_;
            }
            void row(Real time, const std::string& sample, const std::string& quantity, Real value) {
                require(finite(value) && finite(time), "nonfinite exported "+quantity);
                observations_<<case_<<','<<variant_<<','<<resolution_<<','<<time<<','<<sample<<','<<quantity<<','<<value<<",SOLVER_OBSERVATION\n";
                require(bool(observations_), "observation write failure");
            }
            void profile(const FilmProfileResult& r) {
                row(0, "local", "u_mean", r.meanVelocity.x);
                row(0, "local", "u_top", r.topVelocity.x);
                row(0, "local", "tau_bottom", r.bottomShear.x);
                row(0, "local", "phi", r.dissipation);
                row(0, "local", "boundary_work", r.boundaryWork);
                row(0, "local", "mass_flux", r.massFlux.x);
                row(0, "local", "enthalpy_flux", r.enthalpyFlux.x);
                for (int i = 0; i<r.nSamples; ++i)row(0, std::to_string(i), "u_z", r.velocity[i].x);
                std::ostringstream raw;
                raw<<std::setprecision(17)<<"quantity,x,y,z\n";
                auto vectorRow = [&](const char* name, Vec3 v) {
                    raw<<name<<','<<v.x<<','<<v.y<<','<<v.z<<'\n';
                };
                vectorRow("meanVelocity", r.meanVelocity);
                vectorRow("topVelocity", r.topVelocity);
                vectorRow("bottomShear", r.bottomShear);
                vectorRow("massFlux", r.massFlux);
                vectorRow("enthalpyFlux", r.enthalpyFlux);
                raw<<"kineticEnergyPerArea,"<<r.kineticEnergyPerArea<<",0,0\n";
                fileText(directory_/"raw-profile.csv", raw.str());
            }
            void pressure(int index, const FilmPressureResult& r) {
                row(index, "local", "H", r.state.enthalpy);
                row(index, "local", "T", r.aux.temperature);
                row(index, "local", "internal_energy", r.internalEnergy);
                std::ostringstream raw;
                raw<<std::setprecision(17)<<"mass,H,p,area,thickness,T,U,deltaPV\n"<<r.state.mass<<','<<r.state.enthalpy<<','<<r.aux.pressure<<','<<r.aux.area<<','<<r.aux.thickness<<','<<r.aux.temperature<<','<<r.internalEnergy<<','<<r.pressureVolumeDelta<<'\n';
                fileText(directory_/Foam::fileName("raw-pressure-"+std::to_string(index)+".csv"), raw.str());
            }
            void phase(const FilmPhaseResult& r) {
                row(0, "local", "F_gas", r.gasEnergyFlux);
                row(0, "local", "Q_liquid", r.liquidHeatInput);
                row(0, "local", "enthalpy_rhs", r.enthalpyRhs);
                row(0, "local", "rho_liquid", r.liquidDensity);
                row(0, "local", "rho_gas", r.gasDensity);
            }
            void state(const HostState& state, const ModelConfig& model, const Foam::dictionary& d, Real sampleTime) {
                const auto xs = coordinates(state.gasMesh.cellCentres, 0), ys = coordinates(state.gasMesh.cellCentres, 1);
                std::ostringstream gas;
                gas<<std::setprecision(17)<<"cell,V,mass,momentum_x,momentum_y,momentum_z,energy";
                for (int s = 0; s<Ns; ++s)gas<<",species_"<<s;
                gas<<'\n';
                for (std::size_t c = 0; c<state.gas.size(); ++c) {
                    const auto& q = state.gas[c];
                    const Vec3 centre = state.gasMesh.cellCentres[c];
                    const std::string sample = case_.find("ale_") == 0?std::to_string(rank(xs, centre.x))+":"+std::to_string(rank(ys, centre.y)):std::to_string(c);
                    row(sampleTime, sample, "volume", state.gasMesh.volumes[c]);
                    row(sampleTime, sample, "mass", q.mass);
                    row(sampleTime, sample, "momentum_x", q.momentum.x);
                    row(sampleTime, sample, "momentum_y", q.momentum.y);
                    row(sampleTime, sample, "energy", q.energy);
                    for (int s = 0; s<Ns; ++s)row(sampleTime, sample, "species_"+std::to_string(s), q.species[s]);
                    gas<<c<<','<<state.gasMesh.volumes[c]<<','<<q.mass<<','<<q.momentum.x<<','<<q.momentum.y<<','<<q.momentum.z<<','<<q.energy;
                    for (int s = 0; s<Ns; ++s)gas<<','<<q.species[s];
                    gas<<'\n';
                }
                if (!state.gas.empty())fileText(directory_/Foam::fileName("raw-gas-"+std::to_string(state.acceptedSteps)+".csv"), gas.str());
                if (!state.film.empty()) {
                    const auto x = coordinates(state.surface.centre, 0);
                    Real mass = 0;
                    std::ostringstream film;
                    film<<std::setprecision(17)<<"face,area,mass,H,p,delta,T,base_x,mean_x,top_x,solidFront,gasFront\n";
                    for (std::size_t f = 0; f<state.film.size(); ++f) {
                        const auto& q = state.film[f];
                        const auto& a = state.filmAux[f];
                        mass += q.mass;
                        row(sampleTime, std::to_string(rank(x, state.surface.centre[f].x)), "H_bar", q.enthalpy/state.surface.area[f]);
                        film<<f<<','<<state.surface.area[f]<<','<<q.mass<<','<<q.enthalpy<<','<<a.pressure<<','<<a.thickness<<','<<a.temperature<<','<<a.baseVelocity.x<<','<<a.meanVelocity.x<<','<<a.topVelocity.x<<','<<a.solidFront<<','<<a.gasFront<<'\n';
                    }
                    row(sampleTime, "global", "mass", mass);
                    row(sampleTime, "global", "boundary_work", state.budget.supportWork);
                    fileText(directory_/Foam::fileName("raw-film-"+std::to_string(state.acceptedSteps)+".csv"), film.str());
                }
                if (!state.normalEnthalpy.empty()) {
                    require(state.surface.normalOffsets.size() == 2, "normal observation export currently supports one column");
                    Real front = 0;
                    const Real offset = readNumber(d, "mathematicalTemperatureOffset", 0);
                    std::ostringstream normal;
                    normal<<std::setprecision(17)<<"layer,x,volume,mass,H,liquidFraction,T_internal\n";
                    for (std::size_t i = 0; i<state.normalEnthalpy.size(); ++i) {
                        Real T = 0;
                        require(recoverNormalSensibleAverage(state.surface.normalLayerMass[i], state.normalEnthalpy[i], state.normalLiquidFraction[i], state.surface.normalPressure[0], model.physics, T), "normal sensible diagnostic failed");
                        row(sampleTime, std::to_string(i), "T_bar", T-offset);
                        row(sampleTime, std::to_string(i), "h_bar", state.normalEnthalpy[i]/state.surface.normalLayerMass[i]);
                        front += state.normalLiquidFraction[i]*state.surface.normalLayerVolume[i]/state.surface.area[0];
                        normal<<i<<','<<state.surface.normalCoordinates[i]<<','<<state.surface.normalLayerVolume[i]<<','<<state.surface.normalLayerMass[i]<<','<<state.normalEnthalpy[i]<<','<<state.normalLiquidFraction[i]<<','<<T<<'\n';
                    }
                    row(sampleTime, "global", "front", front);
                    require(state.normalAcceptedWallHeat.size() == 1, "missing accepted normal wall heat");
                    row(sampleTime, "global", "wall_heat", state.normalAcceptedWallHeat[0]);
                    fileText(directory_/Foam::fileName("raw-normal-"+std::to_string(state.acceptedSteps)+".csv"), normal.str());
                }
                // Full persistent state is also serialized by the backend checkpoint writer.
                observations_.flush();
                require(bool(observations_), "observation flush failure");
            }
            void remap(const std::vector<Real>& edges, const std::vector<GasQ>& q) {
                std::ostringstream raw;
                raw<<std::setprecision(17)<<"cell,left,right,mass,momentum_x,momentum_y,momentum_z,energy";
                for (int s = 0; s<Ns; ++s)raw<<",species_"<<s;
                raw<<'\n';
                for (std::size_t c = 0; c<q.size(); ++c) {
                    const auto& x = q[c];
                    row(0, std::to_string(c), "mass", x.mass);
                    row(0, std::to_string(c), "momentum_x", x.momentum.x);
                    row(0, std::to_string(c), "energy", x.energy);
                    for (int s = 0; s<Ns; ++s)row(0, std::to_string(c), "species_"+std::to_string(s), x.species[s]);
                    raw<<c<<','<<edges[c]<<','<<edges[c+1]<<','<<x.mass<<','<<x.momentum.x<<','<<x.momentum.y<<','<<x.momentum.z<<','<<x.energy;
                    for (int s = 0; s<Ns; ++s)raw<<','<<x.species[s];
                    raw<<'\n';
                }
                fileText(directory_/"raw-remap.csv", raw.str());
            }
            void acceptedStep(const HostState& state, const StepReport& report) {
                maximumAcceptedDt_ = std::max(maximumAcceptedDt_, report.acceptedDt);
                std::ofstream history((directory_/"steps.csv").c_str(), std::ios::app);
                history<<std::setprecision(17)<<state.acceptedSteps<<','<<state.time<<','<<report.acceptedDt<<','<<report.nextDt<<','<<report.rejectedTrials<<','<<report.couplingResidual<<','<<report.gclResidual<<','<<report.filmReducedResidual<<','<<report.filmKineticDefect<<'\n';
                history.flush();
                require(bool(history), "accepted step-history write failure");
            }
            void stages(const HostState& state) {
                if (!stages_) {
                    return;
                }
                require(state.acceptedSteps>0, "no accepted stage geometry available");
                std::ofstream gcl((directory_/"gcl.csv").c_str(), std::ios::app);
                gcl<<std::setprecision(17);
                const auto& mesh = state.gasMesh;
                for (int stage = 0; stage<2; ++stage) {
                    const auto& s = state.gasStages[stage];
                    require(s.interval>0 && s.oldVolume.size() == mesh.volumes.size() && s.newVolume.size() == mesh.volumes.size() && s.sweptVolume.size() == mesh.owner.size() && s.areaVector.size() == mesh.owner.size(), "missing actual accepted stage geometry");
                    std::ostringstream raw;
                    raw<<std::setprecision(17)<<"face,owner,neighbour,sweptVolume,area_x,area_y,area_z\n";
                    for (std::size_t f = 0; f<mesh.owner.size(); ++f)raw<<f<<','<<mesh.owner[f]<<','<<mesh.neighbour[f]<<','<<s.sweptVolume[f]<<','<<s.areaVector[f].x<<','<<s.areaVector[f].y<<','<<s.areaVector[f].z<<'\n';
                    fileText(directory_/Foam::fileName("raw-stage-"+std::to_string(state.acceptedSteps-1)+"-"+std::to_string(stage)+".csv"), raw.str());
                    for (std::size_t c = 0; c<mesh.volumes.size(); ++c) {
                        Real sweep[4]{};
                        int count[4]{};
                        for (int entry = mesh.cellFaceOffsets[c]; entry<mesh.cellFaceOffsets[c+1]; ++entry) {
                            const int f = mesh.cellFaces[entry], sign = mesh.cellFaceSigns[entry];
                            const Vec3 a = s.areaVector[f]*sign;
                            const Real w = s.sweptVolume[f]*sign;
                            int side = -1;
                            if (std::abs(a.x)>std::abs(a.y) && std::abs(a.x)>std::abs(a.z))side = a.x<0?0:1;
                            else if (std::abs(a.y)>std::abs(a.z))side = a.y<0?2:3;
                            else require(std::abs(w) <= 1e-13*std::max(s.oldVolume[c], s.newVolume[c]), "2D GCL export encountered nonzero depth-face sweep");
                            if (side >= 0) {
                                sweep[side] += w;
                                count[side]++;
                            }
                        }
                        for (int k = 0; k<4; ++k) {
                            require(count[k] == 1, "2D GCL export requires exactly one actual face per side");
                        }
                        gcl<<state.acceptedSteps-1<<','<<stage<<','<<c<<','<<s.oldVolume[c]<<','<<s.newVolume[c];
                        for (Real w:sweep)gcl<<','<<w;
                        gcl<<'\n';
                    }
                }
                gcl.flush();
                require(bool(gcl), "GCL export failure");
            }
            void complete(std::uint64_t steps, Real dt, Real time, const std::string& kind, const Foam::dictionary& d) {
                observations_.flush();
                require(bool(observations_), "observations flush failure");
                std::ostringstream value;
                value<<std::setprecision(17)<<"{\"artifact_kind\":\"CHMT_ACCEPTED_OUTPUT\",\"completed\":true,\"build_id\":"<<jsonString(CHMT_BUILD_ID)<<",\"steps\":"<<steps<<",\"time_step\":"<<std::max(dt, maximumAcceptedDt_)<<",\"time\":"<<time<<",\"execution_kind\":"<<jsonString(kind)<<",\"mathematical_temperature_offset\":"<<readNumber(d, "mathematicalTemperatureOffset", 0)<<",\"reconstruction\":"<<jsonString(text(d, "reconstruction", "LimitedLinear"))<<"}\n";
                fileText(directory_/"run-state.json", value.str());
            }
        };
    }
}
#endif
