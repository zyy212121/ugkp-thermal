#ifndef CHMT_CONFIGURATION_INITIALCONDITIONS_H
#define CHMT_CONFIGURATION_INITIALCONDITIONS_H
#include "io/MeshIO.H"
#include "film/ResolvedNormalThermal.H"
#include <numeric>
namespace chmt {
    namespace io {
        inline Real sinc(Real x) {
            return std::abs(x)<1e-8?1-x*x/6+x*x*x*x/120:std::sin(x)/x;
        }
        inline void rectangular(const Bounds& b, Real volume) {
            const auto d = b.hi-b.lo;
            require(d.x>0 && d.y>0 && d.z>0 && closeEnough(d.x*d.y*d.z, volume, 1e-13, 1e-10),
                "analytic cell-average initializer requires actual rectangular imported cells");
        }
        inline void initialGas(const Foam::fvMesh& source, const Foam::dictionary& d, const ModelConfig& model,
            HostState& state) {
            const auto& init = d.subDict(d.found("gasInitial")?"gasInitial":"initial");
            const auto kind = text(init, "kind");
            if (kind == "PrimitiveFields") {
                importGasFields(source, model, state);
                return;
            }
            require(kind == "SineDensity" || kind == "DensityStep", "unsupported gas initialization kind");
            GasPrimitive w;
            w.velocity = readVector(init, "velocity");
            array(init, "Y", w.Y);
            const Real pressure = readNumber(init, "pressure");
            Real R = 0;
            for (int s = 0; s<Ns; ++s) {
                R += w.Y[s]*model.physics.species[s].R;
                require(model.physics.species[s].cp1 == 0,
                    "exact isobaric average initializer requires constant heat capacities");
            }
            require(R>0, "invalid initial gas constant");
            state.gas.resize(state.gasMesh.volumes.size());
            for (std::size_t c = 0; c<state.gas.size(); ++c) {
                const auto b = cellBounds(state.gasMesh, c);
                rectangular(b, state.gasMesh.volumes[c]);
                if (kind == "SineDensity") {
                    const Vec3 k = readVector(init, "waveNumber"), mid = (b.lo+b.hi)*.5, width = b.hi-b.lo;
                    w.rho = readNumber(init, "mean")+readNumber(init, "amplitude")*std::sin(dot(k,
                        mid))*sinc(k.x*width.x*.5)*sinc(k.y*width.y*.5)*sinc(k.z*width.z*.5);
                } else {
                    const Real location = readNumber(init, "location"), length = b.hi.x-b.lo.x,
                        f = std::max(0.0, std::min(length, location-b.lo.x))/length;
                    w.rho = f*readNumber(init, "leftDensity")+(1-f)*readNumber(init, "rightDensity");
                }
                w.temperature = pressure/(w.rho*R);
                state.gas[c] = conservativeGas(w, state.gasMesh.volumes[c], model.physics);
                GasPrimitive recovered;
                require(recoverGas(state.gas[c], state.gasMesh.volumes[c], model.physics, recovered),
                    "invalid analytic initial gas cell average");
            }
            for (std::size_t f = 0; f<state.gasMesh.owner.size(); ++f) {
                GasPrimitive w0;
                const int c = state.gasMesh.owner[f];
                require(recoverGas(state.gas[c], state.gasMesh.volumes[c], model.physics, w0),
                    "initial boundary recovery failed");
                state.gasMesh.boundaryPrimitive[f] = w0;
            }
        }
        inline void initialFilm(const Foam::dictionary& d, const ModelConfig& model,
            const SurfaceImport& imported, HostState& state) {
            const auto& surface = d.subDict("surface");
            const auto& init = d.subDict(d.found("filmInitial")?"filmInitial":"initial");
            const auto kind = text(init, "kind");
            const auto& p = model.physics;
            const int n = imported.mesh.area.size();
            if (kind == "FilmSineTemperature") {
                require(p.liquid.cp1 == 0, "exact film sine average requires constant cp");
                const int axis = init.lookupOrDefault<Foam::label>("axis", 0);
                for (int face = 0; face<n; ++face) requireAxisAlignedRectangle(imported, face, axis);
            }
            state.surface = imported.mesh;
            state.film.resize(n);
            state.filmAux.resize(n);
            Real Y[Ns]{};
            array(surface, "Y", Y);
            const Real delta = readNumber(surface, "thickness"), pressure = readNumber(surface, "pressure");
            std::vector<Real> givenH, givenM;
            std::vector<std::vector<Real>> givenSpecies;
            if (kind == "FilmIntegralFields") {
                givenH = scalars(init, "enthalpy");
                givenM = scalars(init, "mass");
                require(givenH.size() == static_cast<std::size_t>(n)
                    && givenM.size() == static_cast<std::size_t>(n),
                    "film integral array size must match imported patch");
                for (int s = 0; s<Ns; ++s) {
                    givenSpecies.push_back(scalars(init, ("species_"+std::to_string(s)).c_str()));
                    require(givenSpecies.back().size() == static_cast<std::size_t>(n),
                        "film species array size mismatch");
                }
            } else require(kind == "FilmSineTemperature" || kind == "UniformFilmTemperature",
                "unsupported film initializer");
            for (int f = 0; f<n; ++f) {
                auto& q = state.film[f];
                auto& aux = state.filmAux[f];
                aux.baseVelocity = state.surface.baseVelocity[f];
                aux.normal = state.surface.normal[f];
                aux.area = state.surface.area[f];
                aux.pressure = pressure;
                if (kind == "FilmIntegralFields") {
                    q.mass = givenM[f];
                    q.enthalpy = givenH[f];
                    for (int s = 0; s<Ns; ++s)q.species[s] = givenSpecies[s][f];
                } else {
                    Real T = readNumber(init, "mean");
                    if (kind == "FilmSineTemperature") {
                        require(p.liquid.cp1 == 0, "exact film sine average requires constant cp");
                        const int axis = init.lookupOrDefault<Foam::label>("axis", 0);
                        requireAxisAlignedRectangle(imported, f, axis);
                        const auto& b = imported.bounds[f];
                        const Real a = component(b.lo, axis), z = component(b.hi, axis), k = readNumber(init,
                            "waveNumber");
                        require(z>a, "zero film cell extent");
                        T += readNumber(init, "amplitude")*std::sin(k*(a+z)/2)*sinc(k*(z-a)/2);
                    }
                    q.mass = p.liquid.rho*delta*aux.area;
                    q.enthalpy = q.mass*liquidH(p, T, pressure);
                    for (int s = 0; s<Ns; ++s)q.species[s] = q.mass*Y[s];
                }
                require(recoverFilm(q, aux.area, pressure, p, aux), "invalid initial film inventory");
            }
        }
        inline Real similarityLambda(Real value) {
            require(value>0, "inverseStefan must be positive");
            const Real pi = std::acos(-1.0);
            Real lo = 0, hi = 1;
            auto fun = [&](Real x) {
                return std::sqrt(pi)*x*std::exp(x*x)*std::erf(x);
            };
            while (fun(hi)<value) {
                hi *= 2;
                require(hi<16, "unsupported similarity parameter");
            }
            for (int i = 0; i<100; ++i) {
                Real mid = (lo+hi)*.5;
                if (fun(mid)<value)lo = mid;
                else hi = mid;
            }
            return (lo+hi)*.5;
        }
        inline void initialNormal(const Foam::fvMesh& source, const HostMesh& geometry,
            const Foam::dictionary& d, const ModelConfig& model, HostState& state) {
            const auto& column = d.subDict("normalColumn");
            const auto& init = d.subDict("initial");
            require(text(init, "kind") == "StefanSimilarity",
                "normal initialization currently requires explicit StefanSimilarity data");
            const int axis = column.lookup<Foam::label>("axis");
            require(axis >= 0 && axis<3, "invalid normal axis");
            const auto& bottom = source.boundaryMesh()[patchIndex(source, text(column, "bottomPatch"))];
            const auto& top = source.boundaryMesh()[patchIndex(source, text(column, "topPatch"))];
            require(bottom.size() == 1 && top.size() == 1,
                "normal column requires exactly one bottom/top geometric face");
            const int face = bottom.start();
            const Real area = mag(geometry.areaVectors[face]);
            const Vec3 normal = -normalized(geometry.areaVectors[face]);
            require(std::abs(component(normal, axis)-1)<1e-10,
                "normal column axis does not match imported bottom normal");
            auto& s = state.surface;
            s.gasFace = {-1};
            s.solidFace = {-1};
            s.solidCell = {-1};
            s.persistentId = {geometry.faceIds[face]};
            s.area = {area};
            s.oldArea = {area};
            s.centre = {geometry.faceCentres[face]};
            s.normal = {normal};
            s.meshVelocity = {{}};
            s.baseVelocity = {{}};
            s.gasDistance = {0};
            s.solidDistance = {0};
            s.normalOffsets = {0, static_cast<int>(geometry.volumes.size())};
            s.normalPressure = {readNumber(column, "pressure", 0)};
            s.normalPressureRate = {0};
            s.bottomTemperature = {readNumber(column, "bottomTemperature")};
            s.topTemperature = {readNumber(column, "topTemperature")};
            state.normalAcceptedWallHeat = {0};
            std::vector<int> sorted(geometry.volumes.size());
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                return component(geometry.cellCentres[a], axis)<component(geometry.cellCentres[b], axis);
            });
            const auto& p = model.physics;
            const int material = p.material.phaseCondensed;
            require(material >= 0 && material<Nc, "normal phase condensed index required");
            const auto& solid = p.condensed[material];
            require(solid.cp1 == 0 && p.liquid.cp1 == 0 && solid.cp0 == p.liquid.cp0,
                "similarity initial averages require equal constant cp");
            const Real front = readNumber(init, "initialFront"), lambda = similarityLambda(readNumber(init,
                "inverseStefan"));
            const Real a = front/lambda, origin = component(geometry.faceCentres[face], axis);
            const Real contrast = readNumber(init, "wallTemperature")-readNumber(init, "meltTemperature"),
                pi = std::acos(-1.0);
            const Real hs = condensedE(solid, p.meltTemperature)+s.normalPressure[0]/solid.rho,
                latent = liquidH(p, p.meltTemperature, s.normalPressure[0])-hs;
            auto integralErf = [&](Real x) {
                return x*std::erf(x/a)+a*std::exp(-x*x/(a*a))/std::sqrt(pi);
            };
            Real previous = origin;
            for (int cell:sorted) {
                const auto b = cellBounds(geometry, cell);
                rectangular(b, geometry.volumes[cell]);
                const Real left = component(b.lo, axis)-origin, right = component(b.hi, axis)-origin,
                    width = right-left;
                require(closeEnough(width*area, geometry.volumes[cell], 1e-13, 1e-10)
                    && closeEnough(left+origin, previous, 1e-13, 1e-10),
                    "normal cells do not form a contiguous constant-area column");
                previous = right+origin;
                const Real liquidEnd = std::max(left, std::min(right, front));
                const Real fraction = (liquidEnd-left)/width;
                const Real excess = contrast*((liquidEnd-left)-(integralErf(liquidEnd)-integralErf(left))/std::erf(lambda))/width;
                const Real mass = solid.rho*geometry.volumes[cell];
                s.normalCoordinates.push_back((left+right)*.5);
                s.normalThickness.push_back(width);
                s.normalLayerVolume.push_back(geometry.volumes[cell]);
                s.oldNormalLayerVolume.push_back(geometry.volumes[cell]);
                s.normalLayerMass.push_back(mass);
                state.normalLiquidFraction.push_back(fraction);
                state.normalEnthalpy.push_back(mass*(hs+fraction*latent+solid.cp0*excess));
            }
        }
    }
}
#endif
