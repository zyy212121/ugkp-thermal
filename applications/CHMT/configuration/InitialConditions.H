#ifndef CHMT_CONFIGURATION_INITIALCONDITIONS_H
#define CHMT_CONFIGURATION_INITIALCONDITIONS_H
#include "io/MeshIO.H"
namespace chmt { namespace io {
inline Real sinc(Real x) { return std::abs(x)<1e-8?1-x*x/6+x*x*x*x/120:std::sin(x)/x; }
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
}}
#endif
