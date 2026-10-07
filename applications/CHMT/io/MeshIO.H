#ifndef CHMT_IO_MESHIO_H
#define CHMT_IO_MESHIO_H
#include "configuration/InputValues.H"
#include "mesh/Geometry.H"
#include "core/ThermalBoundary.H"
#include "materials/MaterialCaloric.H"
#include "cyclicPolyPatch.H"
#include "emptyPolyPatch.H"
#include "wallPolyPatch.H"
#include <map>
#include <set>
namespace chmt {
    namespace io {
        inline Vec3 point(const Foam::vector& p) {
            return {p.x(), p.y(), p.z()};
        }
        inline Real component(Vec3 p, int axis) {
            return axis == 0?p.x:axis == 1?p.y:p.z;
        }
        struct Bounds {
            Vec3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
        };
        inline void include(Bounds& b, Vec3 p) {
            b.lo = {std::min(b.lo.x, p.x), std::min(b.lo.y, p.y), std::min(b.lo.z, p.z)};
            b.hi = {std::max(b.hi.x, p.x), std::max(b.hi.y, p.y), std::max(b.hi.z, p.z)};
        }
        inline Bounds cellBounds(const HostMesh& m, int cell) {
            Bounds b;
            for (int ci = m.cellFaceOffsets.at(cell); ci<m.cellFaceOffsets.at(cell+1); ++ci) {
                int f = m.cellFaces[ci];
                for (int i = m.faceOffsets[f]; i<m.faceOffsets[f+1]; ++i)include(b, m.points[m.facePoints[i]]);
            }
            return b;
        }
        inline Bounds faceBounds(const HostMesh& m, int face) {
            Bounds b;
            for (int i = m.faceOffsets.at(face); i<m.faceOffsets.at(face+1); ++i)include(b,
                m.points[m.facePoints[i]]);
            return b;
        }
        inline BoundaryKind boundaryKind(const std::string& value) {
            if (value == "Slip") {
                return BoundaryKind::Slip;
            }
            if (value == "NoSlip") {
                return BoundaryKind::NoSlip;
            }
            if (value == "Inlet") {
                return BoundaryKind::Inlet;
            }
            if (value == "Outlet") {
                return BoundaryKind::Outlet;
            }
            if (value == "Interface") {
                return BoundaryKind::Interface;
            }
            if (value == "Empty") {
                return BoundaryKind::Empty;
            }
            throw std::runtime_error("unsupported physical boundary kind "+value);
        }
        inline HostMesh importMesh(const Foam::fvMesh& mesh, const Foam::dictionary& properties) {
            HostMesh out;
            for (const auto& p:mesh.points())out.points.push_back(point(p));
            out.oldPoints = out.points;
            out.referencePoints = out.points;
            const auto& faces = mesh.faces();
            const auto& owners = mesh.faceOwner();
            const auto& neighbours = mesh.faceNeighbour();
            const int nf = faces.size();
            out.faceOffsets.push_back(0);
            out.neighbour.assign(nf, -1);
            out.periodicPartner.assign(nf, -1);
            out.boundaryKind.assign(nf, BoundaryKind::Internal);
            out.boundaryPrimitive.resize(nf);
            out.thermalBoundary.assign(nf,ThermalBoundaryKind::ZeroGradient);
            out.boundarySst.resize(nf);
            for (int f = 0; f<nf; ++f) {
                out.owner.push_back(owners[f]);
                if (f<neighbours.size())out.neighbour[f] = neighbours[f];
                for (auto v:faces[f])out.facePoints.push_back(v);
                out.faceOffsets.push_back(out.facePoints.size());
                out.faceIds.push_back(f);
            }
            const Foam::dictionary* prescribed = properties.subDictPtr("boundaries");
            for (const auto& patch:mesh.boundaryMesh()) {
                if (Foam::isA<Foam::cyclicPolyPatch>(patch)) {
                    const auto& cyclic = Foam::refCast<const Foam::cyclicPolyPatch>(patch);
                    require(!cyclic.transform().transforms(),
                        "rotational/scaled cyclic patches require a vector-transform backend; only translational cyclic is supported");
                    for (int i = 0; i<patch.size(); ++i) {
                        const int f = patch.start()+i;
                        out.boundaryKind[f] = BoundaryKind::Periodic;
                        out.periodicPartner[f] = cyclic.transformGlobalFace(f);
                    }
                } else {
                    require(!patch.coupled(), "unsupported coupled patch type "+std::string(patch.type()));
                    BoundaryKind kind = Foam::isA<Foam::emptyPolyPatch>(patch)?BoundaryKind::Empty:Foam::isA<Foam::wallPolyPatch>(patch)?BoundaryKind::NoSlip:BoundaryKind::Slip;
                    if (prescribed && prescribed->found(patch.name()))kind = boundaryKind(text(prescribed->subDict(patch.name()), "kind"));
                    for (int i = 0; i<patch.size(); ++i)out.boundaryKind[patch.start()+i] = kind;
                }
            }
            std::string error;
            const bool rebuilt = rebuildGeometry(out, error);
            require(rebuilt, "mesh import: "+error);
            out.oldVolumes = out.volumes;
            return out;
        }
        inline int patchIndex(const Foam::fvMesh& mesh, const std::string& name) {
            int i = mesh.boundaryMesh().findPatchID(name);
            require(i >= 0, "mesh has no patch "+name);
            return i;
        }
        struct SurfaceImport {
            SurfaceMesh mesh;
            std::vector<int> sourceFaces;
            std::vector<Bounds> bounds;
            std::vector<std::vector<Vec3>> vertices;
        };
        inline void requireAxisAlignedRectangle(const SurfaceImport& imported, int face, int axis) {
            const auto& points = imported.vertices.at(face);
            const auto& bounds = imported.bounds.at(face);
            const Vec3 extent = bounds.hi-bounds.lo;
            const Real scale = std::max(1.0, mag(extent)), tolerance = 1e-11*scale;
            require(points.size() == 4 && axis >= 0 && axis<3,
                "FilmSineTemperature requires an axis-aligned rectangle");
            std::vector<int> tangentAxes;
            for (int a = 0; a<3; ++a) if (component(extent, a)>tolerance) tangentAxes.push_back(a);
            require(tangentAxes.size() == 2 && component(extent, axis)>tolerance,
                "FilmSineTemperature requires an axis-aligned rectangle with a tangential sine axis");
            const Real rectangleArea = component(extent, tangentAxes[0])*component(extent, tangentAxes[1]);
            require(closeEnough(rectangleArea, imported.mesh.area.at(face), 1e-13*scale*scale, 1e-11),
                "FilmSineTemperature requires an axis-aligned rectangle, not a bounding-box approximation");
            std::set<int> corners;
            std::vector<int> ordered;
            for (Vec3 point:points) {
                int mask = 0;
                for (int j = 0; j<2; ++j) {
                    const int a = tangentAxes[j];
                    const Real coordinate = component(point, a);
                    if (std::abs(coordinate-component(bounds.hi, a)) <= tolerance) mask|=1<<j;
                    else require(std::abs(coordinate-component(bounds.lo, a)) <= tolerance,
                        "FilmSineTemperature requires axis-aligned rectangle vertices");
                }
                require(corners.insert(mask).second, "duplicate rectangle corner");
                ordered.push_back(mask);
            }
            for (int j = 0; j<4; ++j) {
                const int changed = ordered[j]^ordered[(j+1)%4];
                require(changed == 1 || changed == 2, "noncyclic rectangle corner order");
            }
        }
        inline void initializeDryInterfaceAux(HostState& state) {
            const auto& surface = state.surface;
            require(surface.area.size() == surface.normal.size()
                && surface.area.size() == surface.baseVelocity.size(), "incomplete dry interface geometry");
            require(state.film.empty(), "dry interface must not fabricate film inventory");
            state.filmAux.resize(surface.area.size());
            for (std::size_t face = 0; face<surface.area.size(); ++face) {
                auto& auxiliary = state.filmAux[face];
                auxiliary.area = surface.area[face];
                auxiliary.normal = surface.normal[face];
                auxiliary.baseVelocity = surface.baseVelocity[face];
                auxiliary.meanVelocity = auxiliary.baseVelocity;
                auxiliary.topVelocity = auxiliary.baseVelocity;
                auxiliary.thickness = 0;
                auxiliary.temperature = undefinedValue();
            }
        }
        inline void attachCoupledSurface(const Foam::fvMesh& gasSource, const Foam::dictionary& d,
            HostState& state) {
            const auto& specification = d.subDict("surface");
            const auto& patch = gasSource.boundaryMesh()[patchIndex(gasSource, text(specification, "gasPatch"))];
            auto& surface = state.surface;
            require(state.filmAux.size() == surface.area.size(),
                "coupled surface requires explicit dry/film geometry auxiliaries");
            std::vector<Real> thickness;
            for (const auto& auxiliary:state.filmAux) thickness.push_back(auxiliary.thickness);
            std::vector<int> candidates;
            for (int i = 0; i<patch.size(); ++i) candidates.push_back(patch.start()+i);
            std::vector<int> mapping, vertices;
            std::string error;
            require(mapConformalInterface(state.gasMesh, state.solidMesh, surface.solidFace, thickness,
                candidates, mapping, vertices, error), error);
            surface.gasFace = mapping;
            for (std::size_t face = 0; face<mapping.size(); ++face) {
                const int gasFace = mapping[face];
                const Vec3 gasNormal = normalized(state.gasMesh.areaVectors[gasFace]);
                surface.gasDistance[face] = std::abs(dot( state.gasMesh.cellCentres[state.gasMesh.owner[gasFace]] -state.gasMesh.faceCentres[gasFace], gasNormal));
                require(surface.gasDistance[face]>0, "nonpositive actual gas interface distance");
                require(!fixedTemperature(state.gasMesh.thermalBoundary.empty()?nullptr:state.gasMesh.thermalBoundary.data(),gasFace)
                    && !fixedTemperature(state.solidMesh.thermalBoundary.empty()?nullptr:state.solidMesh.thermalBoundary.data(),surface.solidFace[face]),
                    "coupled Interface owns thermal exchange; use zeroGradient temperature input on its patches");
                state.gasMesh.boundaryKind[gasFace] = BoundaryKind::Interface;
                state.solidMesh.boundaryKind[surface.solidFace[face]] = BoundaryKind::Interface;
            }
        }
        inline SurfaceImport importSurface(const Foam::fvMesh& source, const HostMesh& geometry,
            const Foam::dictionary& d, bool standalone) {
            SurfaceImport result;
            auto& s = result.mesh;
            const auto& patch = source.boundaryMesh()[patchIndex(source, text(d, "patch"))];
            require(patch.size()>0, "selected surface patch is empty");
            const Vec3 base = readVector(d, "baseVelocity"), traction = readVector(d, "prescribedTopTraction"),
                gradient = readVector(d, "prescribedPressureGradient");
            std::map<std::pair<int, int>, std::vector<std::pair<int, std::pair<int, int>>>> edges;
            for (int i = 0; i<patch.size(); ++i) {
                const int f = patch.start()+i;
                result.sourceFaces.push_back(f);
                result.bounds.push_back(faceBounds(geometry, f));
                const Real area = mag(geometry.areaVectors[f]);
                const Vec3 normal = geometry.areaVectors[f]/area;
                require(std::abs(dot(base, normal))<1e-10*std::max(1.0, mag(base)),
                    "surface base velocity must be tangential");
                require(std::abs(dot(traction, normal))<1e-10*std::max(1.0, mag(traction))
                    && std::abs(dot(gradient, normal))<1e-10*std::max(1.0, mag(gradient)),
                    "prescribed surface drive must be tangential");
                require(standalone || (mag(traction) == 0 && mag(gradient) == 0),
                    "prescribed and gas-computed driving cannot be mixed");
                s.gasFace.push_back(-1);
                s.solidFace.push_back(standalone?-1:f);
                s.solidCell.push_back(standalone?-1:geometry.owner[f]);
                s.persistentId.push_back(geometry.faceIds[f]);
                s.area.push_back(area);
                s.oldArea.push_back(area);
                s.centre.push_back(geometry.faceCentres[f]);
                s.normal.push_back(normal);
                s.baseVelocity.push_back(base);
                s.meshVelocity.push_back({});
                s.gasDistance.push_back(0);
                s.solidDistance.push_back(standalone?0:std::abs(dot(geometry.faceCentres[f]-geometry.cellCentres[geometry.owner[f]], normal)));
                s.prescribedTopTraction.push_back(traction);
                s.prescribedPressureGradient.push_back(gradient);
                const auto& face = source.faces()[f];
                result.vertices.emplace_back();
                for (int vertex:face) result.vertices.back().push_back(geometry.points[vertex]);
                for (int k = 0; k<face.size(); ++k) {
                    int a = face[k], b = face[(k+1)%face.size()];
                    edges[std::minmax(a, b)].push_back({i, {a, b}});
                }
            }
            // Seam matches come from OpenFOAM cyclic point addressing, not coordinate/n labels.
            std::vector<std::map<int, int>> cyclicMaps;
            for (const auto& p:source.boundaryMesh())if (Foam::isA<Foam::cyclicPolyPatch>(p)) {
                const auto& c = Foam::refCast<const Foam::cyclicPolyPatch>(p);
                std::map<int, int> map;
                const auto& own = c.meshPoints();
                const auto& other = c.nbrPatch().meshPoints();
                for (const auto& pair:c.coupledPoints())map[own[pair[0]]] = other[pair[1]];
                cyclicMaps.push_back(std::move(map));
            }
            std::set<std::pair<int, int>> consumed;
            for (const auto& edge:edges) {
                if (consumed.count(edge.first)) {
                    continue;
                }
                const auto& adjacent = edge.second;
                require(adjacent.size() <= 2, "nonmanifold film edge");
                int own = adjacent[0].first, neighbour = -1;
                Vec3 neighbourEdge{};
                const int a = adjacent[0].second.first, b = adjacent[0].second.second;
                const Vec3 centre = (geometry.points[a]+geometry.points[b])*.5;
                if (adjacent.size() == 2) {
                    neighbour = adjacent[1].first;
                    neighbourEdge = centre;
                } else for (const auto& map:cyclicMaps) {
                    auto ia = map.find(a), ib = map.find(b);
                    if (ia == map.end() || ib == map.end())continue;
                    const std::pair<int, int> key = std::minmax(ia->second, ib->second);
                    auto match = edges.find(key);
                    if (match != edges.end() && key != edge.first && match->second.size() == 1) {
                        require(neighbour<0 || neighbour == match->second[0].first, "ambiguous cyclic film seam");
                        neighbour = match->second[0].first;
                        neighbourEdge = (geometry.points[ia->second]+geometry.points[ib->second])*.5;
                        consumed.insert(key);
                    }
                }
                const Vec3 tangent = geometry.points[b]-geometry.points[a];
                const Real length = mag(tangent);
                require(length>0, "zero-length surface edge");
                Vec3 conormal = normalized(cross(tangent, s.normal[own]));
                if (dot(conormal, centre-s.centre[own])<0)conormal = -conormal;
                s.edgeOwner.push_back(own);
                s.edgeNeighbour.push_back(neighbour);
                s.edgeLength.push_back(length);
                s.edgeConormal.push_back(conormal);
                s.edgeOwnerOffset.push_back(centre-s.centre[own]);
                s.edgeNeighbourOffset.push_back(neighbour >= 0?neighbourEdge-s.centre[neighbour]:Vec3{});
                s.sweptEdgeArea.push_back(0);
                consumed.insert(edge.first);
            }
            return result;
        }
        inline Foam::volScalarField scalarField(const Foam::fvMesh& mesh, const Foam::word& name,
            const Foam::dimensionSet& dimensions) {
            Foam::volScalarField f(Foam::IOobject(name, mesh.time().timeName(), mesh, Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE), mesh);
            require(f.dimensions() == dimensions, "field dimensions mismatch for "+std::string(name));
            return f;
        }
        inline void importGasFields(const Foam::fvMesh& mesh, const ModelConfig& model, HostState& state) {
            auto rho = scalarField(mesh, "rho", Foam::dimDensity);
            auto T = scalarField(mesh, "T", Foam::dimTemperature);
            Foam::volVectorField U(Foam::IOobject("U", mesh.time().timeName(), mesh, Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE), mesh);
            require(U.dimensions() == Foam::dimVelocity, "U dimensions mismatch");
            Foam::PtrList<Foam::volScalarField> Y(Ns);
            for (int s = 0; s<Ns; ++s)Y.set(s, new Foam::volScalarField(scalarField(mesh,
                "Y_"+model.speciesNames[s], Foam::dimless)));
            state.gasMesh.thermalBoundary.assign(mesh.nFaces(),ThermalBoundaryKind::ZeroGradient);
            for(int patch=0;patch<mesh.boundaryMesh().size();++patch) {
                for(int f=0;f<mesh.boundaryMesh()[patch].size();++f) {
                    const int face=mesh.boundaryMesh()[patch].start()+f;
                    state.gasMesh.thermalBoundary[face]=importedThermalBoundary(
                        T.boundaryField()[patch].type(),state.gasMesh.boundaryKind[face],true);
                }
            }
            // Evaluate zeroGradient from its owner before importing a valid EOS
            // primitive. The operator itself is retained independently above.
            rho.correctBoundaryConditions();T.correctBoundaryConditions();U.correctBoundaryConditions();
            for(int species=0;species<Ns;++species)Y[species].correctBoundaryConditions();
            state.gas.resize(mesh.nCells());
            auto primitive = [&](int cell, int patch, int face) {
                GasPrimitive w;
                if (patch<0) {
                    w.rho = rho[cell];
                    w.temperature = T[cell];
                    w.velocity = point(U[cell]);
                    for (int s = 0; s<Ns; ++s)w.Y[s] = Y[s][cell];
                } else {
                    w.rho = rho.boundaryField()[patch][face];
                    w.temperature = T.boundaryField()[patch][face];
                    w.velocity = point(U.boundaryField()[patch][face]);
                    for (int s = 0; s<Ns; ++s)w.Y[s] = Y[s].boundaryField()[patch][face];
                }
                GasPrimitive recovered;
                const auto q = conservativeGas(w, 1, model.physics);
                require(recoverGas(q, 1, model.physics, recovered), "inadmissible imported gas primitive");
                return recovered;
            };
            for (int c = 0; c<mesh.nCells(); ++c)state.gas[c] = conservativeGas(primitive(c, -1, -1),
                state.gasMesh.volumes[c], model.physics);
            for (int patch = 0; patch<mesh.boundaryMesh().size(); ++patch)for (int f = 0; f<mesh.boundaryMesh()[patch].size(); ++f) {
                const int face=mesh.boundaryMesh()[patch].start()+f;
                // Empty poly patches have geometric faces but zero field values.
                // Their unused per-face storage gets a valid owner EOS state.
                state.gasMesh.boundaryPrimitive[face]=state.gasMesh.boundaryKind[face]==BoundaryKind::Empty
                    ?primitive(state.gasMesh.owner[face],-1,-1):primitive(-1,patch,f);
            }
            if (model.physics.enableSst) {
                auto k = scalarField(mesh, "k", Foam::sqr(Foam::dimVelocity));
                auto omega = scalarField(mesh, "omega", Foam::dimless/Foam::dimTime);
                state.sst.resize(mesh.nCells());
                for (int c = 0; c<mesh.nCells(); ++c)state.sst[c] = {state.gas[c].mass*k[c],
                    state.gas[c].mass*omega[c]};
                for (int p = 0; p<mesh.boundaryMesh().size(); ++p)for (int f = 0; f<mesh.boundaryMesh()[p].size(); ++f) {
                    const int face=mesh.boundaryMesh()[p].start()+f;
                    auto& w = state.gasMesh.boundarySst[face];
                    if(state.gasMesh.boundaryKind[face]==BoundaryKind::Empty) {
                        const int owner=state.gasMesh.owner[face];w.k=k[owner];w.omega=omega[owner];
                    } else {
                        w.k = k.boundaryField()[p][f];
                        w.omega = omega.boundaryField()[p][f];
                    }
                }
            }
        }
        inline void importParticles(const Foam::dictionary& d, const ModelConfig& model, HostState& state) {
            if (!model.physics.enableParticles) {
                require(!d.found("particles"), "particle input supplied while particles are disabled");
                return;
            }
            require(model.physics.enableGas && !state.gas.empty(), "dilute particles require imported gas");
            const auto& policy = d.subDict("particle");
            require(text(policy, "volumeMode") == "DilutePoint",
                "only explicit DilutePoint particle volumeMode is supported");
            std::vector<Real> fractions(state.gas.size(), 1);
            if (policy.found("voidFractionValues")) fractions = scalars(policy, "voidFractionValues");
            require(fractions.size() == state.gas.size(),
                "particle void-fraction count must match imported gas cells");
            for (Real fraction:fractions) require(fraction == 1,
                "occupied-volume/dense particle fractions are unsupported; DilutePoint requires all ones");
            std::vector<ParticleQ> particles;
            if (d.found("particles")) {
                const auto& dictionary = d.subDict("particles");
                for (const auto& key:dictionary.toc()) {
                    const auto& value = dictionary.subDict(key);
                    ParticleQ p;
                    p.id = std::stoull(text(value, "id"));
                    p.cell = value.lookup<Foam::label>("cell");
                    p.face = value.lookupOrDefault<Foam::label>("face", -1);
                    p.state = value.lookupOrDefault<Foam::label>("state", 0);
                    p.position = readVector(value, "position");
                    p.velocity = readVector(value, "velocity");
                    p.mass = readNumber(value, "mass");
                    p.energy = readNumber(value, "thermalEnergy");
                    p.diameter = readNumber(value, "diameter");
                    p.contactAge = readNumber(value, "contactAge", 0);
                    p.contactDuration = readNumber(value, "contactDuration", 0);
                    p.contactArea = readNumber(value, "contactArea", 0);
                    p.contactRelativeVelocity = readVector(value, "contactRelativeVelocity");
                    p.contactMechanicalEnergy = readNumber(value, "contactMechanicalEnergy", 0);
                    p.rng = std::stoull(text(value, "rng", "1"));
                    particles.push_back(p);
                }
            }
            state.gasVoidFraction.swap(fractions);
            state.particles.swap(particles);
        }
        inline void importSolidFields(const Foam::fvMesh& mesh, const ModelConfig& model, HostState& state) {
            const int n = mesh.nCells();
            state.solid.resize(n);
            auto energy = scalarField(mesh, "solidEnergyDensity", Foam::dimEnergy/Foam::dimVolume);
            auto porosity = scalarField(mesh, "porosity", Foam::dimless);
            // Energy density initializes cell inventory; it is not a thermal
            // boundary prescription and must not masquerade as a temperature.
            for(int patch=0;patch<mesh.boundaryMesh().size();++patch) {
                const std::string type=energy.boundaryField()[patch].type();
                require(type=="zeroGradient" || type=="calculated" || type=="cyclic" || type=="empty",
                    "solidEnergyDensity boundary operator "+type+" is unsupported; prescribe fixed temperature with solidTemperature [K]");
            }
            for (int c = 0; c<n; ++c) {
                state.solid[c].energy = energy[c]*state.solidMesh.volumes[c];
                state.solid[c].porosity = porosity[c];
            }
            for (int j = 0; j<Nc; ++j) {
                auto rho = scalarField(mesh, "rho_"+model.condensedNames[j], Foam::dimDensity);
                for (int c = 0; c<n; ++c)state.solid[c].condensed[j] = rho[c]*state.solidMesh.volumes[c];
            }
            for (int j = 0; j<Ns; ++j) {
                auto rho = scalarField(mesh, "rhoPore_"+model.speciesNames[j], Foam::dimDensity);
                for (int c = 0; c<n; ++c)state.solid[c].pore[j] = rho[c]*state.solidMesh.volumes[c];
            }
            for (int r = 0; r<model.physics.nReactions; ++r) {
                auto extent = scalarField(mesh, "reactionExtentDensity_"+std::to_string(r),
                    Foam::dimMoles/Foam::dimVolume);
                for (int c = 0; c<n; ++c)state.solid[c].progress[r] = extent[c]*state.solidMesh.volumes[c];
            }
            for (const auto& q:state.solid) {
                Real T;
                require(recoverSolid(q, model.physics, T), "inadmissible imported solid state");
            }
            state.solidMesh.thermalBoundary.assign(mesh.nFaces(),ThermalBoundaryKind::ZeroGradient);
            Foam::IOobject thermalHeader("solidTemperature",mesh.time().timeName(),mesh,
                Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE);
            if(Foam::isFile(thermalHeader.objectPath(false)) || Foam::isFile(thermalHeader.objectPath(false)+".gz")) {
                auto thermal=scalarField(mesh,"solidTemperature",Foam::dimTemperature);
                for(int patch=0;patch<mesh.boundaryMesh().size();++patch)for(int f=0;f<mesh.boundaryMesh()[patch].size();++f) {
                    const int face=mesh.boundaryMesh()[patch].start()+f,owner=state.solidMesh.owner[face];
                    const auto policy=importedThermalBoundary(thermal.boundaryField()[patch].type(),state.solidMesh.boundaryKind[face],false);
                    state.solidMesh.thermalBoundary[face]=policy;
                    Real T=0;require(recoverSolid(state.solid[owner],model.physics,T),"solid thermal owner EOS failed");
                    if(policy==ThermalBoundaryKind::FixedValue) {
                        T=thermal.boundaryField()[patch][f];MaterialCaloric caloric;std::string error;
                        require(materialCaloric(state.solid[owner],model.physics,caloric,error),error);
                        require(finite(T) && T>=caloric.minimumTemperature && T<=caloric.maximumTemperature,
                            "solidTemperature fixedValue outside active owner-composition caloric range");
                    }
                    state.solidMesh.boundaryPrimitive[face].temperature=T;
                }
            }
        }
    }
}
#endif
