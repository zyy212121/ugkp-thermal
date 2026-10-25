// Frozen serial geometry oracle from PR25 63158141dd82c88d60e848463271a1eeea64da08.
// Only the two function names and internal geometry calls are renamed. Include
// after Geometry.C to reuse its unchanged geometry primitives. Keeping this
// test-only oracle avoids requiring Git history or a particular compiler's hashes.
namespace chmt {
namespace {
        bool referenceGeometryInPlace(HostMesh& m, std::string& error,empty_detail::Extrusion* extrusion=nullptr,bool* numericalFailure=nullptr,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
            if(numericalFailure)*numericalFailure=true;
            if(extrusion)*extrusion=empty_detail::Extrusion{};
            const int nf = static_cast<int>(m.owner.size()), np = static_cast<int>(m.points.size());
            if (nf == 0 && np == 0) {
                m.volumes.clear();
                m.cellCentres.clear();
                m.areaVectors.clear();
                m.faceCentres.clear();
                m.cellFaceOffsets = {0};
                m.cellFaces.clear();
                m.cellFaceSigns.clear();
                return true;
            }

            if ((nf<1 || np<4 || m.neighbour.size() != m.owner.size()
                || m.boundaryKind.size() != m.owner.size() || m.faceOffsets.size() != m.owner.size()+1
                || m.faceOffsets.front() != 0
                || m.faceOffsets.back() != static_cast<int>(m.facePoints.size())))return errorAt(error,
                "incomplete polyhedral topology", -1);
            for (int f = 0; f<nf; ++f)if ((m.faceOffsets[f]<0 || m.faceOffsets[f+1]<m.faceOffsets[f]
                || m.faceOffsets[f+1]>static_cast<int>(m.facePoints.size())))return errorAt(error,
                "invalid face CSR offset", f);
            if (m.periodicPartner.empty())m.periodicPartner.assign(nf, -1);
            if ((m.periodicPartner.size() != m.owner.size()))return errorAt(error, "invalid periodic array", -1);
            if (m.faceIds.empty()) {
                m.faceIds.resize(nf);
                for (int f = 0; f<nf; ++f)m.faceIds[f] = f;
            }
            if ((m.faceIds.size() != m.owner.size()))return errorAt(error, "invalid persistent face IDs", -1);
            std::set<std::uint64_t> ids;
            for (auto id:m.faceIds)if ((!ids.insert(id).second))return errorAt(error,
                "duplicate persistent face ID", -1);
            for (int i = 0; i<np; ++i)if ((!finite(m.points[i])))return errorAt(error, "nonfinite vertex", i);
            int nc = 0;
            for (int f = 0; f<nf; ++f) {
                if ((m.owner[f]<0 || m.owner[f] >= nf || m.neighbour[f] >= nf || m.neighbour[f] == m.owner[f]
                    || m.neighbour[f]<-1))return errorAt(error, "invalid cell incidence", f);
                nc = std::max(nc, 1+std::max(m.owner[f], m.neighbour[f]));
                if (((m.neighbour[f] >= 0) != (m.boundaryKind[f] == BoundaryKind::Internal)))return errorAt(error,
                    "boundary/internal incidence mismatch", f);
                if ((m.faceOffsets[f+1]-m.faceOffsets[f]<3))return errorAt(error,
                    "face has fewer than three vertices", f);
                std::set<int> distinct;
                for (int k = m.faceOffsets[f]; k<m.faceOffsets[f+1]; ++k) {
                    int v = m.facePoints[k];
                    if ((v<0 || v >= np || !distinct.insert(v).second))return errorAt(error,
                        "invalid/duplicate face vertex", f);
                }
            }
            m.cellFaceOffsets.assign(nc+1, 0);
            for (int f = 0; f<nf; ++f) {
                ++m.cellFaceOffsets[m.owner[f]+1];
                if (m.neighbour[f] >= 0)++m.cellFaceOffsets[m.neighbour[f]+1];
            }
            for (int c = 0; c<nc; ++c)m.cellFaceOffsets[c+1] += m.cellFaceOffsets[c];
            m.cellFaces.resize(m.cellFaceOffsets.back());
            m.cellFaceSigns.resize(m.cellFaces.size());
            std::vector<int> cursor = m.cellFaceOffsets;
            for (int f = 0; f<nf; ++f) {
                int k = cursor[m.owner[f]]++;
                m.cellFaces[k] = f;
                m.cellFaceSigns[k] = 1;
                if (m.neighbour[f] >= 0) {
                    k = cursor[m.neighbour[f]]++;
                    m.cellFaces[k] = f;
                    m.cellFaceSigns[k] = -1;
                }
            }
            m.areaVectors.assign(nf, {});
            m.faceCentres.assign(nf, {});
            for (int f = 0; f<nf; ++f) {
                const int begin = m.faceOffsets[f], end = m.faceOffsets[f+1];
                const Vec3 a = m.points[m.facePoints[begin]];
                for (int j = begin+1; j+1<end; ++j)m.areaVectors[f] += triangleArea(a,
                    m.points[m.facePoints[j]], m.points[m.facePoints[j+1]]);
                const Real area = mag(m.areaVectors[f]);
                if ((!(area>0) || !finite(area)))return errorAt(error, "degenerate face area", f);
                const Vec3 n = m.areaVectors[f]/area;
                Real weight = 0;
                for (int j = begin+1; j+1<end; ++j) {
                    const Vec3 b = m.points[m.facePoints[j]], c = m.points[m.facePoints[j+1]];
                    const Real w = dot(triangleArea(a, b, c), n);
                    m.faceCentres[f] += (a+b+c)*(w/3);
                    weight += w;
                }
                m.faceCentres[f] = m.faceCentres[f]/weight;
            }
            m.volumes.assign(nc, 0);
            m.cellCentres.assign(nc, {});
            for (int c = 0; c<nc; ++c) {
                std::set<int> vertices;
                std::map<std::pair<int, int>, std::pair<int, int>> edges;
                Vec3 closure{};
                Real areaSum = 0;
                for (int k = m.cellFaceOffsets[c]; k<m.cellFaceOffsets[c+1]; ++k) {
                    int f = m.cellFaces[k], sign = m.cellFaceSigns[k], begin = m.faceOffsets[f],
                        end = m.faceOffsets[f+1];
                    {closure += m.areaVectors[f]*sign;
                    areaSum += mag(m.areaVectors[f]);}
                    for (int j = begin; j<end; ++j) {
                        int a = m.facePoints[j], b = m.facePoints[j+1 == end?begin:j+1];
                        vertices.insert(a);
                        {auto& count = edges[std::minmax(a, b)];
                        ++count.first;
                        count.second += (a<b?1:-1)*sign;}
                    }
                }
                if (vertices.size()<4)return errorAt(error,"degenerate cell geometry",c);
                if(!finite(closure)||!finite(areaSum))return errorAt(error,"nonfinite cell area closure",c);
                if(mag(closure)>1e-10*areaSum)geometryWarning(diagnostics,mag(closure),1e-10*areaSum,c,true,ugkp::diagnostics::EntityKind::Cell);
                for (const auto& edge:edges)if ((edge.second.first != 2
                    || edge.second.second != 0))return errorAt(error, "nonclosed/nonmanifold cell edges", c);
                Vec3 ref{};
                for (int v:vertices)ref += m.points[v];
                ref = ref/static_cast<Real>(vertices.size());
                Vec3 moment{};
                Real vol = 0;
                for (int k = m.cellFaceOffsets[c]; k<m.cellFaceOffsets[c+1]; ++k) {
                    int f = m.cellFaces[k], sign = m.cellFaceSigns[k], begin = m.faceOffsets[f],
                        end = m.faceOffsets[f+1];
                    Vec3 a = m.points[m.facePoints[begin]];
                    for (int j = begin+1; j+1<end; ++j) {
                        Vec3 b = m.points[m.facePoints[j]], d = m.points[m.facePoints[j+1]];
                        Real tetra = sign*dot(a-ref, cross(b-ref, d-ref))/6;
                        vol += tetra;
                        moment += (ref+a+b+d)*(tetra/4);
                    }
                }
                if ((!finite(vol) || vol <= 0))return errorAt(error, "inverted/nonpositive cell volume", c);
                m.volumes[c] = vol;
                m.cellCentres[c] = moment/vol;
                if ((!finite(m.cellCentres[c])))return errorAt(error, "nonfinite cell centroid", c);
            }
            if (!periodicGeometry(m, error))return false;
            m.wallDistance.assign(nc, 0);
            std::vector<int> wallFaces;
            bool hasEmpty=false;
            for (int f = 0; f<nf; ++f) {
                const auto kind = m.boundaryKind[f];
                hasEmpty=hasEmpty || kind==BoundaryKind::Empty;
                if (kind == BoundaryKind::NoSlip || kind == BoundaryKind::Slip
                    || kind == BoundaryKind::Interface)wallFaces.push_back(f);
            }
            if(hasEmpty) {
                empty_detail::Extrusion representation;
                if(!(empty_detail::build(m,representation,error)))return false;
                if(extrusion)*extrusion=std::move(representation);
            }
            if (!wallFaces.empty())for (int c = 0; c<nc; ++c) {
                Real distance = std::numeric_limits<Real>::max();
                for (int f:wallFaces) {
                    int begin = m.faceOffsets[f], end = m.faceOffsets[f+1];
                    for (int j = begin+1; j+1<end; ++j) {
                        const Real triangle=triangleDistance(m.cellCentres[c], m.points[m.facePoints[begin]],
                            m.points[m.facePoints[j]], m.points[m.facePoints[j+1]]);
                        if(!finite(triangle))return errorAt(error,"nonfinite wall triangle distance",f);
                        distance=std::min(distance,triangle);
                    }
                }
                if ((!(distance>0) || !finite(distance)))return errorAt(error, "invalid wall distance", c);
                m.wallDistance[c] = distance;
            }
            std::uint64_t h = 14695981039346656037ULL;
            auto mix = [&](std::uint64_t v) {
                for (int k = 0; k<8; ++k) {
                    h^=(v>>(8*k))&255;
                    h *= 1099511628211ULL;
                }
            };
            for (auto v:m.faceOffsets) {
                mix(v);
            }
            for (auto v:m.facePoints) {
                mix(v);
            }
            for (auto v:m.owner) {
                mix(v);
            }
            for (auto v:m.neighbour) {
                mix(v);
            }
            for (auto v:m.periodicPartner) {
                mix(v);
            }
            for (auto v:m.boundaryKind) {
                mix(static_cast<unsigned>(v));
            }
            m.topologyHash = h;
            error.clear();
            return true;
        }
    bool referenceMakeStageGeometry(const HostMesh& accepted, const std::vector<Vec3>& endpoint, double interval,
        HostMesh& output, std::vector<double>& outputSweep, std::string& error, const Tolerances& tol,bool* numericalFailure) {
        if(numericalFailure)*numericalFailure=true;

        if ((!finite(tol.absoluteGeometry) || tol.absoluteGeometry<0
            || !finite(tol.relativeGeometry) || tol.relativeGeometry<0))
            return errorAt(error, "invalid stage geometry tolerances", -1);
        if ((!finite(interval) || interval <= 0))return errorAt(error, "invalid stage interval", -1);
        if ((endpoint.size() != accepted.points.size()))return errorAt(error,
            "topology-changing point count unsupported", -1);
        HostMesh old = accepted;
        empty_detail::Extrusion extrusion;
        if (!referenceGeometryInPlace(old, error,&extrusion,numericalFailure,tol.diagnostics) || (!empty_detail::trajectory(old,extrusion,endpoint,error)))return false;
        HostMesh candidate = old;
        candidate.points = endpoint;
        if (!referenceGeometryInPlace(candidate, error,nullptr,numericalFailure,tol.diagnostics))return false;
        if ((candidate.topologyHash != old.topologyHash))return errorAt(error, "topology change unsupported", -1);
        {
        if (!certifyLinearTrajectory(old, endpoint, error,tol.diagnostics))return false;
        // Reject a crossed/inverted path even if endpoint orientation recovered.
        HostMesh mid = old;
        for (std::size_t p = 0; p<endpoint.size(); ++p)mid.points[p] = extrusion.axis<0?
            (old.points[p]+endpoint[p])*.5:old.points[p]+(endpoint[p]-old.points[p])*.5;
        if (!referenceGeometryInPlace(mid, error,nullptr,numericalFailure,tol.diagnostics))return false;
        // Cell volume is a cubic polynomial along linear vertex trajectories.
        // Check its interior extrema as well as endpoints, so two inversions
        // between sampling times cannot escape an endpoint/midpoint-only test.
        HostMesh oneThird = old, twoThird = old;
        for (std::size_t p = 0; p<endpoint.size(); ++p) {
            oneThird.points[p] = extrusion.axis<0?old.points[p]*(2.0/3)+endpoint[p]/3:
                old.points[p]+(endpoint[p]-old.points[p])/3;
            twoThird.points[p] = extrusion.axis<0?old.points[p]/3+endpoint[p]*(2.0/3):
                old.points[p]+(endpoint[p]-old.points[p])*(2.0/3);
        }
        if (!referenceGeometryInPlace(oneThird, error,nullptr,numericalFailure,tol.diagnostics) || !referenceGeometryInPlace(twoThird, error,nullptr,numericalFailure,tol.diagnostics))return false;
        for (int c = 0; c<static_cast<int>(old.volumes.size()); ++c) {
            Real v0 = old.volumes[c], v1 = oneThird.volumes[c], v2 = twoThird.volumes[c],
                v3 = candidate.volumes[c];
            Real a = 4.5*(v3-3*v2+3*v1-v0), b = 4.5*(v2-2*v1+v0)-a, d = 3*(v1-v0)-a/9-b/3;
            const Real small = 256*std::numeric_limits<Real>::epsilon()*maxValue(v0, v3);
            Real critical[2] = {-1, -1};
            if (absValue(a) <= small) {
                if (absValue(b)>small)critical[0] = -d/(2*b);
            } else {
                Real disc = 4*b*b-12*a*d;
                if (disc >= 0) {
                    Real root = ::sqrt(disc);
                    critical[0] = (-2*b-root)/(6*a);
                    critical[1] = (-2*b+root)/(6*a);
                }
            }
            for (Real t:critical)if ((t>0 && t<1 && ((a*t+b)*t+d)*t+v0 <= 0))return errorAt(error,
                "cell inversion inside linear stage trajectory", c);
        }
        }
        std::vector<Vec3> area;
        std::vector<Real>sweep;
        if (!stageIntegrals(old, endpoint, area, sweep, error,extrusion.axis>=0?&extrusion:nullptr,numericalFailure))return false;
        {
        for (int c = 0; c<static_cast<int>(old.volumes.size()); ++c) {
            Real sum = 0;
            Vec3 closure{};
            Real a = 0;
            for (int k = old.cellFaceOffsets[c]; k<old.cellFaceOffsets[c+1]; ++k) {
                int f = old.cellFaces[k], sign = old.cellFaceSigns[k];
                sum += sign*sweep[f];
                closure += area[f]*sign;
                a += mag(area[f]);
            }
            const Real limit = tol.absoluteGeometry+tol.relativeGeometry*std::max(old.volumes[c],
                candidate.volumes[c]);
            if(!finite(closure)||!finite(a))return errorAt(error,"nonfinite stage area closure",c);
            if (mag(closure)>1e-10*a)geometryWarning(tol.diagnostics,mag(closure),1e-10*a,c,true,ugkp::diagnostics::EntityKind::Cell);
            if (absValue(candidate.volumes[c]-old.volumes[c]-sum)>limit)
                geometryWarning(tol.diagnostics,candidate.volumes[c]-old.volumes[c]-sum,limit,c,true,ugkp::diagnostics::EntityKind::Cell);
        }
        for (int f = 0; f<static_cast<int>(old.owner.size()); ++f)if (old.boundaryKind[f] == BoundaryKind::Periodic) {
            int p = old.periodicPartner[f];
            Real limit = tol.absoluteGeometry+tol.relativeGeometry*std::max(absValue(sweep[f]),
                absValue(sweep[p]));
            if (absValue(sweep[f]+sweep[p])>limit) geometryWarning(tol.diagnostics,sweep[f]+sweep[p],limit,f);
        }
        }
        candidate.oldPoints = old.points;
        candidate.oldVolumes = old.volumes;
        candidate.geometryVersion = accepted.geometryVersion+1;
        candidate.meshVelocity.resize(old.owner.size());
        for (std::size_t f = 0; f < old.owner.size(); ++f) {
            candidate.meshVelocity[f] = (candidate.faceCentres[f]-old.faceCentres[f])/interval;
        }
        output = std::move(candidate);
        outputSweep = std::move(sweep);
        error.clear();
        return true;
    }
}
}
