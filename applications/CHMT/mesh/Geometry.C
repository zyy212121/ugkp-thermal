#include "mesh/Geometry.H"
#include "mesh/GeometryDiagnostics.H"
#include "mesh/Motion.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "mesh/EmptyExtrusion.H"
#include "mesh/SweepWorkspace.H"
#include <functional>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
namespace chmt {
    namespace {
        bool errorAt(std::string& e, const char* what, int index) {
            e = std::string(what)+" at index "+std::to_string(index);
            return false;
        }
        Real meshScale(const std::vector<Vec3>& p) {
            if (p.empty())return 0;
            Vec3 lo = p[0], hi = p[0];
            for (auto x:p) {
                lo.x = std::min(lo.x, x.x);
                lo.y = std::min(lo.y, x.y);
                lo.z = std::min(lo.z, x.z);
                hi.x = std::max(hi.x, x.x);
                hi.y = std::max(hi.y, x.y);
                hi.z = std::max(hi.z, x.z);
            }
            return mag(hi-lo);
        }
        Vec3 triangleArea(Vec3 a, Vec3 b, Vec3 c) {
            return cross(b-a, c-a)*.5;
        }
        Real triangleDistance(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
            Vec3 ab = b-a, ac = c-a, ap = p-a;
            Real d1 = dot(ab, ap), d2 = dot(ac, ap);
            if (d1 <= 0 && d2 <= 0)return mag(ap);
            Vec3 bp = p-b;
            Real d3 = dot(ab, bp), d4 = dot(ac, bp);
            if (d3 >= 0 && d4 <= d3)return mag(bp);
            Real vc = d1*d4-d3*d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0)return mag(p-(a+ab*(d1/(d1-d3))));
            Vec3 cp = p-c;
            Real d5 = dot(ab, cp), d6 = dot(ac, cp);
            if (d6 >= 0 && d5 <= d6)return mag(cp);
            Real vb = d5*d2-d1*d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0)return mag(p-(a+ac*(d2/(d2-d6))));
            Real va = d3*d6-d5*d4;
            if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0)return mag(p-(b+(c-b)*((d4-d3)/(d4-d3+d5-d6))));
            const Real den = va+vb+vc;
            if (!(den>0))return std::min(mag(ap), std::min(mag(bp), mag(cp)));
            return mag(p-(a+ab*(vb/den)+ac*(vc/den)));
        }
        bool periodicGeometry(const HostMesh& m, std::string& error) {
            const int nf = static_cast<int>(m.owner.size());
            const Real scale = meshScale(m.points), tol = 1e-10*scale;
            for (int f = 0; f<nf; ++f)if (m.boundaryKind[f] == BoundaryKind::Periodic) {
                int p = m.periodicPartner[f];
                if (p<0 || p >= nf || p == f || m.periodicPartner[p] != f
                    || m.boundaryKind[p] != BoundaryKind::Periodic || m.neighbour[f] >= 0
                    || m.neighbour[p] >= 0)return errorAt(error, "invalid periodic pair", f);
                const int ownerVertices = m.faceOffsets[f+1]-m.faceOffsets[f];
                const int partnerVertices = m.faceOffsets[p+1]-m.faceOffsets[p];
                if (ownerVertices != partnerVertices) {
                    return errorAt(error, "nonconformal periodic face", f);
                }
                if (mag(m.areaVectors[f]+m.areaVectors[p])>1e-10*maxValue(mag(m.areaVectors[f]),
                    mag(m.areaVectors[p])))return errorAt(error,
                    "rotational/nonmatching periodic areas unsupported", f);
                const Vec3 shift = m.faceCentres[p]-m.faceCentres[f];
                std::set<int> used;
                for (int i = m.faceOffsets[f]; i<m.faceOffsets[f+1]; ++i) {
                    int found = -1;
                    for (int j = m.faceOffsets[p]; j < m.faceOffsets[p+1]; ++j) {
                        const Vec3 separation = m.points[m.facePoints[i]]+shift-m.points[m.facePoints[j]];
                        if (mag(separation) > tol)continue;
                        found = j;
                        break;
                    }
                    if (found<0 || !used.insert(found).second)return errorAt(error,
                        "rotational/nontranslational periodic vertices unsupported", f);
                }
            }
            return true;
        }
        bool geometryInPlace(HostMesh& m, std::string& error,empty_detail::Extrusion* extrusion=nullptr,bool* numericalFailure=nullptr) {
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
                if ((vertices.size()<4 || mag(closure)>1e-10*areaSum))return errorAt(error,
                    "open/degenerate cell geometry", c);
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
                    for (int j = begin+1; j+1<end; ++j)distance = std::min(distance,
                        triangleDistance(m.cellCentres[c], m.points[m.facePoints[begin]],
                        m.points[m.facePoints[j]], m.points[m.facePoints[j+1]]));
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
        Real polynomialValue(const Real* coefficient, Real t) {
            return ((coefficient[3]*t+coefficient[2])*t+coefficient[1])*t+coefficient[0];
        }
        void polynomialBounds(const Real* coefficient, Real& minimum, Real& maximum) {
            minimum = minValue(polynomialValue(coefficient, 0), polynomialValue(coefficient, 1));
            maximum = maxValue(polynomialValue(coefficient, 0), polynomialValue(coefficient, 1));
            const Real a = 3*coefficient[3], b = 2*coefficient[2], c = coefficient[1];
            Real roots[2] = {-1, -1};
            const Real scale = maxValue(absValue(a), maxValue(absValue(b), absValue(c)));
            const Real cutoff = 32*std::numeric_limits<Real>::epsilon()*scale;
            if (absValue(a) <= cutoff) {
                if (absValue(b)>cutoff)roots[0] = -c/b;
            } else {
                const Real discriminant = b*b-4*a*c;
                if (discriminant >= 0) {
                    // Stable quadratic roots, including a root close to zero.
                    const Real root = ::sqrt(discriminant);
                    const Real q = -.5*(b+(b >= 0?root:-root));
                    if (q != 0) {
                        roots[0] = q/a;
                        roots[1] = c/q;
                    } else roots[0] = -b/(2*a);
                }
            }
            for (Real t:roots)if (t>0 && t<1) {
                const Real value = polynomialValue(coefficient, t);
                minimum = minValue(minimum, value);
                maximum = maxValue(maximum, value);
            }
        }
        bool certifyLinearTrajectory(const HostMesh& mesh, const std::vector<Vec3>& endpoint, std::string& error) {
            bool moving = false;
            for (std::size_t p = 0; p<endpoint.size(); ++p)moving = moving || mag(endpoint[p]-mesh.points[p])>0;
            if (!moving)return true;
            // Check the full polynomial trajectory, not just sampled mesh quality.
            // Every boundary triangle retains orientation and the fixed oriented
            // tetrahedral fan about each cell centre stays positive throughout.
            for (int cell = 0; cell<static_cast<int>(mesh.volumes.size()); ++cell) {
                std::set<int> vertices;
                for (int k = mesh.cellFaceOffsets[cell]; k<mesh.cellFaceOffsets[cell+1]; ++k) {
                    int f = mesh.cellFaces[k];
                    for (int j = mesh.faceOffsets[f]; j<mesh.faceOffsets[f+1]; ++j)vertices.insert(mesh.facePoints[j]);
                }
                // The vertex-average centre follows the same linear path.
                // A positive oriented tetrahedral fan certifies a star-shaped
                // cell without demanding that every polygon be exactly planar.
                Vec3 centre{},centreDelta{};
                for(int v:vertices){centre+=mesh.points[v];centreDelta+=endpoint[v]-mesh.points[v];}
                centre=centre/Real(vertices.size());centreDelta=centreDelta/Real(vertices.size());
                long double solidAngle=0;int fanTriangles=0;
                for (int k = mesh.cellFaceOffsets[cell]; k<mesh.cellFaceOffsets[cell+1]; ++k) {
                    int face = mesh.cellFaces[k], sign = mesh.cellFaceSigns[k];
                    int begin = mesh.faceOffsets[face], end = mesh.faceOffsets[face+1];
                    int ia = mesh.facePoints[begin];
                    for (int j = begin+1; j+1<end; ++j) {
                        int ib = mesh.facePoints[j], ic = mesh.facePoints[j+1];
                        Vec3 a = mesh.points[ia], b = mesh.points[ib], c = mesh.points[ic];
                        Vec3 da = endpoint[ia]-a, db = endpoint[ib]-b, dc = endpoint[ic]-c;
                        Vec3 area0 = cross(b-a, c-a);
                        Vec3 area1 = cross(db-da, c-a)+cross(b-a, dc-da);
                        Vec3 area2 = cross(db-da, dc-da);
                        Real orientation[4] = {dot(area0, area0), dot(area0, area1), dot(area0, area2), 0};
                        Real lower = 0, upper = 0;
                        polynomialBounds(orientation, lower, upper);
                        if (!finite(lower)
                            || lower <= 128*std::numeric_limits<Real>::epsilon()*orientation[0]) {
                            return errorAt(error,
                                "moving face triangle collapses or cannot retain certified orientation", face);
                        }
                        const Vec3 r=a-centre,u=b-centre,v=c-centre;
                        const Vec3 dr=da-centreDelta,du=db-centreDelta,dv=dc-centreDelta;
                        // Positive tetrahedra alone could cover the same volume
                        // twice (for example a self-intersecting star bipyramid).
                        // A closed, oriented fan must cover the unit sphere once.
                        // Positivity over the full path preserves this degree.
                        long double unit[3][3]={{r.x,r.y,r.z},{u.x,u.y,u.z},{v.x,v.y,v.z}};
                        for(auto& w:unit){
                            const long double norm=std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
                            if(!(norm>0))return errorAt(error,"zero-radius cell fan",cell);
                            for(auto& x:w)x/=norm;
                        }
                        const auto& a0=unit[0];const auto& b0=unit[1];const auto& c0=unit[2];
                        const long double determinant=a0[0]*(b0[1]*c0[2]-b0[2]*c0[1])
                            +a0[1]*(b0[2]*c0[0]-b0[0]*c0[2])+a0[2]*(b0[0]*c0[1]-b0[1]*c0[0]);
                        long double denominator=1;
                        for(int component=0;component<3;++component)
                            denominator+=a0[component]*b0[component]+b0[component]*c0[component]+c0[component]*a0[component];
                        solidAngle+=sign*2*std::atan2(determinant,denominator);++fanTriangles;

                        Real tetra[4]={
                            sign*dot(r,cross(u,v)),
                            sign*(dot(dr,cross(u,v))+dot(r,cross(du,v))+dot(r,cross(u,dv))),
                            sign*(dot(dr,cross(du,v))+dot(dr,cross(u,dv))+dot(r,cross(du,dv))),
                            sign*dot(dr,cross(du,dv))
                        };
                        polynomialBounds(tetra,lower,upper);
                        if(!finite(lower)||!(tetra[0]>0)
                            ||lower<=128*std::numeric_limits<Real>::epsilon()*tetra[0])
                            return errorAt(error,
                                "moving cell tetrahedral fan collapses or inverts",cell);

                    }
                }
                const long double angleTolerance=1024*std::numeric_limits<Real>::epsilon()*fanTriangles;
                if(!std::isfinite(solidAngle)||std::abs(solidAngle-4*std::acos(-1.L))>angleTolerance)
                    return errorAt(error,"cell fan is not a single star-shaped volume",cell);
            }
            return true;
        }
        bool stageIntegrals(const HostMesh& m, const std::vector<Vec3>& endpoint, std::vector<Vec3>& area,
            std::vector<Real>& sweep, std::string& error,const empty_detail::Extrusion* representation=nullptr,bool* numericalFailure=nullptr) {
            if(numericalFailure)*numericalFailure=true;

            if ((endpoint.size() != m.points.size()))return errorAt(error,
                "topology-changing point count unsupported", -1);
            for (int p = 0; p < static_cast<int>(endpoint.size()); ++p) {
                if ((!finite(endpoint[p])))return errorAt(error, "nonfinite trajectory endpoint", p);
            }
            area.assign(m.owner.size(), {});
            sweep.assign(m.owner.size(), 0);
            bool emptyChecked=representation!=nullptr;
            if(representation && !empty_detail::trajectory(m,*representation,endpoint,error))return false;
            for (int f = 0; f<static_cast<int>(m.owner.size()); ++f) {
                if(!emptyChecked && m.boundaryKind[f]==BoundaryKind::Empty) {
                    empty_detail::Extrusion extrusion;
                    if(!empty_detail::build(m,extrusion,error) || !empty_detail::trajectory(m,extrusion,endpoint,error))return false;
                    emptyChecked=true;
                }
                int begin = m.faceOffsets[f], end = m.faceOffsets[f+1], ia = m.facePoints[begin];
                for (int j = begin+1; j+1<end; ++j) {
                    int ib = m.facePoints[j], ic = m.facePoints[j+1];
                    Vec3 a = m.points[ia], b = m.points[ib], c = m.points[ic], da = endpoint[ia]-a,
                        db = endpoint[ib]-b, dc = endpoint[ic]-c;
                    // Exact moving-triangle/prism integral for linear vertex paths.
                    Vec3 average = (triangleArea(a, b, c)+triangleArea(a+da*.5, b+db*.5,
                        c+dc*.5)*4+triangleArea(a+da, b+db, c+dc))/6;
                    area[f] += average;
                    sweep[f] += dot((da+db+dc)/3, average);
                }
                if (!finite(area[f]) || !finite(sweep[f]))
                    return errorAt(error,"nonfinite stage area/swept volume",f);
            }
            return true;
        }
    }
    bool rebuildGeometryImpl(HostMesh& mesh, std::string& error,bool* numericalFailure) {
        HostMesh candidate = mesh;
        if (!geometryInPlace(candidate, error,nullptr,numericalFailure))return false;
        mesh = std::move(candidate);
        return true;
    }
    bool makeStageAreaVectorsImpl(const HostMesh& accepted, const std::vector<Vec3>& endpoint,
        std::vector<Vec3>& output, std::string& error,bool* numericalFailure) {
        HostMesh old = accepted;
        empty_detail::Extrusion extrusion;
        if (!geometryInPlace(old, error,&extrusion,numericalFailure))return false;
        std::vector<Vec3> a;
        std::vector<Real>s;
        if (!stageIntegrals(old, endpoint, a, s, error,extrusion.axis>=0?&extrusion:nullptr,numericalFailure))return false;
        output = std::move(a);
        error.clear();
        return true;
    }
    Real maximumGclResidual(const HostMesh& oldMesh, const HostMesh& newMesh, const std::vector<Real>& sweep) {
        Real r = 0;
        if (sweep.size() != oldMesh.owner.size()
            || oldMesh.volumes.size() != newMesh.volumes.size())return std::numeric_limits<Real>::infinity();
        for (int c = 0; c<static_cast<int>(oldMesh.volumes.size()); ++c) {
            Real sum = 0;
            for (int k = oldMesh.cellFaceOffsets[c]; k < oldMesh.cellFaceOffsets[c+1]; ++k) {
                sum += oldMesh.cellFaceSigns[k]*sweep[oldMesh.cellFaces[k]];
            }
            r = std::max(r, absValue(newMesh.volumes[c]-oldMesh.volumes[c]-sum));
        }
        return r;
    }
    bool makeStageGeometryImpl(const HostMesh& accepted, const std::vector<Vec3>& endpoint, double interval,
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
        if (!geometryInPlace(old, error,&extrusion,numericalFailure) || (!empty_detail::trajectory(old,extrusion,endpoint,error)))return false;
        HostMesh candidate = old;
        candidate.points = endpoint;
        if (!geometryInPlace(candidate, error,nullptr,numericalFailure))return false;
        if ((candidate.topologyHash != old.topologyHash))return errorAt(error, "topology change unsupported", -1);
        {
        if (!certifyLinearTrajectory(old, endpoint, error))return false;
        // Reject a crossed/inverted path even if endpoint orientation recovered.
        HostMesh mid = old;
        for (std::size_t p = 0; p<endpoint.size(); ++p)mid.points[p] = extrusion.axis<0?
            (old.points[p]+endpoint[p])*.5:old.points[p]+(endpoint[p]-old.points[p])*.5;
        if (!geometryInPlace(mid, error))return false;
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
        if (!geometryInPlace(oneThird, error) || !geometryInPlace(twoThird, error))return false;
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
            if (mag(closure)>1e-10*a)return errorAt(error,"independent stage area closure failed",c);
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
    namespace {
    bool harmonicPointMotionRepresentedImpl(const HostMesh& mesh, const std::vector<Vec3>& prescribedDisplacement,
        const std::vector<unsigned char>& prescribed, std::vector<Vec3>& output, std::string& error,
        const empty_detail::Extrusion* knownExtrusion,bool* numericalFailure=nullptr,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
        if(numericalFailure)*numericalFailure=true;

        const int np = static_cast<int>(mesh.points.size());
        if ((prescribed.size() != mesh.points.size()
            || prescribedDisplacement.size() != mesh.points.size()))return errorAt(error,
            "point-motion array size", -1);
        std::vector<std::set<int>> adjacency(np);
        bool hasEmpty=false;
        for (std::size_t f = 0; f<mesh.owner.size(); ++f) {
            hasEmpty=hasEmpty || (f<mesh.boundaryKind.size() && mesh.boundaryKind[f]==BoundaryKind::Empty);
            int begin = mesh.faceOffsets[f], end = mesh.faceOffsets[f+1];
            for (int k = begin; k<end; ++k) {
                int a = mesh.facePoints[k], b = mesh.facePoints[k+1 == end?begin:k+1];
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }
        if(hasEmpty) {
            empty_detail::Extrusion local;
            if(!knownExtrusion) {
                if(!empty_detail::build(mesh,local,error))return false;
                knownExtrusion=&local;
            }
            if(!empty_detail::prescribedMotion(*knownExtrusion,prescribedDisplacement,prescribed,error))return false;
        }
        std::vector<Vec3> displacement(np), next(np);
        bool any = false;
        for (int p = 0; p<np; ++p) {
            if (prescribed[p]) {
                if ((!finite(prescribedDisplacement[p])))return errorAt(error, "nonfinite prescribed motion", p);
                displacement[p] = prescribedDisplacement[p];
                any = true;
            }
            if ((adjacency[p].empty()))return errorAt(error, "isolated mesh point", p);
        }
        if ((np && !any))return errorAt(error, "harmonic motion lacks Dirichlet anchor", -1);
        const Real tolerance = 1e-12*meshScale(mesh.points);
        bool converged = np == 0;
        Real finalResidual=0;
        for (int iter = 0; iter<20000 && !converged; ++iter) {
            Real residual = 0;
            for (int p = 0; p<np; ++p) {
                if (prescribed[p]) {
                    next[p] = displacement[p];
                    continue;
                }
                Vec3 sum{};
                Real total = 0;
                for (int other:adjacency[p]) {
                    const Real length = mag(mesh.points[p]-mesh.points[other]);
                    if ((!(length>0)))return errorAt(error, "zero-length motion edge", p);
                    Real weight = 1/length;
                    sum += displacement[other]*weight;
                    total += weight;
                }
                next[p] = sum/total;
                if (!finite(next[p]))return errorAt(error,"nonfinite harmonic displacement",p);
                residual = std::max(residual, mag(next[p]-displacement[p]));
            }
            displacement.swap(next);
            finalResidual=residual; converged = residual <= tolerance;
        }
        if (!converged) geometryWarning(diagnostics,finalResidual,tolerance,-1,false);
        output = mesh.points;
        for (int p = 0; p<np; ++p)output[p] += displacement[p];
        error.clear();
        return true;
    }
    }
    bool harmonicPointMotion(const HostMesh& mesh, const std::vector<Vec3>& prescribedDisplacement,
        const std::vector<unsigned char>& prescribed, std::vector<Vec3>& output, std::string& error,
        ugkp::diagnostics::DeviceContext* diagnostics) {
        return harmonicPointMotionRepresentedImpl(mesh,prescribedDisplacement,prescribed,output,error,nullptr,nullptr,diagnostics);
    }
    namespace {
        using TrajectoryCheck = std::function<bool(const HostMesh&, std::string&)>;
        struct PlanarTriangleConstraint { int a,b,c,d; Real scale,tolerance; };
        Real triple(Vec3 a, Vec3 b, Vec3 c) { return dot(a,cross(b,c)); }
        Real tripleDerivative(Vec3 a, Vec3 b, Vec3 c, Vec3 da, Vec3 db, Vec3 dc) {
            return triple(da,b,c)+triple(a,db,c)+triple(a,b,dc);
        }
        // Exact derivatives of the same Simpson-integrated triangle polynomial
        // used by stageIntegrals. No finite-difference rank decision is involved.
        Real sweepDerivative(const HostMesh& old, const std::vector<Vec3>& end,
            const Vec3* mode, int f) {
            Real result=0;
            const int begin=old.faceOffsets[f], stop=old.faceOffsets[f+1], ia=old.facePoints[begin];
            for (int k=begin+1;k+1<stop;++k) {
                const int ib=old.facePoints[k],ic=old.facePoints[k+1];
                const Vec3 a=old.points[ia],b=old.points[ib],c=old.points[ic];
                const Vec3 da=end[ia]-a,db=end[ib]-b,dc=end[ic]-c;
                const Vec3 ea=mode[ia],eb=mode[ib],ec=mode[ic];
                const Vec3 u=b-a,v=c-a,du=db-da,dv=dc-da,eu=eb-ea,ev=ec-ea;
                const Vec3 area=cross(u,v)*.5+(cross(du,v)+cross(u,dv))*.25+cross(du,dv)/6;
                const Vec3 derivative=(cross(eu,v)+cross(u,ev))*.25
                    +(cross(eu,dv)+cross(du,ev))/6;
                result+=dot((ea+eb+ec)/3,area)+dot((da+db+dc)/3,derivative);
            }
            return result;
        }
        void planarPolynomial(const HostMesh& old, const std::vector<Vec3>& end,
            const PlanarTriangleConstraint& p, Real out[3], const Vec3* mode=nullptr) {
            const Vec3 u=old.points[p.b]-old.points[p.a],v=old.points[p.c]-old.points[p.a],
                w=old.points[p.d]-old.points[p.a];
            const Vec3 da=end[p.a]-old.points[p.a],du=end[p.b]-old.points[p.b]-da,
                dv=end[p.c]-old.points[p.c]-da,dw=end[p.d]-old.points[p.d]-da;
            if (!mode) {
                out[0]=triple(du,v,w)+triple(u,dv,w)+triple(u,v,dw);
                out[1]=triple(du,dv,w)+triple(du,v,dw)+triple(u,dv,dw);
                out[2]=triple(du,dv,dw);
            } else {
                const Vec3 eu=mode[p.b]-mode[p.a],ev=mode[p.c]-mode[p.a],
                    ew=mode[p.d]-mode[p.a];
                out[0]=triple(eu,v,w)+triple(u,ev,w)+triple(u,v,ew);
                out[1]=tripleDerivative(du,dv,w,eu,ev,{})
                    +tripleDerivative(du,v,dw,eu,{},ew)+tripleDerivative(u,dv,dw,{},ev,ew);
                out[2]=tripleDerivative(du,dv,dw,eu,ev,ew);
            }
        }
        // Row-pivoted, twice-reorthogonalized QR of J^T avoids normal equations.
        // The resulting correction is minimum Euclidean norm in boundary DOFs.
        // Dependent equations are checked for compatibility, never least-squared.
        bool minimumNormCorrection(sweep_detail::Rows rows, sweep_detail::Reals rhs,
            Real rankTolerance, Real consistencyTolerance, sweep_detail::Reals& delta, int& rank,ugkp::diagnostics::DeviceContext* diagnostics) {
            const int nr=static_cast<int>(rows.size()),nc=static_cast<int>(delta.size());
            sweep_detail::Reals norm(nr,0,delta.get_allocator());
            Real largest=0;
            for (int i=0;i<nr;++i) {
                for (Real x:rows[i]) { if (!finite(x))return false; norm[i]+=x*x; }
                if (!finite(norm[i]))return false;
                largest=maxValue(largest,::sqrt(norm[i]));
            }
            rank=0; std::fill(delta.begin(),delta.end(),0);
            while (rank<nr && rank<nc) {
                int pivot=rank;
                for (int i=rank+1;i<nr;++i) if (norm[i]>norm[pivot]) pivot=i;
                const Real length=::sqrt(norm[pivot]);
                if (!(length>rankTolerance*largest)) break;
                rows[rank].swap(rows[pivot]); std::swap(rhs[rank],rhs[pivot]);
                std::swap(norm[rank],norm[pivot]);
                for (Real& x:rows[rank]) x/=length;
                rhs[rank]/=length;
                if (!finite(rhs[rank])) return false;
                for (int j=0;j<nc;++j) delta[j]+=rows[rank][j]*rhs[rank];
                for (int i=rank+1;i<nr;++i) {
                    for (int pass=0;pass<2;++pass) {
                        Real projection=0;
                        for (int j=0;j<nc;++j) projection+=rows[i][j]*rows[rank][j];
                        for (int j=0;j<nc;++j) rows[i][j]-=projection*rows[rank][j];
                        rhs[i]-=projection*rhs[rank];
                    }
                    norm[i]=0; for (Real x:rows[i]) norm[i]+=x*x;
                }
                ++rank;
            }
            for (int i=rank;i<nr;++i) {
                if (!finite(rhs[i])) return false;
                if (absValue(rhs[i])>consistencyTolerance)
                    geometryWarning(diagnostics,rhs[i],consistencyTolerance,i,false);
            }
            for (Real x:delta) if (!finite(x)) return false;
            return true;
        }
        bool constrainedMotionImpl(const HostMesh& input, const std::vector<int>& faces,
            const std::vector<Real>& targets, const std::vector<Vec3>& prescribedDisplacement,
            const std::vector<unsigned char>& prescribed, const std::vector<Vec3>& directions,
            Real interval, HostMesh& output, std::vector<Real>& actualSweeps,
            SweepConstraintReport& report, std::string& error, const SweepConstraintControls& controls,
            const TrajectoryCheck& extraCheck, const Tolerances& tolerances,
            bool* numericalFailure) {
            report=SweepConstraintReport{};
            if(numericalFailure)*numericalFailure=true;

            auto reject=[&](SweepConstraintStatus status,const std::string& message) {
                report.status=status; error=message; return false;
            };
            const std::size_t np=input.points.size();
            if ((faces.size()!=targets.size() || prescribedDisplacement.size()!=np
                || prescribed.size()!=np || directions.size()!=np || !(interval>0) || !finite(interval)
                || !finite(controls.absoluteVolumeTolerance) || controls.absoluteVolumeTolerance<0
                || !finite(controls.relativeVolumeTolerance) || controls.relativeVolumeTolerance<0
                || (!controls.faceVolumeTolerance.empty()&&controls.faceVolumeTolerance.size()!=faces.size())
                || !finite(controls.rankTolerance) || controls.rankTolerance<=0 || controls.rankTolerance>=1
                || controls.maximumIterations<1
                || controls.maximumDegreesOfFreedom<0 || controls.maximumConstraints<0
                || !controls.maximumWorkspaceBytes))
                return reject(SweepConstraintStatus::InvalidInput,"invalid swept-volume constraint inputs/controls");
            if((np>static_cast<std::size_t>(std::numeric_limits<int>::max())
                || input.owner.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())
                || input.facePoints.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())))
                return reject(SweepConstraintStatus::SizeLimit,"swept-volume mesh exceeds index representation");
            for (Real tolerance:controls.faceVolumeTolerance)if((!finite(tolerance)||tolerance<=0))
                return reject(SweepConstraintStatus::InvalidInput,"invalid per-face swept-volume error ceiling");
            HostMesh worldOld=input;
            if (!rebuildGeometryImpl(worldOld,error,numericalFailure)) return false;
            // Solve displacements in a local frame. Repeated endpoint updates
            // must not lose significant bits merely because the mesh is remote
            // from the coordinate origin. Final certification still uses the
            // actual representable world-coordinate vertices and their sweeps.
            const Vec3 origin=worldOld.points.empty()?Vec3{}:worldOld.points.front();
            HostMesh old=worldOld;
            Real coordinateMagnitude=0;
            for (auto& point:old.points) {
                coordinateMagnitude=maxValue(coordinateMagnitude,maxValue(absValue(point.x),
                    maxValue(absValue(point.y),absValue(point.z))));
                point-=origin;
            }
            if (!rebuildGeometryImpl(old,error,numericalFailure)) return false;
            const Real coordinateUncertainty=std::numeric_limits<Real>::epsilon()*coordinateMagnitude;
            std::set<int> used;
            std::vector<Real> sweepScale;
            const Real length=meshScale(old.points);
            for (std::size_t i=0;i<faces.size();++i) {
                const int f=faces[i];
                if ((f<0 || f>=static_cast<int>(old.owner.size()) || old.neighbour[f]>=0
                    || !used.insert(f).second || !finite(targets[i])))
                    return reject(SweepConstraintStatus::InvalidInput,"invalid/duplicate boundary sweep constraint");
                sweepScale.push_back(mag(old.areaVectors[f])*length);
            }
            const auto sweepTolerance=[&](std::size_t i,Real actual) {
                const Real requested=controls.absoluteVolumeTolerance
                    +controls.relativeVolumeTolerance*maxValue(absValue(targets[i]),absValue(actual));
                const Real geometricFloor=32*std::numeric_limits<Real>::epsilon()*sweepScale[i];
                const Real tolerance=maxValue(requested,geometricFloor);
                return controls.faceVolumeTolerance.empty()?tolerance:minValue(tolerance,controls.faceVolumeTolerance[i]);
            };
            std::vector<unsigned char> boundaryPoint(np,0);
            bool hasEmpty=false;
            for (std::size_t f=0;f<old.owner.size();++f) {
                hasEmpty=hasEmpty || old.boundaryKind[f]==BoundaryKind::Empty;
                if (old.neighbour[f]<0)
                    for (int k=old.faceOffsets[f];k<old.faceOffsets[f+1];++k) boundaryPoint[old.facePoints[k]]=1;
            }
            empty_detail::Extrusion extrusion;
            bool paired=true;
            if(hasEmpty) {
                paired=empty_detail::build(old,extrusion,error);

                if(!paired)return false;
                if(paired && !empty_detail::motionData(extrusion,prescribedDisplacement,prescribed,directions,error))return false;
            }
            std::vector<int> movable;
            for (std::size_t p=0;p<np;++p) {
                if ((!finite(directions[p]) || !finite(prescribedDisplacement[p])))
                    return reject(SweepConstraintStatus::InvalidInput,"nonfinite constraint point data");
                if (mag(directions[p])>0) {
                    if ((!prescribed[p] || !boundaryPoint[p])) return reject(SweepConstraintStatus::InvalidInput,
                        "swept-volume correction DOF must be a prescribed boundary vertex");
                    movable.push_back(static_cast<int>(p));
                }
            }
            std::vector<int> metricMovable;
            if(hasEmpty && paired) {
                metricMovable=movable;movable.clear();
                for(int p:metricMovable)if(p<extrusion.mate[p])movable.push_back(p);
            }
            report.degreesOfFreedom=static_cast<int>(movable.size());
            std::vector<PlanarTriangleConstraint> planar;
            for (int f=0;f<static_cast<int>(old.owner.size());++f) {
                const int begin=old.faceOffsets[f],stop=old.faceOffsets[f+1];
                const int a=old.facePoints[begin],b=old.facePoints[begin+1],c=old.facePoints[begin+2];
                Real faceLength=0;
                for (int k=begin+1;k<stop;++k)
                    faceLength=maxValue(faceLength,mag(old.points[old.facePoints[k]]-old.points[a]));
                const Real scale=mag(cross(old.points[b]-old.points[a],old.points[c]-old.points[a]))*faceLength;
                for (int k=begin+3;k<stop;++k) {
                    const int d=old.facePoints[k];
                    const Real volume=triple(old.points[b]-old.points[a],old.points[c]-old.points[a],old.points[d]-old.points[a]);
                    // Account for input-coordinate quantization as well as
                    // local triple-product roundoff. Translating a planar mesh
                    // must not drop equations merely because each stored vertex
                    // acquired a different last representable bit.
                    const Real uncertainty=64*std::numeric_limits<Real>::epsilon()*scale
                        +8*coordinateUncertainty*faceLength*faceLength;
                    if (scale>0 && absValue(volume)<=uncertainty)
                        planar.push_back({a,b,c,d,scale,
                            32*std::numeric_limits<Real>::epsilon()
                                +8*coordinateUncertainty*faceLength*faceLength/scale});
                }
            }
            const std::size_t constraintCount=sweep_detail::sum(faces.size(),sweep_detail::product(3,planar.size()));
            if((constraintCount>static_cast<std::size_t>(std::numeric_limits<int>::max())))
                return reject(SweepConstraintStatus::SizeLimit,"swept-volume constraints exceed index representation");
            report.constraints=static_cast<int>(constraintCount);
            std::vector<Vec3> start;
            if (!harmonicPointMotionRepresentedImpl(old,prescribedDisplacement,prescribed,start,error,
                hasEmpty&&paired?&extrusion:nullptr,numericalFailure,tolerances.diagnostics)) return false;
            auto evaluate=[&](const std::vector<Vec3>& points,std::vector<Real>& residual,
                std::vector<Real>& sweeps, Real& metric) {
                std::vector<Vec3> area;
                if (!stageIntegrals(old,points,area,sweeps,error,hasEmpty&&paired?&extrusion:nullptr,numericalFailure)) return false;
                residual.clear(); metric=0;
                report.maximumSweepResidual=0; report.maximumPlanarityResidual=0;
                for (std::size_t i=0;i<faces.size();++i) {
                    const Real r=sweeps[faces[i]]-targets[i];
                    if (!finite(r))return errorAt(error,"nonfinite swept-volume residual",faces[i]);
                    residual.push_back(r/sweepScale[i]);
                    metric=maxValue(metric,absValue(r)/sweepTolerance(i,sweeps[faces[i]]));
                    report.maximumSweepResidual=maxValue(report.maximumSweepResidual,absValue(r));
                }
                for (const auto& p:planar) {
                    Real coefficients[3]; planarPolynomial(old,points,p,coefficients);
                    for (Real coefficient:coefficients) {
                        const Real r=coefficient/p.scale;
                        if (!finite(r))return errorAt(error,"nonfinite planarity residual",p.a);
                        residual.push_back(r); metric=maxValue(metric,absValue(r)/p.tolerance);
                        report.maximumPlanarityResidual=maxValue(report.maximumPlanarityResidual,absValue(r));
                    }
                }
                return finite(metric);
            };
            auto certify=[&](const std::vector<Vec3>& localPoints,HostMesh& certified,
                std::vector<Real>& measured,std::string& failure,bool enforceTargets) {
                std::vector<Vec3> worldPoints=worldOld.points;
                for (std::size_t p=0;p<np;++p) {
                    const Vec3 displacement=localPoints[p]-old.points[p];
                    if (displacement.x!=0 || displacement.y!=0 || displacement.z!=0)
                        worldPoints[p]+=displacement;
                }
                if (!makeStageGeometry(worldOld,worldPoints,interval,certified,measured,failure,tolerances)
                    || (extraCheck && !extraCheck(certified,failure))) return false;
                if (enforceTargets) {
                    report.maximumSweepResidual=0;
                    for (std::size_t i=0;i<faces.size();++i) {
                        const Real actual=measured[faces[i]],residual=absValue(actual-targets[i]);
                        report.maximumSweepResidual=maxValue(report.maximumSweepResidual,residual);
                        if (residual>sweepTolerance(i,actual)) geometryWarning(tolerances.diagnostics,
                            residual,sweepTolerance(i,actual),faces[i]);
                    }
                }
                return true;
            };
            std::vector<Vec3> points=start;
            std::vector<Real> residual,sweeps;
            Real metric=0;
            if (!evaluate(points,residual,sweeps,metric)) {

                return reject(SweepConstraintStatus::InvalidInput,"nonfinite swept-volume constraint residual");
            }

            if (metric<=1) {
                HostMesh certified; std::vector<Real> measured;
                if (!certify(points,certified,measured,error,true)) {

                    report.status=SweepConstraintStatus::InvalidTrajectory; return false;
                }
                output=std::move(certified); actualSweeps=std::move(measured);
                report.status=SweepConstraintStatus::Success; error.clear(); return true;
            }
            if ((controls.maximumDegreesOfFreedom && report.degreesOfFreedom>controls.maximumDegreesOfFreedom)
                || (controls.maximumModeEntries && np && movable.size()>controls.maximumModeEntries/np)) {

                return reject(SweepConstraintStatus::SizeLimit,"swept-volume correction exceeds explicit DOF/mode limit");
            }
            if (controls.maximumConstraints && report.constraints>controls.maximumConstraints) {

                return reject(SweepConstraintStatus::SizeLimit,"swept-volume correction exceeds explicit constraint limit");
            }
            try {
            // Lumped reference-surface area is the distortion metric. Shared
            // vertices must carry all incident faces; otherwise minimum Euclidean
            // motion bends a uniformly receding planar multi-face surface.
            using namespace sweep_detail;
            sweep_detail::Budget workspace(controls.maximumWorkspaceBytes,report.peakWorkspaceBytes);
            Reals pointWeight(np,0,Allocator<Real>(workspace));
            HarmonicModes harmonic(old,prescribed,workspace,1e-12*length);
            Points boundary(np,Vec3{},Allocator<Vec3>(workspace));
            Points correction(np,Vec3{},Allocator<Vec3>(workspace));
            for (int f:faces) {
                const Real share=mag(old.areaVectors[f])/(old.faceOffsets[f+1]-old.faceOffsets[f]);
                for (int k=old.faceOffsets[f];k<old.faceOffsets[f+1];++k) pointWeight[old.facePoints[k]]+=share;
            }
            Real meanWeight=0; int weighted=0;
            for (int p:hasEmpty?metricMovable:movable) if (pointWeight[p]>0) { meanWeight+=pointWeight[p]; ++weighted; }
            meanWeight=weighted?meanWeight/weighted:1;
            const std::size_t columns=movable.size(),rows=residual.size();
            const std::size_t rowBytes=sum(product(rows,sizeof(Reals)),product(product(rows,columns),sizeof(Real)));
            const std::size_t qrBytes=sum(rowBytes,product(sum(product(2,rows),columns),sizeof(Real)));
            const std::size_t modeBytes=product(np,sizeof(Vec3));
            const std::size_t oneCachedMode=sum(sizeof(Points),modeBytes);
            const bool cache=workspace.fits(qrBytes)
                && columns<=(workspace.limit-workspace.live-qrBytes)/oneCachedMode;
            if(!cache)workspace.check(sum(qrBytes,modeBytes));
            report.streamedModes=!cache;
            Vector<Points> modes{Allocator<Points>(workspace)};
            Points streamedMode{Allocator<Vec3>(workspace)};
            if(cache)modes.reserve(columns);else streamedMode.resize(np);
            auto generateMode=[&](int p,Points& end) {
                std::fill(boundary.begin(),boundary.end(),Vec3{});
                const Real weight=pointWeight[p]>0?pointWeight[p]:meanWeight;
                boundary[p]=normalized(directions[p])*(length*::sqrt(meanWeight/weight));
                if(hasEmpty) {
                    const int q=extrusion.mate[p];const Real other=pointWeight[q]>0?pointWeight[q]:meanWeight;
                    const Real scale=empty_detail::pairedScale(length*::sqrt(meanWeight/weight),length*::sqrt(meanWeight/other));
                    boundary[p]=normalized(directions[p])*scale;boundary[q]=boundary[p];
                }
                if((!finite(boundary[p])))return errorAt(error,"nonfinite correction boundary mode",p);
                ++report.modeSolves;
                if(!harmonic.solve(boundary,end,tolerances.diagnostics))return errorAt(error,"harmonic point motion did not converge",-1);
                return true;
            };
            if(cache)for(int p:movable) {
                Points end{Allocator<Vec3>(workspace)};
                if(!generateMode(p,end))return false;
                modes.push_back(std::move(end));
            }
            for (int iteration=0;iteration<=controls.maximumIterations;++iteration) {
                report.iterations=iteration;
                if (metric<=1) {
                    HostMesh certified; std::vector<Real> measured;
                    if (!certify(points,certified,measured,error,true)) {

                        report.status=SweepConstraintStatus::InvalidTrajectory; return false;
                    }
                    output=std::move(certified); actualSweeps=std::move(measured);
                    report.status=SweepConstraintStatus::Success; error.clear(); return true;
                }
                if (iteration==controls.maximumIterations) break;
                Rows jacobian{Allocator<Reals>(workspace)};
                jacobian.reserve(rows);
                for(std::size_t i=0;i<rows;++i)jacobian.emplace_back(columns,0,Allocator<Real>(workspace));
                for (std::size_t j=0;j<columns;++j) {
                    if(!cache && !generateMode(movable[j],streamedMode))return false;
                    const auto& mode=cache?modes[j]:streamedMode;
                    std::size_t row=0;
                    for (std::size_t i=0;i<faces.size();++i)
                        jacobian[row++][j]=sweepDerivative(old,points,mode.data(),faces[i])/sweepScale[i];
                    for (const auto& p:planar) {
                        Real derivative[3]; planarPolynomial(old,points,p,derivative,mode.data());
                        for (Real value:derivative) {
                            // Keep tangent equations even when the current plane
                            // is within roundoff: the next correction must not
                            // create an unconstrained internal volume transfer.
                            jacobian[row][j]=value/p.scale;
                            ++row;
                        }
                    }
                }
                Reals rhs(residual.begin(),residual.end(),Allocator<Real>(workspace));
                std::size_t planarRow=faces.size();
                for(const auto& p:planar)for(int coefficient=0;coefficient<3;++coefficient){
                    if(absValue(rhs[planarRow])<=p.tolerance)rhs[planarRow]=0;
                    ++planarRow;
                }
                Reals delta(columns,0,Allocator<Real>(workspace));
                Real maxResidual=0;
                for (Real& r:rhs) { maxResidual=maxValue(maxResidual,absValue(r)); r=-r; }
                if (!minimumNormCorrection(std::move(jacobian),std::move(rhs),controls.rankTolerance,
                    maxValue(128*std::numeric_limits<Real>::epsilon(),controls.rankTolerance*maxResidual),delta,report.rank,tolerances.diagnostics)) {
                    std::ostringstream detail; detail << "incompatible swept-volume/planarity constraints: numerical rank "
                        << report.rank << " of " << report.constraints << ", boundary DOFs " << report.degreesOfFreedom;

                    return reject(SweepConstraintStatus::Incompatible,detail.str());
                }
                std::fill(correction.begin(),correction.end(),Vec3{});
                for (std::size_t j=0;j<columns;++j) {
                    if(!cache && !generateMode(movable[j],streamedMode))return false;
                    const auto& mode=cache?modes[j]:streamedMode;
                    for (std::size_t p=0;p<np;++p) correction[p]+=mode[p]*delta[j];
                }
                // One full Newton correction. No candidate-alpha acceptance or replay.
                for (std::size_t p=0;p<np;++p) points[p]+=correction[p];
                if (!evaluate(points,residual,sweeps,metric))
                    return reject(SweepConstraintStatus::InvalidInput,"nonfinite swept-volume Newton residual");
                HostMesh certified; std::vector<Real> measured;
                if (!certify(points,certified,measured,error,false)) {
                    report.status=SweepConstraintStatus::InvalidTrajectory; return false;
                }

            }

            HostMesh certified; std::vector<Real> measured;
            if (!certify(points,certified,measured,error,true)) {
                report.status=SweepConstraintStatus::InvalidTrajectory; return false;
            }
            geometryWarning(tolerances.diagnostics,metric,1,-1,false);
            output=std::move(certified); actualSweeps=std::move(measured);
            report.status=SweepConstraintStatus::Success; error.clear(); return true;
            } catch(const sweep_detail::WorkspaceLimit&) {

                throw;
            }
        }
        bool constrainedMotion(const HostMesh& input,const std::vector<int>& faces,
            const std::vector<Real>& targets,const std::vector<Vec3>& displacement,
            const std::vector<unsigned char>& fixed,const std::vector<Vec3>& directions,
            Real interval,HostMesh& output,std::vector<Real>& sweeps,SweepConstraintReport& report,
            std::string& error,const SweepConstraintControls& controls,
            const TrajectoryCheck& extraCheck,const Tolerances& tolerances,
            bool* numericalFailure=nullptr) {
            try {
                return constrainedMotionImpl(input,faces,targets,displacement,fixed,directions,interval,
                    output,sweeps,report,error,controls,extraCheck,tolerances,numericalFailure);
            } catch(const sweep_detail::WorkspaceLimit&) {
                report.status=SweepConstraintStatus::SizeLimit;
                error="swept-volume correction workspace budget exceeded";
            } catch(const std::bad_alloc&) {
                if(numericalFailure)*numericalFailure=false;
                report.status=SweepConstraintStatus::SizeLimit;
                error="swept-volume geometry allocation failed";
            } catch(const std::length_error&) {
                if(numericalFailure)*numericalFailure=false;
                report.status=SweepConstraintStatus::SizeLimit;
                error="swept-volume geometry allocation size overflow";
            }
            return false;
        }
    }
    bool constrainFaceSweeps(const HostMesh& old, const std::vector<int>& faces,
        const std::vector<Real>& targets, const std::vector<Vec3>& displacement,
        const std::vector<unsigned char>& fixed, const std::vector<Vec3>& directions,
        Real interval, HostMesh& output, std::vector<Real>& sweeps, SweepConstraintReport& report,
        std::string& error, const SweepConstraintControls& controls, const Tolerances& tolerances,bool* numericalFailure) {
        return constrainedMotion(old,faces,targets,displacement,fixed,directions,interval,
            output,sweeps,report,error,controls,TrajectoryCheck{},tolerances,numericalFailure);
    }

    namespace {
        void fixedPhysicalBoundaries(const HostMesh& mesh, std::vector<unsigned char>& fixed) {
            fixed.assign(mesh.points.size(), 0);
            for (std::size_t f = 0; f < mesh.owner.size(); ++f) {
                if (mesh.neighbour[f] >= 0)continue;
                for (int k = mesh.faceOffsets[f]; k < mesh.faceOffsets[f+1]; ++k) {
                    fixed[mesh.facePoints[k]] = 1;
                }
            }
        }
        struct SurfaceEdge {
            int a = -1, b = -1, owner = -1, neighbour = -1, neighbourA = -1, neighbourB = -1;
        };
        bool surfaceEdges(const HostMesh& solid, const SurfaceMesh& surface, std::vector<SurfaceEdge>& edges,
            std::string& error,bool* numericalFailure=nullptr) {
            std::map<std::pair<int, int>, int> map;
            for (int i = 0; i<static_cast<int>(surface.solidFace.size()); ++i) {
                int f = surface.solidFace[i], begin = solid.faceOffsets[f], end = solid.faceOffsets[f+1];
                for (int k = begin; k<end; ++k) {
                    int a = solid.facePoints[k], b = solid.facePoints[k+1 == end?begin:k+1];
                    auto key = std::minmax(a, b);
                    auto hit = map.find(key);
                    if (hit == map.end()) {
                        map[key] = static_cast<int>(edges.size());
                        edges.push_back({a, b, i, -1});
                    } else {
                        auto& edge = edges[hit->second];
                        if ((edge.neighbour >= 0 || edge.a != b || edge.b != a))return errorAt(error,
                            "nonmanifold/inconsistently oriented surface edge", i);
                        edge.neighbour = i;
                        edge.neighbourA = a;
                        edge.neighbourB = b;
                    }
                }
            }
            // Join surface seam edges through the mesh's verified translational cyclic
            // patches. No nearest-neighbour map is substituted for a conforming pair.
            std::vector<unsigned char> removed(edges.size(), 0);
            const Real tol = 1e-10*meshScale(solid.points);
            for (std::size_t i = 0; i<edges.size(); ++i) {
                auto& edge = edges[i];
                if (removed[i] || edge.neighbour >= 0)continue;
                int patch = -1;
                for (int f = 0; f<static_cast<int>(solid.owner.size()); ++f) {
                    if (solid.boundaryKind[f] != BoundaryKind::Periodic)continue;
                    bool hasA = false, hasB = false;
                    for (int k = solid.faceOffsets[f]; k<solid.faceOffsets[f+1]; ++k) {
                        hasA = hasA || solid.facePoints[k] == edge.a;
                        hasB = hasB || solid.facePoints[k] == edge.b;
                    }
                    if (hasA && hasB) {
                        patch = f;
                        break;
                    }
                }
                if (patch<0)continue;
                Vec3 translation = solid.faceCentres[solid.periodicPartner[patch]]-solid.faceCentres[patch];
                int match = -1;
                for (std::size_t j = 0; j<edges.size(); ++j)if (i != j && !removed[j] && edges[j].neighbour<0
                    && mag(solid.points[edge.a]+translation-solid.points[edges[j].b]) <= tol
                    && mag(solid.points[edge.b]+translation-solid.points[edges[j].a]) <= tol) {
                    if (match >= 0)return errorAt(error, "ambiguous periodic surface edge", static_cast<int>(i));
                    match = static_cast<int>(j);
                }
                if (match<0){if(numericalFailure)*numericalFailure=false;
                    return errorAt(error, "unpaired periodic surface edge", static_cast<int>(i));}
                edge.neighbour = edges[match].owner;
                edge.neighbourA = edges[match].a;
                edge.neighbourB = edges[match].b;
                removed[match] = 1;
            }
            std::vector<SurfaceEdge> kept;
            for (std::size_t i = 0; i<edges.size(); ++i)if (!removed[i])kept.push_back(edges[i]);
            edges = std::move(kept);
            return true;
        }
        Vec3 trajectoryPoint(const HostMesh& mesh,const std::vector<Vec3>& endpoint,int p,Real t,bool empty) {
            return empty?mesh.points[p]+(endpoint[p]-mesh.points[p])*t:mesh.points[p]*(1-t)+endpoint[p]*t;
        }
        Vec3 faceAreaAt(const HostMesh& mesh, int f, const std::vector<Vec3>& endpoint, Real t,bool empty) {
            Vec3 result{};
            int begin = mesh.faceOffsets[f], end = mesh.faceOffsets[f+1];
            auto point = [&](int k) {
                int p = mesh.facePoints[k];
                return trajectoryPoint(mesh,endpoint,p,t,empty);
            };
            Vec3 a = point(begin);
            for (int k = begin+1; k+1<end; ++k)result += triangleArea(a, point(k), point(k+1));
            return result;
        }
        Real edgeSweepIntegrand(const HostMesh& mesh, const std::vector<Vec3>& endpoint,
            const SurfaceMesh& surface, const SurfaceEdge& edge, Real t,bool empty) {
            Vec3 a = trajectoryPoint(mesh,endpoint,edge.a,t,empty),
                b = trajectoryPoint(mesh,endpoint,edge.b,t,empty);
            Vec3 normal = normalized(faceAreaAt(mesh, surface.solidFace[edge.owner], endpoint, t,empty));
            if ((mag(normal) == 0))return undefinedValue();
            if (edge.neighbour >= 0) {
                Vec3 other = normalized(faceAreaAt(mesh, surface.solidFace[edge.neighbour], endpoint, t,empty));
                if ((mag(other) == 0))return undefinedValue();
                normal = normalized(normal+other);
            }
            if ((mag(normal) <= 1e-12 || mag(b-a) <= 0))return undefinedValue();
            Vec3 displacement = (endpoint[edge.a]-mesh.points[edge.a]+endpoint[edge.b]-mesh.points[edge.b])*.5;
            return dot(displacement, cross(b-a, normal));
        }
        Real edgeSweepQuadrature(const HostMesh& mesh, const std::vector<Vec3>& endpoint,
            const SurfaceMesh& surface, const SurfaceEdge& edge, int segments,bool empty) {
            // Composite Gauss-Legendre integration of instantaneous normal/conormal.
            // A single conservative co-normal is used at a curved internal edge;
            // physical normal area change is NOT fabricated as an edge sweep.
            constexpr Real x = .774596669241483377;
            Real sum = 0;
            for (int segment = 0; segment<segments; ++segment) {
                Real a = Real(segment)/segments, b = Real(segment+1)/segments, c = .5*(a+b), h = .5*(b-a);
                sum += h*((5.0/9)*edgeSweepIntegrand(mesh, endpoint, surface, edge,
                    c-h*x,empty)+(8.0/9)*edgeSweepIntegrand(mesh, endpoint, surface, edge,
                    c,empty)+(5.0/9)*edgeSweepIntegrand(mesh, endpoint, surface, edge, c+h*x,empty));
            }
            return sum;
        }
    }
    bool integrateSurfaceEdgeSweep(const HostMesh& mesh, const std::vector<Vec3>& endpoint, int faceOwner,
        int faceNeighbour, int pointA, int pointB, int segments, Real& output, std::string& error) {
        if (endpoint.size() != mesh.points.size() || faceOwner<0
            || faceOwner >= static_cast<int>(mesh.owner.size()) || faceNeighbour<-1
            || faceNeighbour >= static_cast<int>(mesh.owner.size()) || pointA<0 || pointB<0
            || pointA >= static_cast<int>(mesh.points.size())
            || pointB >= static_cast<int>(mesh.points.size()) || pointA == pointB
            || segments<1)return errorAt(error, "invalid surface edge quadrature input", faceOwner);
        SurfaceMesh s;
        s.solidFace = {faceOwner};
        if (faceNeighbour >= 0)s.solidFace.push_back(faceNeighbour);
        SurfaceEdge edge{pointA, pointB, 0, faceNeighbour >= 0?1:-1, -1, -1};
        empty_detail::Extrusion extrusion;
        for(auto kind:mesh.boundaryKind)if(kind==BoundaryKind::Empty) {
            if(!empty_detail::build(mesh,extrusion,error) || !empty_detail::trajectory(mesh,extrusion,endpoint,error))return false;
            break;
        }
        const Real value = edgeSweepQuadrature(mesh, endpoint, s, edge, segments,extrusion.axis>=0);
        if (!finite(value))return errorAt(error, "degenerate moving surface hinge/edge", faceOwner);
        output = value;
        error.clear();
        return true;
    }
    bool surfaceAreaMetricChange(const SurfaceMesh& s, std::vector<Real>& output, std::string& error) {
        if (s.oldArea.size() != s.area.size() || s.edgeOwner.size() != s.edgeNeighbour.size()
            || s.edgeOwner.size() != s.sweptEdgeArea.size())return errorAt(error,
            "incomplete surface metric arrays", -1);
        std::vector<Real> result(s.area.size());
        for (std::size_t f = 0; f<s.area.size(); ++f) {
            if (!finite(s.area[f]) || !finite(s.oldArea[f]) || s.area[f] <= 0
                || s.oldArea[f] <= 0)return errorAt(error, "invalid surface area", static_cast<int>(f));
            result[f] = s.area[f]-s.oldArea[f];
        }
        for (std::size_t e = 0; e<s.edgeOwner.size(); ++e) {
            int o = s.edgeOwner[e], n = s.edgeNeighbour[e];
            if (o<0 || o >= static_cast<int>(result.size()) || n<-1 || n >= static_cast<int>(result.size())
                || !finite(s.sweptEdgeArea[e]))return errorAt(error, "invalid surface metric edge",
                static_cast<int>(e));
            result[o] -= s.sweptEdgeArea[e];
            if (n >= 0)result[n] += s.sweptEdgeArea[e];
        }
        output = std::move(result);
        error.clear();
        return true;
    }
    bool mapConformalInterfaceImpl(const HostMesh& gas, const HostMesh& solid,
        const std::vector<int>& solidFaces, const std::vector<Real>& thickness,
        const std::vector<int>& gasCandidates, std::vector<int>& mappedGasFaces,
        std::vector<int>& solidToGas, std::string& error,bool* numericalFailure=nullptr) {
            if(numericalFailure)*numericalFailure=true;
        const std::size_t count=solidFaces.size();

        auto topologyValid=[](const HostMesh& mesh) {
            if (mesh.faceOffsets.size()!=mesh.owner.size()+1 || mesh.faceOffsets.empty()
                || mesh.faceOffsets.front()!=0
                || mesh.faceOffsets.back()!=static_cast<int>(mesh.facePoints.size())) return false;
            for (std::size_t i=1;i<mesh.faceOffsets.size();++i)
                if (mesh.faceOffsets[i]<mesh.faceOffsets[i-1]+3) return false;
            return true;
        };
        if ((!topologyValid(solid) || (!gasCandidates.empty() && !topologyValid(gas))))
            return errorAt(error,"invalid interface face CSR",-1);
        if ((thickness.size()!=count || (!gasCandidates.empty() && gasCandidates.size()!=count)))
            return errorAt(error,"nonconformal interface face counts",-1);
        if ((solid.faceOffsets.size()!=solid.owner.size()+1
            || solid.areaVectors.size()!=solid.owner.size()
            || solid.neighbour.size()!=solid.owner.size()))
            return errorAt(error,"incomplete base interface geometry",-1);
        std::vector<Vec3> normal(solid.points.size());
        std::vector<Real> weight(solid.points.size(),0), depth(solid.points.size(),0);
        std::set<int> baseFaces, candidates;
        for (std::size_t i=0;i<count;++i) {
            const int face=solidFaces[i];
            if ((face<0 || face>=static_cast<int>(solid.owner.size())
                || solid.neighbour[face]>=0 || !baseFaces.insert(face).second
                || !finite(thickness[i]) || thickness[i]<0))
                return errorAt(error,"invalid base interface face/thickness",i);
            const Real area=mag(solid.areaVectors[face]);
            if ((!finite(area) || area<=0)) return errorAt(error,"invalid base face area",i);
            for (int k=solid.faceOffsets[face];k<solid.faceOffsets[face+1];++k) {
                const int vertex=solid.facePoints[k];
                if ((vertex<0 || vertex>=static_cast<int>(solid.points.size())))
                    return errorAt(error,"invalid base interface vertex",i);
                normal[vertex]+=solid.areaVectors[face];
                weight[vertex]+=area;
                depth[vertex]+=area*thickness[i];
            }
        }
        std::vector<int> faces(count,-1), forward(solid.points.size(),-1);
        if (gasCandidates.empty()) {
            mappedGasFaces.swap(faces);solidToGas.swap(forward);error.clear();return true;
        }
        if ((gas.faceOffsets.size()!=gas.owner.size()+1 || gas.neighbour.size()!=gas.owner.size()
            || gas.areaVectors.size()!=gas.owner.size()))
            return errorAt(error,"incomplete gas interface geometry",-1);
        for (int face:gasCandidates) {
            if ((face<0 || face>=static_cast<int>(gas.owner.size())
                || gas.neighbour[face]>=0 || !candidates.insert(face).second))
                return errorAt(error,"invalid/duplicate gas interface candidate",face);
            const Real area=mag(gas.areaVectors[face]);
            if ((!finite(area) || area<=0))
                return errorAt(error,"invalid gas interface area",face);
        }
        for (std::size_t vertex=0;vertex<weight.size();++vertex) if (weight[vertex]>0) {
            if ((!(mag(normal[vertex])>0))) return errorAt(error,"cancelling base vertex normals",vertex);
            normal[vertex]=normalized(normal[vertex]);
            depth[vertex]/=weight[vertex];
        }
        std::vector<int> reverse(gas.points.size(),-1);
        std::set<int> usedFaces;
        const Real tolerance=1e-9*maxValue(meshScale(solid.points),meshScale(gas.points));
        for (std::size_t i=0;i<count;++i) {
            const int base=solidFaces[i], begin=solid.faceOffsets[base];
            const int vertices=solid.faceOffsets[base+1]-begin;
            int faceMatch=-1;
            std::vector<int> matchedVertices;
            for (int candidate:gasCandidates) {
                if ((usedFaces.count(candidate)
                    || gas.faceOffsets[candidate+1]-gas.faceOffsets[candidate]!=vertices
                    || dot(solid.areaVectors[base],gas.areaVectors[candidate])>=0)) continue;
                std::vector<int> candidateVertices(vertices,-1), localPositions(vertices,-1);
                bool valid=true;
                for (int k=0;k<vertices && valid;++k) {
                    const int sv=solid.facePoints[begin+k];
                    const Vec3 expected=solid.points[sv]+normal[sv]*depth[sv];
                    for (int j=0;j<vertices;++j) {
                        const int gv=gas.facePoints[gas.faceOffsets[candidate]+j];
                        if ((gv<0 || gv>=static_cast<int>(gas.points.size()))) {valid=false;break;}
                        if (mag(gas.points[gv]-expected)<=tolerance) {
                            if (candidateVertices[k]>=0) {valid=false;break;}
                            candidateVertices[k]=gv;localPositions[k]=j;
                        }
                    }
                    valid=valid && candidateVertices[k]>=0;
                }
                for (int k=0;k<vertices&&valid;++k)
                    valid=localPositions[(k+1)%vertices]==(localPositions[k]+vertices-1)%vertices;
                if (!valid) continue;
                if ((faceMatch>=0)) return errorAt(error,"ambiguous conforming interface face",i);
                faceMatch=candidate;matchedVertices.swap(candidateVertices);
            }
            if (faceMatch<0) {return errorAt(error,"nonmatching dry/film-offset interface vertices",i);}
            faces[i]=faceMatch;usedFaces.insert(faceMatch);
            for (int k=0;k<vertices;++k) {
                const int sv=solid.facePoints[begin+k],gv=matchedVertices[k];
                if (((forward[sv]>=0 && forward[sv]!=gv) || (reverse[gv]>=0 && reverse[gv]!=sv)))
                    return errorAt(error,"inconsistent shared interface vertex topology",i);
                forward[sv]=gv;reverse[gv]=sv;
            }
        }
        mappedGasFaces.swap(faces);solidToGas.swap(forward);error.clear();return true;
    }
    bool mapConformalInterface(const HostMesh& gas,const HostMesh& solid,const std::vector<int>& faces,
        const std::vector<Real>& thickness,const std::vector<int>& candidates,std::vector<int>& mapped,
        std::vector<int>& vertices,std::string& error) {
        return mapConformalInterfaceImpl(gas,solid,faces,thickness,candidates,mapped,vertices,error);
    }
    bool rebuildTrajectorySurfaceImpl(const HostMesh& acceptedGas,const HostMesh& acceptedSolid,
        const SurfaceMesh& reference,const HostMesh& gasEndpoint,const HostMesh& solidEndpoint,
        Real dt,SurfaceMesh& output,std::string& error,const Tolerances& tolerances,bool* numericalFailure) {
        if(numericalFailure)*numericalFailure=true;

        if ((!(dt>0) || !finite(dt))) return errorAt(error,"invalid surface trajectory interval",-1);
        const int ns=static_cast<int>(reference.solidFace.size());
        if ((reference.gasFace.size()!=reference.solidFace.size()
            || !reference.gasMapOffsets.empty() || !reference.gasMapFaces.empty() || !reference.gasMapWeights.empty()))
            return errorAt(error,"nonconformal/invalid trajectory surface mapping unsupported",-1);
        HostMesh oldSolid=acceptedSolid,oldGas=acceptedGas,endSolid=solidEndpoint,endGas=gasEndpoint;
        if (!rebuildGeometryImpl(oldSolid,error,numericalFailure) || !rebuildGeometryImpl(oldGas,error,numericalFailure)
            || !rebuildGeometryImpl(endSolid,error,numericalFailure) || !rebuildGeometryImpl(endGas,error,numericalFailure)) return false;
        if ((oldSolid.topologyHash!=endSolid.topologyHash || oldGas.topologyHash!=endGas.topologyHash))
            return errorAt(error,"surface trajectory topology change unsupported",-1);
        HostMesh solid,gas; std::vector<Real> actual;
        if (!makeStageGeometryImpl(oldSolid,endSolid.points,dt,solid,actual,error,tolerances,numericalFailure)
            || !makeStageGeometryImpl(oldGas,endGas.points,dt,gas,actual,error,tolerances,numericalFailure)) return false;
        std::set<int> usedSolid,usedGas;
        for (int i=0;i<ns;++i) {
            const int f=reference.solidFace[i],g=reference.gasFace[i];
            if ((f<0 || f>=static_cast<int>(solid.owner.size()) || solid.neighbour[f]>=0
                || g<-1 || g>=static_cast<int>(gas.owner.size()) || (g>=0 && gas.neighbour[g]>=0)
                || !usedSolid.insert(f).second || (g>=0 && !usedGas.insert(g).second)))
                return errorAt(error,"invalid/duplicate trajectory surface face",i);
            if ((g>=0 && (oldSolid.faceOffsets[f+1]-oldSolid.faceOffsets[f]
                !=oldGas.faceOffsets[g+1]-oldGas.faceOffsets[g])))
                return errorAt(error,"nonconformal trajectory surface topology unsupported",i);
        }
        SurfaceMesh surface = reference;
        surface.oldArea.resize(ns);
        surface.area.resize(ns);
        surface.centre.resize(ns);
        surface.normal.resize(ns);
        surface.meshVelocity.resize(ns);
        surface.baseVelocity.resize(ns);
        surface.gasDistance.resize(ns);
        surface.solidDistance.resize(ns);
        for (int i = 0; i<ns; ++i) {
            int f = reference.solidFace[i];
            surface.oldArea[i] = mag(oldSolid.areaVectors[f]);
            surface.area[i] = mag(solid.areaVectors[f]);
            surface.centre[i] = solid.faceCentres[f];
            surface.normal[i] = normalized(solid.areaVectors[f]);
            surface.meshVelocity[i] = (solid.faceCentres[f]-oldSolid.faceCentres[f])/dt;
            int g = reference.gasFace[i];
            surface.solidDistance[i] = dot(solid.faceCentres[f]-solid.cellCentres[solid.owner[f]],
                surface.normal[i]);
            surface.gasDistance[i] = g >= 0?dot(gas.faceCentres[g]-gas.cellCentres[gas.owner[g]],
                normalized(gas.areaVectors[g])):0;
            if ((!(surface.solidDistance[i]>0) || (g >= 0 && !(surface.gasDistance[i]>0))))return errorAt(error,
                "nonpositive interface normal distance", i);
        }
        std::vector<SurfaceEdge> edges;
        if (!surfaceEdges(oldSolid, reference, edges, error,numericalFailure))return false;
        bool emptySolid=false;
        for(auto kind:oldSolid.boundaryKind)emptySolid=emptySolid || kind==BoundaryKind::Empty;
        surface.edgeOwner.clear();
        surface.edgeNeighbour.clear();
        surface.edgeLength.clear();
        surface.edgeConormal.clear();
        surface.sweptEdgeArea.clear();
        surface.edgeOwnerOffset.clear();
        surface.edgeNeighbourOffset.clear();
        const Real edgeMeshScale=edges.empty()?Real(0):meshScale(oldSolid.points);
        for (const auto& edge:edges) {
            Vec3 tangent = solid.points[edge.b]-solid.points[edge.a], normal = surface.normal[edge.owner];
            if (edge.neighbour >= 0)normal = normalized(normal+surface.normal[edge.neighbour]);
            Vec3 conormal = normalized(cross(tangent, normal));
            if ((mag(conormal) == 0))return errorAt(error, "degenerate surface edge conormal", edge.owner);
            Vec3 edgeCentre = (solid.points[edge.a]+solid.points[edge.b])*.5;
            surface.edgeOwnerOffset.push_back(edgeCentre-surface.centre[edge.owner]);
            Vec3 neighbourOffset{};
            if (edge.neighbour >= 0) {
                const Vec3 neighbourEdgeCentre = (solid.points[edge.neighbourA]+solid.points[edge.neighbourB])*.5;
                neighbourOffset = neighbourEdgeCentre-surface.centre[edge.neighbour];
            }
            surface.edgeNeighbourOffset.push_back(neighbourOffset);
            surface.edgeOwner.push_back(edge.owner);
            surface.edgeNeighbour.push_back(edge.neighbour);
            surface.edgeLength.push_back(mag(tangent));
            surface.edgeConormal.push_back(conormal);
            Real sweep = edgeSweepQuadrature(oldSolid, solid.points, reference, edge, 8,emptySolid);
            bool converged = false;
            const Real scale = edgeMeshScale*edgeMeshScale;
            Real quadratureResidual=0,quadratureTolerance=0;
            for (int segments = 16; segments <= 1024; segments *= 2) {
                Real refined = edgeSweepQuadrature(oldSolid, solid.points, reference, edge, segments,emptySolid);
                if ((!finite(refined) || !finite(sweep)))return errorAt(error, "degenerate moving surface hinge",
                    edge.owner);
                const Real tolerance = 256*std::numeric_limits<Real>::epsilon()*scale
                    +1e-11*maxValue(absValue(refined), absValue(sweep));
                quadratureResidual=absValue(refined-sweep); quadratureTolerance=tolerance;
                if (quadratureResidual <= tolerance) {
                    sweep = refined;
                    converged = true;
                    break;
                }
                sweep = refined;
            }
            if (!converged) geometryWarning(tolerances.diagnostics,quadratureResidual,quadratureTolerance,edge.owner,false);
            surface.sweptEdgeArea.push_back(sweep);
        }
        output=std::move(surface); error.clear(); return true;
    }
    bool rebuildGeometry(HostMesh& m,std::string& e) {return rebuildGeometryImpl(m,e,nullptr);}
    bool makeStageAreaVectors(const HostMesh& m,const std::vector<Vec3>& p,std::vector<Vec3>& a,std::string& e) {
        return makeStageAreaVectorsImpl(m,p,a,e,nullptr);
    }
    bool makeStageGeometry(const HostMesh& m,const std::vector<Vec3>& p,double dt,HostMesh& out,
        std::vector<double>& sweep,std::string& e,const Tolerances& tol,bool* numericalFailure) {
        return makeStageGeometryImpl(m,p,dt,out,sweep,e,tol,numericalFailure);
    }
    bool rebuildTrajectorySurface(const HostMesh& ag,const HostMesh& as,const SurfaceMesh& ref,
        const HostMesh& g,const HostMesh& so,Real dt,SurfaceMesh& out,std::string& e,const Tolerances& tol,bool* numericalFailure) {
        return rebuildTrajectorySurfaceImpl(ag,as,ref,g,so,dt,out,e,tol,numericalFailure);
    }

    bool rebuildTrajectorySurface(const HostState& base,const HostMesh& gas,const HostMesh& solid,
        Real interval,SurfaceMesh& output,std::string& error,const Tolerances& tolerances,bool* numericalFailure) {
        return rebuildTrajectorySurface(base.gasMesh,base.solidMesh,base.surface,gas,solid,interval,output,error,tolerances,numericalFailure);
    }
    bool moveCoupledMeshesConstrainedImpl(const HostState& accepted, const std::vector<FilmAux>& candidateFilm,
        const std::vector<Real>& solidTargets, Real dt, HostMesh& gasOut, HostMesh& solidOut,
        SurfaceMesh& surfaceOut, SweepConstraintReport& report, std::string& error,
        const SweepConstraintControls& controls, const Tolerances& tolerances,bool* numericalFailure) {
        report=SweepConstraintReport{};
        if(numericalFailure)*numericalFailure=true;

        if ((!finite(dt) || dt <= 0))return errorAt(error, "invalid coupled motion interval", -1);
        const auto& reference = accepted.surface;
        const int ns = static_cast<int>(reference.solidFace.size());
        if ((!reference.gasMapOffsets.empty() || !reference.gasMapFaces.empty()
            || !reference.gasMapWeights.empty()))return errorAt(error,
            "nonconformal coupled surface mapping not implemented", -1);
        if ((reference.gasFace.size() != reference.solidFace.size()
            || solidTargets.size() != reference.solidFace.size()
            || candidateFilm.size() != reference.solidFace.size()
            || accepted.filmAux.size() != reference.solidFace.size()))return errorAt(error,
            "coupled surface/film size mismatch", -1);
        HostMesh oldSolid = accepted.solidMesh, oldGas = accepted.gasMesh;
        if (!rebuildGeometryImpl(oldSolid,error,numericalFailure) || !rebuildGeometryImpl(oldGas,error,numericalFailure))return false;
        if (ns == 0) {
            std::vector<Real>s; HostMesh gas,solid;
            if (!makeStageGeometryImpl(oldGas,oldGas.points,dt,gas,s,error,tolerances,numericalFailure) || !makeStageGeometryImpl(oldSolid,oldSolid.points,dt,solid,s,error,tolerances,numericalFailure))return false;
            gasOut=std::move(gas); solidOut=std::move(solid); surfaceOut = reference;
            report.status=SweepConstraintStatus::Success;
            return true;
        }
        std::vector<Real> acceptedThickness(ns);
        std::vector<int> gasCandidates;
        for (int i=0;i<ns;++i) {
            acceptedThickness[i]=accepted.filmAux[i].thickness;
            if (reference.gasFace[i]>=0) gasCandidates.push_back(reference.gasFace[i]);
        }
        std::vector<int> mappedGasFaces,solidToGas;
        if (!mapConformalInterfaceImpl(oldGas,oldSolid,reference.solidFace,acceptedThickness,
            gasCandidates,mappedGasFaces,solidToGas,error,numericalFailure))return false;
        if ((mappedGasFaces!=reference.gasFace))
            return errorAt(error,"coupled face identity differs from canonical correspondence",-1);
        std::vector<unsigned char> fixedSolid, fixedGas;
        fixedPhysicalBoundaries(oldSolid, fixedSolid);
        fixedPhysicalBoundaries(oldGas, fixedGas);
        std::vector<Vec3> displacement(oldSolid.points.size()),directions(oldSolid.points.size()),support(oldSolid.points.size());
        std::vector<Real> weight(oldSolid.points.size(), 0), newThickness(weight);
        for (int i = 0; i<ns; ++i) {
            int f = reference.solidFace[i], g = reference.gasFace[i];
            if ((f<0 || f >= static_cast<int>(oldSolid.owner.size()) || g<-1
                || g >= static_cast<int>(oldGas.owner.size()) || oldSolid.neighbour[f] >= 0 || (g >= 0
                && oldGas.neighbour[g] >= 0)))return errorAt(error, "invalid coupled physical face", i);
            const auto& old = accepted.filmAux[i];
            const auto& now = candidateFilm[i];
            Vec3 n = normalized(oldSolid.areaVectors[f]);
            Real area = mag(oldSolid.areaVectors[f]);
            if ((!finite(now.thickness) || now.thickness<0 || !finite(old.thickness) || old.thickness<0
                || !finite(now.solidFront) || !finite(old.solidFront)
                || !finite(now.baseVelocity)))return errorAt(error, "invalid coupled front data", i);
            if ((absValue(dot(now.baseVelocity, n))>1e-12*maxValue(1,
                mag(now.baseVelocity))))return errorAt(error,
                "normal moving support requires separate support trajectory (unsupported)", i);
            const Vec3 delta = n*(now.solidFront-old.solidFront)+now.baseVelocity*dt;
            for (int k = oldSolid.faceOffsets[f]; k<oldSolid.faceOffsets[f+1]; ++k) {
                int v = oldSolid.facePoints[k];
                displacement[v] += delta*area;
                support[v] += now.baseVelocity*(dt*area);
                directions[v] += oldSolid.areaVectors[f];
                weight[v] += area;
                newThickness[v] += now.thickness*area;
                fixedSolid[v] = 1;
            }
        }
        for (std::size_t v = 0; v<weight.size(); ++v)if (weight[v]>0) {
            displacement[v] = displacement[v]/weight[v];
            support[v] = support[v]/weight[v];
            newThickness[v] /= weight[v];
        }
        // Material inventories are attached to cells; there is no condensed-
        // material remap here. Keep recession vertices on adjacent non-interface
        // face planes. In a wedge this selects radial, rather than averaged
        // surface-normal, motion. Prescribed support velocity is not projected.
        // The full trajectory/planarity checks remain authoritative.
        std::vector<std::vector<Vec3>> sideNormals(oldSolid.points.size());
        for (std::size_t f=0;f<oldSolid.owner.size();++f)if(oldSolid.boundaryKind[f]!=BoundaryKind::Interface){
            const Vec3 normal=normalized(oldSolid.areaVectors[f]);
            for(int k=oldSolid.faceOffsets[f];k<oldSolid.faceOffsets[f+1];++k){
                const int p=oldSolid.facePoints[k];if(weight[p]<=0)continue;
                Vec3 q=normal;for(const auto& n:sideNormals[p])q-=n*dot(q,n);
                if(mag(q)>1e-10)sideNormals[p].push_back(normalized(q));
            }
        }
        for(std::size_t p=0;p<directions.size();++p){
            Vec3 recession=displacement[p]-support[p];
            for(const auto& n:sideNormals[p]){
                recession-=n*dot(recession,n);
                directions[p]-=n*dot(directions[p],n);
            }
            if(sideNormals[p].size()==3){recession={};directions[p]={};}
            displacement[p]=recession+support[p];
        }
        for (auto& direction:directions) direction=normalized(direction);
        HostMesh gas,solid;
        std::vector<Real> gasSweeps,solidSweeps;
        auto buildGas=[&](const HostMesh& trialSolid,std::string& failure) {
            auto stage=[&](const std::vector<Vec3>& points) {
                return makeStageGeometry(oldGas,points,dt,gas,gasSweeps,failure,tolerances);
            };
            // An unchanged accepted offset is already authoritative. Recomputing
            // X_s+n*delta here can manufacture a one-ULP gas motion after a
            // coordinate translation and falsely activate the moving-film guard.
            bool unchanged=true;
            for (std::size_t p=0;p<oldSolid.points.size();++p)
                unchanged=unchanged && mag(trialSolid.points[p]-oldSolid.points[p])==0;
            for (int i=0;i<ns;++i)
                unchanged=unchanged && candidateFilm[i].thickness==accepted.filmAux[i].thickness;
            if (unchanged) return stage(oldGas.points);
            std::vector<Vec3> newNormal(oldSolid.points.size());
            for (int f:reference.solidFace)
                for (int k=trialSolid.faceOffsets[f];k<trialSolid.faceOffsets[f+1];++k)
                    newNormal[trialSolid.facePoints[k]]+=trialSolid.areaVectors[f];
            for (auto& n:newNormal) n=normalized(n);
            std::vector<Vec3> gasDisplacement(oldGas.points.size());
            for (std::size_t sv=0;sv<solidToGas.size();++sv) if (solidToGas[sv]>=0) {
                const int gv=solidToGas[sv];
                gasDisplacement[gv]=trialSolid.points[sv]+newNormal[sv]*newThickness[sv]-oldGas.points[gv];
                fixedGas[gv]=1;
            }
            std::vector<Vec3> gasPoints;
            const bool harmonicOk=harmonicPointMotion(oldGas,gasDisplacement,fixedGas,gasPoints,failure,tolerances.diagnostics);
            if(!harmonicOk || !stage(gasPoints))return false;
            // Keep the existing reduced-film boundary: moving curved offset
            // interfaces with unequal areas/normals are not a liquid-volume model.
            for (int i=0;i<ns;++i) {
                const int f=reference.solidFace[i],g=reference.gasFace[i];
                if (g<0 || (accepted.filmAux[i].thickness==0 && candidateFilm[i].thickness==0)) continue;
                bool moving=false;
                for (int k=oldSolid.faceOffsets[f];k<oldSolid.faceOffsets[f+1];++k) {
                    const int sv=oldSolid.facePoints[k],gv=solidToGas[sv];
                    moving=moving || mag(trialSolid.points[sv]-oldSolid.points[sv])>0
                        || (gv>=0 && mag(gas.points[gv]-oldGas.points[gv])>0);
                }
                auto offset=[&](const HostMesh& gm,const HostMesh& sm) {
                    const Real ga=mag(gm.areaVectors[g]),sa=mag(sm.areaVectors[f]);
                    return absValue(ga-sa)>1e-10*maxValue(ga,sa)
                        || mag(normalized(gm.areaVectors[g])+normalized(sm.areaVectors[f]))>1e-10;
                };
                if (moving && (offset(oldGas,oldSolid) || offset(gas,trialSolid)))
                    return errorAt(failure,"moving curved wet-offset geometry unsupported",i);
            }
            return true;
        };
        // Every material-bearing route (predictor, integrated corrector and
        // legacy wrapper) must meet the same cell-inventory accuracy. Geometry-
        // only callers retain their explicit controls and geometric floor.
        SweepConstraintControls materialControls=controls;
        if (!accepted.solid.empty()) {
            std::vector<Real> limits;
            if ((accepted.solid.size()!=oldSolid.volumes.size())
                ||!materialSweepResidualLimits(oldSolid,reference.solidFace,limits)
                ||(!controls.faceVolumeTolerance.empty()&&controls.faceVolumeTolerance.size()!=limits.size()))
                return errorAt(error,"invalid material swept-volume accuracy ownership",-1);
            for (Real limit:controls.faceVolumeTolerance)if((!finite(limit)||limit<=0))
                return errorAt(error,"invalid per-face material swept-volume error ceiling",-1);
            if (!controls.faceVolumeTolerance.empty())for(std::size_t i=0;i<limits.size();++i)
                limits[i]=minValue(limits[i],controls.faceVolumeTolerance[i]);
            materialControls.faceVolumeTolerance=std::move(limits);
        }
        // Physical transfers remain immutable. Only the geometry solve carries
        // the previously unrepresented signed volume into this target.
        std::vector<Real> solveTargets;
        if(!compensateMaterialSweepTargets(accepted,solidTargets,solveTargets,error,tolerances.diagnostics)){return false;}
        // Directed representable rounding leaves a dense material cell enough
        // geometric volume. This is an auxiliary geometry target only: the
        // physical transfer is immutable, and updateMaterialSweepRemainder
        // carries physical-minus-measured volume into the next accepted step.
        // The original requested accuracy and explicit face ceilings still cap
        // the total difference from the unshifted physical target.
        const auto physicalSolveTargets=solveTargets;
        auto physicalLimits=materialControls.faceVolumeTolerance;
        const Real materialMeshScale=!accepted.solid.empty()&&!solveTargets.empty()
            ?meshScale(oldSolid.points):Real(0);
        if(!accepted.solid.empty())for(std::size_t f=0;f<solveTargets.size();++f){
            const Real requested=controls.absoluteVolumeTolerance+controls.relativeVolumeTolerance*absValue(solveTargets[f]);
            const Real floor=32*std::numeric_limits<Real>::epsilon()*mag(oldSolid.areaVectors[reference.solidFace[f]])*materialMeshScale;
            physicalLimits[f]=minValue(physicalLimits[f],maxValue(requested,floor));
            if(solveTargets[f]!=0){
                solveTargets[f]+=physicalLimits[f]*.5;
                materialControls.faceVolumeTolerance[f]=physicalLimits[f]*.5;
            }else materialControls.faceVolumeTolerance[f]=physicalLimits[f];
        }
        if (!constrainedMotion(oldSolid,reference.solidFace,solveTargets,displacement,fixedSolid,directions,
            dt,solid,solidSweeps,report,error,materialControls,buildGas,tolerances,numericalFailure)) return false;
        if(!accepted.solid.empty()){
            report.maximumSweepResidual=0;
            for(std::size_t f=0;f<physicalLimits.size();++f){
                const Real residual=solidSweeps[reference.solidFace[f]]-physicalSolveTargets[f];
                if(absValue(residual)>physicalLimits[f]) geometryWarning(tolerances.diagnostics,residual,physicalLimits[f],f);
                report.maximumSweepResidual=maxValue(report.maximumSweepResidual,absValue(residual));
            }
        }
        SurfaceMesh surface;
        if (!rebuildTrajectorySurfaceImpl(oldGas,oldSolid,reference,gas,solid,dt,surface,error,tolerances,numericalFailure)) {
            report.status=SweepConstraintStatus::InvalidTrajectory; return false;
        }
        for (int i=0;i<ns;++i) surface.baseVelocity[i]=candidateFilm[i].baseVelocity;
        gasOut = std::move(gas);
        solidOut = std::move(solid);
        surfaceOut = std::move(surface);
        error.clear();
        return true;
    }
    bool moveCoupledMeshesConstrained(const HostState& state,const std::vector<FilmAux>& film,const std::vector<Real>& targets,
        Real dt,HostMesh& gas,HostMesh& solid,SurfaceMesh& surface,SweepConstraintReport& report,std::string& error,
        const SweepConstraintControls& controls,const Tolerances& tolerances,bool* numericalFailure) {
        try{return moveCoupledMeshesConstrainedImpl(state,film,targets,dt,gas,solid,surface,report,error,controls,tolerances,numericalFailure);}
        catch(const std::bad_alloc&){if(numericalFailure)*numericalFailure=false;throw;}
        catch(const std::length_error&){if(numericalFailure)*numericalFailure=false;throw;}
    }

    bool moveCoupledMeshes(const HostState& accepted, const std::vector<FilmAux>& candidateFilm, double dt,
        HostMesh& gasOut, HostMesh& solidOut, SurfaceMesh& surfaceOut, std::string& error,
        const Tolerances& tolerances) {
        // Compatibility with the original explicit front convention only. The
        // multirate material owner must call the explicit integrated-volume API.
        if (candidateFilm.size()!=accepted.surface.solidFace.size()
            || accepted.filmAux.size()!=candidateFilm.size())
            return errorAt(error,"coupled surface/film size mismatch",-1);
        HostMesh old=accepted.solidMesh;
        if (!rebuildGeometry(old,error)) return false;
        std::vector<Real> target(candidateFilm.size());
        for (std::size_t i=0;i<target.size();++i) {
            const int f=accepted.surface.solidFace[i];
            if (f<0 || f>=static_cast<int>(old.owner.size()))
                return errorAt(error,"invalid coupled physical face",static_cast<int>(i));
            target[i]=mag(old.areaVectors[f])*(candidateFilm[i].solidFront-accepted.filmAux[i].solidFront);
        }
        SweepConstraintReport report;
        return moveCoupledMeshesConstrained(accepted,candidateFilm,target,dt,gasOut,solidOut,surfaceOut,
            report,error,SweepConstraintControls{},tolerances);
    }

}
// namespace chmt
