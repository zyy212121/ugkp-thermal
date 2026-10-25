
#include "mesh/Geometry.C"
#include "geometry_execution_reference.H"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <type_traits>
using namespace chmt;
namespace {
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
    static_assert(std::is_trivially_copyable<T>::value,"raw numerical fixture field");
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool sameMesh(const HostMesh& a,const HostMesh& b) {
#define FIELD(f) if(!same(a.f,b.f)) {std::cerr<<"mesh field " #f " differs\n";return false;}
    FIELD(referencePoints) FIELD(points) FIELD(oldPoints) FIELD(cellCentres)
    FIELD(faceCentres) FIELD(areaVectors) FIELD(meshVelocity) FIELD(volumes)
    FIELD(oldVolumes) FIELD(wallDistance) FIELD(faceOffsets) FIELD(facePoints)
    FIELD(owner) FIELD(neighbour) FIELD(periodicPartner) FIELD(cellFaceOffsets)
    FIELD(cellFaces) FIELD(cellFaceSigns) FIELD(boundaryKind) FIELD(boundaryPrimitive)
    FIELD(thermalBoundary) FIELD(boundarySst) FIELD(faceIds)
#undef FIELD
    return a.geometryVersion==b.geometryVersion && a.topologyHash==b.topologyHash;
}
HostMesh boxes(int cells,Real scale=1,Real offset=0,bool skew=false,bool empty=false,bool overlap=false) {
    HostMesh m;m.faceOffsets.push_back(0);
    for(int c=0;c<cells;++c) {
        Real x=2*c;if(overlap && c==61)x=120.5;
        const int first=m.points.size();
        for(Vec3 p:std::vector<Vec3>{{x,0,0},{x+1,0,0},{x+1,1,0},{x,1,0},
            {x,0,1},{x+1,0,1},{x+1,1,1},{x,1,1}})
            m.points.push_back({offset+scale*(p.x+(skew?.125*p.y:0)),
                -offset+scale*(p.y+(skew?.0625*p.z:0)),offset+scale*p.z});
        int face=0;
        for(const auto& f:std::vector<std::vector<int>>{{0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}}) {
            for(int p:f)m.facePoints.push_back(first+p);
            m.faceOffsets.push_back(m.facePoints.size());m.owner.push_back(c);m.neighbour.push_back(-1);
            m.boundaryKind.push_back(empty && face<2?BoundaryKind::Empty:BoundaryKind::Slip);++face;
        }
    }
    m.referencePoints=m.points;m.oldPoints={{-7,8,9}};m.oldVolumes={41};m.meshVelocity={{-0.,2,3}};
    m.geometryVersion=19;m.topologyHash=31;m.boundaryPrimitive.resize(m.owner.size());
    m.boundarySst.resize(m.owner.size());m.thermalBoundary.resize(m.owner.size());
    for(std::size_t f=0;f<m.owner.size();++f){m.boundaryPrimitive[f].temperature=f+273;m.boundarySst[f].k=f;}
    return m;
}
int checks=0;
void environment(int rounding) {
    assert(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);errno=E2BIG;
}
void check(HostMesh mesh,std::vector<Vec3> endpoint,int rounding,Real dt=0.125,Tolerances tol={}) {
    std::string aError="sentinel",bError=aError;HostMesh a=boxes(1),b=a;
    std::vector<Real> aSweep={-0.,7},bSweep=aSweep;bool aNumerical=false,bNumerical=false;
    environment(rounding);
    const bool aOk=referenceMakeStageGeometry(mesh,endpoint,dt,a,aSweep,aError,tol,&aNumerical);
    const int aFlags=std::fetestexcept(FE_ALL_EXCEPT),aErrno=errno,aRounding=std::fegetround();
    environment(rounding);
    const bool bOk=makeStageGeometry(mesh,endpoint,dt,b,bSweep,bError,tol,&bNumerical);
    const int bFlags=std::fetestexcept(FE_ALL_EXCEPT),bErrno=errno,bRounding=std::fegetround();
    if(aOk!=bOk || aError!=bError || aNumerical!=bNumerical || !sameMesh(a,b) || !same(aSweep,bSweep)
        || aFlags!=bFlags || aErrno!=bErrno || aRounding!=bRounding) {
        std::cerr<<"stage check "<<checks<<" differs: "<<aOk<<'/'<<bOk<<" errors "<<aError<<'/'<<bError
            <<" flags "<<aFlags<<'/'<<bFlags<<" errno "<<aErrno<<'/'<<bErrno<<'\n';std::abort();
    }
    a=mesh;b=mesh;aError=bError="sentinel";
    environment(rounding);const bool ar=referenceGeometryInPlace(a,aError);
    const int af=std::fetestexcept(FE_ALL_EXCEPT),ae=errno;
    environment(rounding);const bool br=geometryInPlace(b,bError);
    const int bf=std::fetestexcept(FE_ALL_EXCEPT),be=errno;
    assert(ar==br && aError==bError && sameMesh(a,b) && af==bf && ae==be);
    ++checks;
}
void exercise(HostMesh m,int rounding,bool empty=false) {
    check(m,m.points,rounding);
    auto moved=m.points;for(auto& p:moved){p.x+=.03125;p.y+=.0625;if(!empty)p.z*=1.015625;}
    check(m,moved,rounding);
    auto zero=m.points;for(auto& p:zero){if(p.x==0)p.x=-0.;if(p.y==0)p.y=-0.;if(p.z==0)p.z=-0.;}
    check(m,zero,rounding);
}
}
int main() {
#ifdef _OPENMP
    omp_set_dynamic(0);
    // Seed persistent workers with different environments. The geometry caller
    // must neither inherit these modes nor overwrite them after its computation.
#pragma omp parallel
    {std::fesetround(omp_get_thread_num()%2?FE_DOWNWARD:FE_UPWARD);
     std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID);errno=EDOM;}
#endif
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        std::fesetround(FE_TONEAREST);
        exercise(boxes(1),rounding);exercise(boxes(3,.001,11.125,true),rounding);
        exercise(boxes(2,0x1p-8,0x1p20),rounding);exercise(boxes(2,0x1p10,-0x1p20,true),rounding);
        exercise(boxes(2,1,0,false,true),rounding,true);exercise(HostMesh{},rounding);
        // More than 65536 ordered triangle queries enters the parallel path.
        exercise(boxes(128),rounding);exercise(boxes(128,.125,7.25,true),rounding);
        auto invalid=boxes(128,1,0,false,false,true);check(invalid,invalid.points,rounding);
        invalid=boxes(2);invalid.facePoints[2]=invalid.facePoints[1];check(invalid,invalid.points,rounding);
        auto m=boxes(2);auto end=m.points;end.pop_back();check(m,end,rounding);
        end=m.points;end[0].x=std::numeric_limits<Real>::quiet_NaN();check(m,end,rounding);
        end=m.points;for(auto& p:end){p.x=-p.x;p.y=-p.y;}check(m,end,rounding);
        check(m,m.points,rounding,0);Tolerances tol;tol.absoluteGeometry=-1;check(m,m.points,rounding,.125,tol);
        // Noncanonical signed zero and interpolation-roundoff fixtures.
        m=boxes(1,.1,.2);m.points[0].z=-0.;check(m,m.points,rounding);
    }
#ifdef _OPENMP
    int workerFailures=0;
#pragma omp parallel reduction(+:workerFailures)
    if(omp_get_thread_num()>0) {
        if(std::fegetround()!=(omp_get_thread_num()%2?FE_DOWNWARD:FE_UPWARD)
            || std::fetestexcept(FE_ALL_EXCEPT)!=FE_INVALID || errno!=EDOM)++workerFailures;
    }
    assert(workerFailures==0);
#endif
    std::fesetround(FE_TONEAREST);
    std::cout<<"exact geometry reference comparisons: "<<checks<<'\n';
}
