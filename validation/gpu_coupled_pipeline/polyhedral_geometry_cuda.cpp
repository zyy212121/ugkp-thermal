// The original Geometry.C is selected from a separate frozen checkout by -I.
#include "mesh/Geometry.C"
#include "gpu/PolyhedralGeometry.H"
#include <cassert>
#include <iostream>
#include <cstring>
namespace device=ugkp::gpu::geometry;
using chmt::HostMesh;using chmt::Vec3;
std::vector<device::Point> points(const std::vector<Vec3>& p){std::vector<device::Point> out;for(auto x:p)out.push_back({x.x,x.y,x.z});return out;}
void exact(double a,double b){if(std::memcmp(&a,&b,sizeof(double))){std::cerr.precision(17);std::cerr<<"mismatch "<<a<<' '<<b<<'\n';std::abort();}}
void exact(Vec3 a,device::Point b){exact(a.x,b.x);exact(a.y,b.y);exact(a.z,b.z);}
HostMesh boxes(int cells,double scale,bool skew){
    HostMesh m;m.faceOffsets.push_back(0);
    for(int c=0;c<cells;++c){const double x=2*c;const int first=m.points.size();
        for(Vec3 p:std::vector<Vec3>{{x,0,0},{x+1,0,0},{x+1,1,0},{x,1,0},{x,0,1},{x+1,0,1},{x+1,1,1},{x,1,1}})
            m.points.push_back({11.125+scale*(p.x+(skew?.125*p.y:0)),-3+scale*(p.y+(skew?.0625*p.z:0)),scale*p.z});
        for(const auto& face:std::vector<std::vector<int>>{{0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}}){
            for(int p:face)m.facePoints.push_back(first+p);m.faceOffsets.push_back(m.facePoints.size());
            m.owner.push_back(c);m.neighbour.push_back(-1);m.boundaryKind.push_back(chmt::BoundaryKind::Slip);}}
    return m;
}
device::Topology topology(const HostMesh& m){
    device::Topology t;t.faceOffsets=m.faceOffsets;t.facePoints=m.facePoints;t.cellOffsets=m.cellFaceOffsets;t.cellFaces=m.cellFaces;t.cellSigns=m.cellFaceSigns;
    t.vertexOffsets.push_back(0);
    for(int c=0;c<int(m.volumes.size());++c){std::set<int> vertices;
        for(int k=m.cellFaceOffsets[c];k<m.cellFaceOffsets[c+1];++k){const int f=m.cellFaces[k];
            for(int j=m.faceOffsets[f];j<m.faceOffsets[f+1];++j)vertices.insert(m.facePoints[j]);}
        t.vertices.insert(t.vertices.end(),vertices.begin(),vertices.end());t.vertexOffsets.push_back(t.vertices.size());}
    return t;
}
int main(){
    device::PolyhedralGeometry gpu;int geometries=0,motions=0;double maximumWallRelative=0,maximumGclRelative=0;
    for(int count:{1,3,128})for(double scale:{.001,1.,1024.})for(bool skew:{false,true}){
        auto mesh=boxes(count,scale,skew);std::string error;assert(chmt::rebuildGeometry(mesh,error));
        auto t=topology(mesh);const auto p=points(mesh.points);auto faces=gpu.faces(t,p);
        std::vector<device::Point> area;
        for(std::size_t f=0;f<faces.size();++f){assert(!faces[f].invalid);exact(mesh.areaVectors[f],faces[f].area);exact(mesh.faceCentres[f],faces[f].centre);area.push_back(faces[f].area);}
        auto cells=gpu.cells(t,p,area);
        for(std::size_t c=0;c<cells.size();++c){assert(!cells[c].invalid);exact(mesh.volumes[c],cells[c].volume);exact(mesh.cellCentres[c],cells[c].centre);
            Vec3 closure;double total=0;for(int k=mesh.cellFaceOffsets[c];k<mesh.cellFaceOffsets[c+1];++k){const int f=mesh.cellFaces[k];closure+=mesh.areaVectors[f]*mesh.cellFaceSigns[k];total+=chmt::mag(mesh.areaVectors[f]);}
            exact(closure,cells[c].closure);exact(total,cells[c].areaSum);}
        std::vector<int> walls;for(std::size_t f=0;f<mesh.owner.size();++f)walls.push_back(f);
        auto distance=gpu.wallDistance(t,p,points(mesh.cellCentres),walls);
        for(std::size_t c=0;c<distance.size();++c){assert(distance[c].invalidFace<0);
            const double error=std::abs((distance[c].distance-mesh.wallDistance[c])/mesh.wallDistance[c]);
            maximumWallRelative=std::max(maximumWallRelative,error);assert(error<2e-13);}
        for(int moving=0;moving<2;++moving){auto end=mesh.points;
            if(moving)for(auto& x:end){x.x+=.03125*scale;x.y+=.0625*scale;x.z*=1.015625;}
            std::vector<Vec3> expectedArea;std::vector<double> expectedSweep;assert(chmt::stageIntegrals(mesh,end,expectedArea,expectedSweep,error));
            auto sweep=gpu.sweep(t,p,points(end));for(std::size_t f=0;f<sweep.size();++f){assert(!sweep[f].invalid);exact(expectedArea[f],sweep[f].area);exact(expectedSweep[f],sweep[f].volume);}
            auto newMesh=mesh;newMesh.points=end;assert(chmt::rebuildGeometry(newMesh,error));
            const auto newFaces=gpu.faces(t,points(end));std::vector<device::Point> newArea;for(auto f:newFaces)newArea.push_back(f.area);
            const auto newCells=gpu.cells(t,points(end),newArea);
            for(std::size_t c=0;c<newCells.size();++c){exact(newMesh.volumes[c],newCells[c].volume);double sum=0;
                for(int k=mesh.cellFaceOffsets[c];k<mesh.cellFaceOffsets[c+1];++k)sum+=mesh.cellFaceSigns[k]*sweep[mesh.cellFaces[k]].volume;
                const double residual=std::abs(newCells[c].volume-cells[c].volume-sum)/std::max(newCells[c].volume,cells[c].volume);
                maximumGclRelative=std::max(maximumGclRelative,residual);assert(residual<2e-13);}
            ++geometries;}
        std::vector<std::set<int>> adjacency(mesh.points.size());
        for(std::size_t f=0;f<mesh.owner.size();++f)for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k){
            const int a=mesh.facePoints[k],b=mesh.facePoints[k+1==mesh.faceOffsets[f+1]?mesh.faceOffsets[f]:k+1];adjacency[a].insert(b);adjacency[b].insert(a);}
        std::vector<int> offsets(1,0),neighbours;for(const auto& row:adjacency){neighbours.insert(neighbours.end(),row.begin(),row.end());offsets.push_back(neighbours.size());}
        std::vector<unsigned char> fixed(mesh.points.size(),0);for(int c=0;c<count;++c){fixed[c*8]=1;fixed[c*8+6]=1;}
        device::HarmonicPointMotion motion(p,offsets,neighbours,fixed,1e-12*chmt::meshScale(mesh.points));
        for(int mode=0;mode<3;++mode){std::vector<Vec3> boundary(mesh.points.size());
            for(int c=0;c<count;++c){boundary[c*8]={.001*(c+1),.003*(mode+1),-.007};boundary[c*8+6]={-.002*(mode+1),.001*(c+1),.004};}
            std::vector<Vec3> expected;assert(chmt::harmonicPointMotion(mesh,boundary,fixed,expected,error));
            const auto actual=motion.solve(points(boundary));assert(actual.errorPoint<0);
            for(std::size_t v=0;v<expected.size();++v)exact(expected[v],actual.points[v]);
            const auto represented=motion.solve(points(boundary),true);assert(represented.errorPoint<0);
            for(std::size_t v=0;v<expected.size();++v)exact(fixed[v]?boundary[v]:expected[v]-mesh.points[v],represented.points[v]);
            ++motions;}
    }
    device::Topology empty;empty.faceOffsets={0};empty.cellOffsets={0};empty.vertexOffsets={0};
    assert(gpu.faces(empty,{}).empty()&&gpu.cells(empty,{},{}).empty()&&gpu.sweep(empty,{},{}).empty());
    device::HarmonicPointMotion noPoints({}, {0},{},{},1e-12);const auto none=noPoints.solve({});assert(none.points.empty()&&none.iterations==0&&none.errorPoint<0);
    std::cout.precision(17);std::cout<<"GEOMETRY_METRICS maximumWallRelative="<<maximumWallRelative<<" maximumGclRelative="<<maximumGclRelative<<'\n';
    std::cout<<"PASS frozen geometry: "<<geometries<<" static/moving face-cell-sweep configurations and "<<motions<<" repeated harmonic modes (positions and represented displacements) bitwise; empty reuse\n";
}
