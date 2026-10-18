#include "mesh/SweepConstraints.H"
#include <iostream>
#include <map>
#include <algorithm>
#include <random>
#include <iomanip>
using namespace chmt;
void require(bool x,const std::string&e){if(!x){std::cerr<<e;std::exit(2);}}
HostMesh tetrahedralBulk() {
    HostMesh mesh; mesh.points={{0,0,0},{1.1,.1,0},{.2,1.2,.1},{.1,.2,1.3},{.35,.375,.35}};
    const int cells[4][4]={{4,1,2,3},{0,4,2,3},{0,1,4,3},{0,1,2,4}};
    std::map<std::vector<int>,int> incidence; std::vector<std::vector<int>> polygons;
    for (int c=0;c<4;++c) for (int omitted=0;omitted<4;++omitted) {
        std::vector<int> face;
        for (int j=0;j<4;++j) if (j!=omitted) face.push_back(cells[c][j]);
        Vec3 a=mesh.points[face[0]],b=mesh.points[face[1]],d=mesh.points[face[2]];
        if (dot(cross(b-a,d-a),mesh.points[cells[c][omitted]]-a)>0) std::swap(face[1],face[2]);
        auto key=face; std::sort(key.begin(),key.end()); auto hit=incidence.find(key);
        if (hit!=incidence.end()) {
            mesh.neighbour[hit->second]=c; mesh.boundaryKind[hit->second]=BoundaryKind::Internal;
        } else {
            incidence[key]=static_cast<int>(polygons.size()); polygons.push_back(face);
            mesh.owner.push_back(c); mesh.neighbour.push_back(-1); mesh.boundaryKind.push_back(BoundaryKind::Interface);
        }
    }
    mesh.faceOffsets.push_back(0);
    for (const auto& face:polygons) {
        mesh.facePoints.insert(mesh.facePoints.end(),face.begin(),face.end());
        mesh.faceOffsets.push_back(static_cast<int>(mesh.facePoints.size()));
    }
    std::string error; require(rebuildGeometry(mesh,error),error); return mesh;
}
int main(){int failures=0;auto mesh=tetrahedralBulk();std::vector<int> faces;for(int f=0;f<int(mesh.owner.size());f++)if(mesh.neighbour[f]<0)faces.push_back(f);std::vector<unsigned char>fixed(5,1);fixed[4]=0;std::mt19937_64 gen(928731);std::uniform_real_distribution<double>u(-1,1);std::cout<<std::setprecision(17);
for(int k=0;k<600;k++){std::vector<Vec3>direction(5),known(5),zero(5),points;for(int j=0;j<4;j++){direction[j]=normalized(mesh.points[j]-mesh.points[4]);known[j]=direction[j]*(.7*u(gen)+.2);}std::string error;harmonicPointMotion(mesh,known,fixed,points,error);HostMesh expected;std::vector<Real>targetAll;bool valid=makeStageGeometry(mesh,points,1,expected,targetAll,error);if(!valid){std::cout<<k<<" INVALID\n";continue;}std::vector<Real>target;for(int f:faces)target.push_back(targetAll[f]);HostMesh out;std::vector<Real>sweeps;ugkp::diagnostics::DeviceContext ctx{};Tolerances tol;tol.diagnostics=&ctx;SweepConstraintControls c;c.maximumIterations=60;SweepConstraintReport r;bool ok=constrainFaceSweeps(mesh,faces,target,zero,fixed,direction,1,out,sweeps,r,error,c,tol);double err=0;if(ok)for(int j=0;j<int(faces.size());j++)err=std::max(err,std::abs(sweeps[faces[j]]-target[j]));std::cout<<k<<' '<<ok<<' '<<r.iterations<<' '<<r.lineSearchTrials<<' '<<err<<' '<<error<<" fatal="<<ctx.fatalClaim<<" modes="<<r.modeSolves<<" warnings="<<(ctx.warnings[0].count+ctx.warnings[1].count+ctx.warnings[2].count)<<'\n';if(!ok||err>=1e-6||ctx.fatalClaim||r.modeSolves!=std::size_t(2*r.degreesOfFreedom*r.iterations))++failures;}return failures?1:0;}
