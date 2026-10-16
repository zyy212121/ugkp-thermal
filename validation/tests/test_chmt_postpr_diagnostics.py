"""Exercise CHMT invalid comparison/stencil paths and finite retained solutions."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_stencil_and_interval_physical_errors(tmp_path):
    source = tmp_path / 'probe.cpp'
    source.write_text(r'''
#include "materials/StencilMath.H"
#include "coupling/IntervalAudit.H"
#include <cassert>
#include <limits>
using namespace chmt;
int main(){
 const Real inf=std::numeric_limits<Real>::infinity();
 Symmetric3 identity;identity.xx=identity.yy=identity.zz=1;
 const auto solved=solveLeastSquares(identity,{2,3,4});
 assert(solved.x==2&&solved.y==3&&solved.z==4);
 assert(!chmt::finite(solveLeastSquares(identity,{inf,0,0})));
 auto corrupt=identity;corrupt.xy=inf;
 assert(!chmt::finite(solveLeastSquares(corrupt,{1,0,0})));
 Symmetric3 rankOne;rankOne.xx=1;
 const auto rank=solveLeastSquares(rankOne,{2,0,0});assert(rank.x==2&&rank.y==0&&rank.z==0);
 const auto zero=solveLeastSquares(Symmetric3{},{});assert(zero.x==0&&zero.y==0&&zero.z==0);
 Vec3 rhs{};addLeastSquares(rankOne,rhs,{inf,0,0},1);assert(!chmt::finite(solveLeastSquares(rankOne,rhs)));
 assert(!chmt::finite(intervalNormalizedError(inf,1,1,0)));
 assert(intervalNormalizedError(2,1,0,0)>1&&chmt::finite(intervalNormalizedError(2,1,0,0)));
 WallProgram a;a.interval.end=1;a.surface.area={1};a.knots.resize(2);
 for(int i=0;i<2;++i){a.knots[i].time=i;a.knots[i].faces.resize(1);a.knots[i].faces[0].temperature=300;a.knots[i].gasPoints={{0,0,0},{1,0,0}};}
 auto b=a;Tolerances t;WallProgramComparison result;std::string error;
 assert(compareWallPrograms(a,b,t,result,error)&&result.thermalError==0);
 for(auto& k:b.knots)k.faces[0].temperature=500;
 assert(compareWallPrograms(a,b,t,result,error)&&result.thermalError>1&&chmt::finite(result.thermalError));
 b.knots[0].faces[0].temperature=inf;
 assert(!compareWallPrograms(a,b,t,result,error));
 b=a;b.knots[0].gasPoints[0].x=inf;assert(!compareWallPrograms(a,b,t,result,error));
 b=a;b.knots[0].faces[0].speciesRate[0]=inf;assert(!compareWallPrograms(a,b,t,result,error));
}
''')
    exe = tmp_path / 'probe'
    subprocess.run(['g++','-std=c++17','-O1','-g','-fsanitize=undefined',
                    '-fno-sanitize-recover=undefined','-I'+str(SOURCE_ROOT/'applications/CHMT'),
                    '-I'+str(SOURCE_ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_active_reconstruction_consumers_and_reaction_budget(tmp_path):
    source=tmp_path/'consumers.cpp'
    source.write_text(r'''
#define main historical_reaction_probe
#include "validation/solvers/CHMT/devtools/multirate/test_reaction_accuracy.cpp"
#undef main
#include "materials/MaterialTransport.C"
#include "film/CpuFilmDriver.C"
#include <cassert>
int main(){
 auto p=config();ugkp::diagnostics::DeviceContext diagnostics{};p.diagnostics=&diagnostics;
 SolidQ out;std::uint64_t count=0;std::string error;
 assert(advanceMaterialReactions(initial(p),.001,p,.001,0,.1,1,out,count,error));
 assert(count>1&&diagnostics.warnings[1].count==1&&!diagnostics.fatalClaim);
 const auto reference=solve(p,initial(p),.001,.1);
 assert(count==reference.steps&&out.energy==reference.q.energy&&out.condensed[0]==reference.q.condensed[0]&&out.condensed[1]==reference.q.condensed[1]);
 HostMesh m;m.volumes={1,1};m.cellCentres={{0,0,0},{1,0,0}};m.faceCentres={{.5,0,0}};
 m.owner={0};m.neighbour={1};m.cellFaceOffsets={0,1,2};m.cellFaces={0,0};m.cellFaceSigns={1,-1};m.boundaryKind={BoundaryKind::Internal};
 std::vector<SolidQ> states(2);states[1].energy=std::numeric_limits<Real>::infinity();out.energy=123;
 assert(!reconstruct(m,states,p,0,0,out)&&out.energy==123);
 SurfaceMesh surface;surface.area={1,1};surface.edgeOwner={0};surface.edgeNeighbour={1};surface.edgeOwnerOffset={{.5,0,0}};surface.edgeNeighbourOffset={{-.5,0,0}};
 Evaluation e;e.q.resize(2);e.aux.resize(2);e.incidentOffsets={0,1,2};e.incident={0,0};e.q[0].mass=e.q[1].mass=1;e.q[1].enthalpy=std::numeric_limits<Real>::infinity();
 FilmQ film;film.mass=123;assert(!reconstruct(0,{.5,0,0},e,surface,p,film)&&film.mass==123);
 e.aux[1].pressure=std::numeric_limits<Real>::infinity();assert(!chmt::finite(surfaceGradient(0,-1,e,surface)));
 Symmetric3 matrix;matrix.xx=1;assert(!chmt::finite(darcyGradientSensitivity(matrix,{std::numeric_limits<Real>::infinity(),0,0})));
}
''')
    exe=tmp_path/'consumers'
    subprocess.run(['g++','-std=c++17','-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=undefined',
                    '-I'+str(SOURCE_ROOT),'-I'+str(SOURCE_ROOT/'applications/CHMT'),
                    '-I'+str(SOURCE_ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
