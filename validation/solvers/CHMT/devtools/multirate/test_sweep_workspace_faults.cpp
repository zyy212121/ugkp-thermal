// Test-only single-allocation fault injection; production has no global new hook.
// Enumerates every allocation in successful cached and streamed transactions.
#include <cstdlib>
#include <new>
#include <cstdio>
#include <cstddef>
static long failAfter=-1,allocationCalls=0,liveAllocations=0;
void* operator new(std::size_t n){++allocationCalls;if(failAfter==0){failAfter=-1;throw std::bad_alloc();}if(failAfter>0)--failAfter;void*p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();++liveAllocations;return p;}
void operator delete(void*p)noexcept{if(p){--liveAllocations;std::free(p);}}
void operator delete(void*p,std::size_t)noexcept{::operator delete(p);}
#include "mesh/SweepWorkspace.H"
#define main inherited_sweep_tests_main
#include "test_sweep_constraints.cpp"
#undef main
int main(){
 using namespace sweep_detail;
 std::size_t peak=0;sweep_detail::Budget budget(128,peak);
 { Reals x{Allocator<Real>(budget)};x.reserve(8);require(budget.live==64&&peak==64,"reserve charged");
   try{x.reserve(16);require(false,"old+new simultaneously charged");}catch(const WorkspaceLimit&){}
   require(budget.live==64&&x.capacity()==8,"failed resize preserves budget and capacity");
   Reals y(std::move(x));require(budget.live==64,"move does not double charge"); }
 require(budget.live==0,"destruction releases budget");
 for(int operation=0;operation<2;++operation){try{if(operation)sum(std::size_t(-1),1);else product(std::size_t(-1),2);require(false,"overflow rejected");}catch(const WorkspaceLimit&){}}
 {Allocator<Real> a(budget);failAfter=0;try{a.allocate(2);require(false,"allocation fault required");}catch(const std::bad_alloc&){}require(budget.live==0,"failed underlying allocate leaves accounting unchanged");}
 const HostMesh mesh=tetrahedralBulk();std::vector<int> faces;std::vector<Vec3> directions(mesh.points.size()),zero(mesh.points.size());std::vector<unsigned char> fixed(mesh.points.size(),1);fixed[4]=0;
 for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0)faces.push_back(f);
 for(int p=0;p<4;++p)directions[p]=normalized(mesh.points[p]-mesh.points[4]);
 std::vector<Real> targets{-.01,-.02,-.015,-.01};
 SweepConstraintControls controls;std::size_t cachedPeak=0;
 for(int mode=0;mode<2;++mode){
  if(mode)controls.maximumWorkspaceBytes=cachedPeak-1;
  long baselineCalls;{
   HostMesh output;output.geometryVersion=777;std::vector<Real>sweeps{123};SweepConstraintReport report;std::string error;
   long before=allocationCalls;bool ok=constrainFaceSweeps(mesh,faces,targets,zero,fixed,directions,1,output,sweeps,report,error,controls);baselineCalls=allocationCalls-before;
   require(ok,error);require(report.streamedModes==bool(mode),"mode routing");if(!mode)cachedPeak=report.peakWorkspaceBytes;
  }
  long rejected=0,accepted=0;
  for(long at=0;at<baselineCalls;++at){long beforeLive=liveAllocations;{
   HostMesh output;output.geometryVersion=777;std::vector<Real>sweeps{123};SweepConstraintReport report;std::string error;
   failAfter=at;bool ok=false;
   try{ok=constrainFaceSweeps(mesh,faces,targets,zero,fixed,directions,1,output,sweeps,report,error,controls);}catch(...){failAfter=-1;std::fprintf(stderr,"escaped allocation fault mode=%d at=%ld\n",mode,at);return 9;}failAfter=-1;
   if(!ok){++rejected;require(report.status==SweepConstraintStatus::SizeLimit,"allocation fault typed SizeLimit");require(output.geometryVersion==777&&output.points.empty()&&sweeps==std::vector<Real>{123},"allocation fault outputs unchanged");}
   else ++accepted;
   require(report.peakWorkspaceBytes<=controls.maximumWorkspaceBytes,"accounted peak within budget");
  }if(liveAllocations!=beforeLive){std::fprintf(stderr,"live delta mode=%d at=%ld before=%ld after=%ld\n",mode,at,beforeLive,liveAllocations);return 8;}}
  std::printf("mode=%d baseline_allocations=%ld rejected=%ld accepted=%ld\n",mode,baselineCalls,rejected,accepted);
 }
 std::puts("PASS allocator preflight/overflow/capacity/move/fault cleanup and public cached/streamed allocation transaction");
}
