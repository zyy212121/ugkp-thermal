#include "mesh/SweepConstraints.H"
#include "mesh/Geometry.C"
#include <iostream>
#include <cfenv>
#include <fenv.h>
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

int main(){auto mesh=tetrahedralBulk();std::vector<int> faces;for(int f=0;f<int(mesh.owner.size());++f)if(mesh.neighbour[f]<0)faces.push_back(f);
std::vector<Vec3> zero(5),direction(5),known(5),points;std::vector<unsigned char> fixed(5,1);fixed[4]=0;
for(int j=0;j<4;++j){direction[j]=normalized(mesh.points[j]-mesh.points[4]);known[j]=direction[j]*.3;}
std::string error;require(harmonicPointMotion(mesh,known,fixed,points,error),error);HostMesh expected;std::vector<Real> all;
require(makeStageGeometry(mesh,points,1,expected,all,error),error);std::vector<Real>target;for(int f:faces)target.push_back(all[f]);
ugkp::diagnostics::DeviceContext ctx{};Tolerances tol;tol.diagnostics=&ctx;SweepConstraintControls controls;controls.maximumIterations=60;
HostMesh out;out.volumes={123};std::vector<Real> sweeps={987};SweepConstraintReport report;
auto checkInvalid=[&](std::vector<int> fs,std::vector<Real> ts,double dt,SweepConstraintControls c,Tolerances t){
 require(!constrainFaceSweeps(mesh,fs,ts,zero,fixed,direction,dt,out,sweeps,report,error,c,t),"invalid input accepted");
 require(report.status==SweepConstraintStatus::InvalidInput&&report.lineSearchTrials==0,"invalid input was searched");
 require(out.volumes==std::vector<Real>{123}&&sweeps==std::vector<Real>{987},"failed operation changed output");};
auto bad=faces;bad[0]=-1;checkInvalid(bad,target,1,controls,tol);bad=faces;bad[1]=bad[0];checkInvalid(bad,target,1,controls,tol);
auto nonfinite=target;nonfinite[0]=std::numeric_limits<Real>::quiet_NaN();checkInvalid(faces,nonfinite,1,controls,tol);
checkInvalid(faces,target,0,controls,tol);auto invalidTolerance=tol;invalidTolerance.absoluteGeometry=-1;checkInvalid(faces,target,1,controls,invalidTolerance);
controls.maximumIterations=1;require(constrainFaceSweeps(mesh,faces,target,zero,fixed,direction,1,out,sweeps,report,error,controls,tol),error);
require(report.maximumSweepResidual>1e-10&&ctx.warnings[0].count>0&&ctx.fatalClaim==0,"finite exhausted accuracy policy changed");
controls.maximumIterations=60;ctx={};std::vector<Vec3> noDirections(5);
require(constrainFaceSweeps(mesh,faces,target,zero,fixed,noDirections,1,out,sweeps,report,error,controls,tol),error);
require(report.rank==0&&report.maximumSweepResidual>0&&ctx.fatalClaim==0,"rank-deficient finite accuracy policy changed");
// Real producer calls with fault injection at its private geometry callback.
ctx={};const int savedMask=fegetexcept();std::feclearexcept(FE_ALL_EXCEPT);
feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);const int trialMask=fegetexcept();
int trialCalls=0,finalCalls=0;TrajectoryCheck domain=[&](const HostMesh&,std::string& e,ugkp::diagnostics::DeviceContext* d,bool&){
 if(d==&ctx){++finalCalls;return true;}++trialCalls;if(trialCalls==1){volatile double large=std::numeric_limits<double>::max();volatile double overflow=large*large;(void)overflow;ugkp::diagnostics::recordFatal(d,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,777001));e="injected trial domain";return false;}return true;};
require(constrainedMotion(mesh,faces,target,zero,fixed,direction,1,out,sweeps,report,error,controls,domain,tol),error);
require(trialCalls>1&&finalCalls==1&&ctx.fatalClaim==0,"trial domain poisoned final state");
require(fegetexcept()==trialMask&&std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW)==0,"trial altered caller trap mask/error flags");
ctx={};trialCalls=0;auto injected=ugkp::diagnostics::Code::MissingOperator;TrajectoryCheck missing=[&](const HostMesh&,std::string&e,ugkp::diagnostics::DeviceContext*d,bool&){++trialCalls;ugkp::diagnostics::recordFatal(d,ugkp::diagnostics::makeRecord(injected,777002));e="injected missing operator";return false;};
require(!constrainedMotion(mesh,faces,target,zero,fixed,direction,1,out,sweeps,report,error,controls,missing,tol),"missing operator hidden");
require(trialCalls==1&&ctx.fatalClaim&&ctx.firstFatal.code==ugkp::diagnostics::Code::MissingOperator,"real configuration failure was searched/suppressed");
ctx={};trialCalls=0;injected=ugkp::diagnostics::Code::InvalidIndex;require(!constrainedMotion(mesh,faces,target,zero,fixed,direction,1,out,sweeps,report,error,controls,missing,tol),"bad index hidden");require(trialCalls==1&&ctx.fatalClaim&&ctx.firstFatal.code==injected,"real index failure was searched/suppressed");
require(fegetexcept()==trialMask,"true error altered caller trap mask");fedisableexcept(FE_ALL_EXCEPT);feenableexcept(savedMask);
std::cout<<"invalid-input,finite-budget,rank,trial-domain,missing-operator passed\n";
}
