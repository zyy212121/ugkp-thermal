#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>

using namespace chmt;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
HostMesh twoHexahedra(Real bottom) {
    HostMesh mesh;
    auto index=[](int i,int j,int k) { return i+3*j+6*k; };
    for (int k=0;k<2;++k) for (int j=0;j<2;++j) for (int i=0;i<3;++i)
        mesh.points.push_back({Real(i),Real(j),bottom+k});
    std::vector<std::vector<int>> polygons;
    std::map<std::vector<int>,int> incidence;
    for (int c=0;c<2;++c) {
        int a=index(c,0,0), b=index(c+1,0,0), d=index(c,1,0), e=index(c+1,1,0);
        int A=index(c,0,1), B=index(c+1,0,1), D=index(c,1,1), E=index(c+1,1,1);
        for (const auto& polygon:std::vector<std::vector<int>>{
            {a,d,e,b},{A,B,E,D},{a,b,B,A},{d,D,E,e},{a,A,D,d},{b,e,E,B}}) {
            auto key=polygon; std::sort(key.begin(),key.end());
            auto found=incidence.find(key);
            if (found!=incidence.end()) {
                mesh.neighbour[found->second]=c;
                mesh.boundaryKind[found->second]=BoundaryKind::Internal;
            } else {
                incidence[key]=static_cast<int>(polygons.size());
                polygons.push_back(polygon); mesh.owner.push_back(c); mesh.neighbour.push_back(-1);
                mesh.boundaryKind.push_back(BoundaryKind::Slip);
            }
        }
    }
    mesh.faceOffsets.push_back(0);
    for (const auto& polygon:polygons) {
        mesh.facePoints.insert(mesh.facePoints.end(),polygon.begin(),polygon.end());
        mesh.faceOffsets.push_back(static_cast<int>(mesh.facePoints.size()));
    }
    std::string error; require(rebuildGeometry(mesh,error),error); return mesh;
}
HostState coupledPair() {
    HostState base; base.solidMesh=twoHexahedra(0); base.gasMesh=twoHexahedra(1);
    for (int c=0;c<2;++c) {
        int solid=-1,gas=-1;
        for (int f=0;f<static_cast<int>(base.solidMesh.owner.size());++f)
            if (base.solidMesh.owner[f]==c && base.solidMesh.faceCentres[f].z==1
                && base.solidMesh.areaVectors[f].z>0) solid=f;
        for (int f=0;f<static_cast<int>(base.gasMesh.owner.size());++f)
            if (base.gasMesh.owner[f]==c && base.gasMesh.faceCentres[f].z==1
                && base.gasMesh.areaVectors[f].z<0) gas=f;
        require(solid>=0 && gas>=0,"test fixture conforming face lookup");
        base.surface.solidFace.push_back(solid); base.surface.gasFace.push_back(gas);
        base.solidMesh.boundaryKind[solid]=BoundaryKind::Interface;
        base.gasMesh.boundaryKind[gas]=BoundaryKind::Interface;
        base.surface.solidCell.push_back(c); base.surface.persistentId.push_back(c);
    }
    base.filmAux.resize(2);
    std::string error; require(rebuildGeometry(base.solidMesh,error),error);
    require(rebuildGeometry(base.gasMesh,error),error); return base;
}
void actualNonuniformDrySweep() {
    const HostState base=coupledPair(); auto candidate=base.filmAux;
    candidate[0].solidFront=-0.04; candidate[1].solidFront=-0.10;
    HostMesh gas,solid; SurfaceMesh surface; std::string error;
    require(moveCoupledMeshes(base,candidate,1,gas,solid,surface,error),error);
    HostMesh measured; std::vector<Real> solidSweep,gasSweep;
    require(makeStageGeometry(base.solidMesh,solid.points,1,measured,solidSweep,error),error);
    require(makeStageGeometry(base.gasMesh,gas.points,1,measured,gasSweep,error),error);
    for (int i=0;i<2;++i) {
        Real target=candidate[i].solidFront*mag(base.solidMesh.areaVectors[base.surface.solidFace[i]]);
        std::ostringstream detail; detail << "nonuniform shared-vertex face " << i
            << " actual sweep " << solidSweep[base.surface.solidFace[i]] << " target " << target;
        require(absValue(solidSweep[base.surface.solidFace[i]]-target)<1e-11,detail.str());
        require(absValue(gasSweep[base.surface.gasFace[i]]+target)<1e-11,"actual opposite gas sweep");
    }
    require(maximumGclResidual(base.solidMesh,solid,solidSweep)<1e-12,"independent solid GCL");
    require(maximumGclResidual(base.gasMesh,gas,gasSweep)<1e-12,"independent gas GCL");
}
void explicitIntegratedTargets() {
    const HostState base=coupledPair(); auto candidate=base.filmAux;
    // Deliberately stale front guesses: only the integrated volume is authoritative.
    candidate[0].solidFront=-.003; candidate[1].solidFront=-.006;
    const std::vector<Real> target={-.023,-.071};
    HostMesh gas,solid; SurfaceMesh surface; SweepConstraintReport report; std::string error;
    require(moveCoupledMeshesConstrained(base,candidate,target,.7,gas,solid,surface,report,error),error);
    require(report.status==SweepConstraintStatus::Success && report.iterations>0,"explicit target correction ran");
    HostMesh measured; std::vector<Real> sweeps;
    require(makeStageGeometry(base.solidMesh,solid.points,.7,measured,sweeps,error),error);
    for (int i=0;i<2;++i)
        require(absValue(sweeps[base.surface.solidFace[i]]-target[i])<1e-12,"integrated target overrides front guess");
    require(candidate[0].solidFront==-.003 && candidate[1].solidFront==-.006,"physical front input unchanged");
}
void rejectIncompatibleTargets() {
    const HostState base=coupledPair(); const auto& mesh=base.solidMesh;
    std::vector<Vec3> displacement(mesh.points.size()),direction(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1);
    for (std::size_t p=0;p<mesh.points.size();++p)
        if (mesh.points[p].x==1 && mesh.points[p].z==1) direction[p]={0,0,1};
    HostMesh output=base.gasMesh; output.geometryVersion=987;
    const auto sentinelPoints=output.points; std::vector<Real> sweeps={123,456};
    SweepConstraintReport report; std::string error;
    require(!constrainFaceSweeps(mesh,base.surface.solidFace,{-.03,-.08},displacement,fixed,direction,
        1,output,sweeps,report,error),"incompatible shared-edge targets must be rejected");
    require(report.status==SweepConstraintStatus::Incompatible,"numerical rank/incompatibility diagnostic: "+error);
    require(report.rank<report.constraints,"rank deficiency reported");
    require(output.geometryVersion==987 && output.points.size()==sentinelPoints.size()
        && sweeps==std::vector<Real>({123,456}),"failed solve preserves output sentinels");
    for (std::size_t p=0;p<sentinelPoints.size();++p)
        require(mag(output.points[p]-sentinelPoints[p])==0,"failed solve preserves output coordinates");
}
void stationaryNeedsNoDenseSolve() {
    const HostState base=coupledPair(); HostMesh gas,solid; SurfaceMesh surface;
    SweepConstraintControls controls; controls.maximumDegreesOfFreedom=1; controls.maximumConstraints=1;
    SweepConstraintReport report; std::string error;
    require(moveCoupledMeshesConstrained(base,base.filmAux,{0,0},1,gas,solid,surface,report,error,controls),
        "unchanged valid trajectory must bypass dense-solve size limits: "+error);
    require(report.iterations==0 && report.lineSearchTrials==0,"stationary needs no correction");
}
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
// Independent five-point Gauss quadrature of velocity dot moving triangle area.
Real quadratureSweep(const HostMesh& mesh,const std::vector<Vec3>& endpoint,int f) {
    const Real node[5]={-.9061798459386639928,-.538469310105683091,0,.538469310105683091,.9061798459386639928};
    const Real weight[5]={.236926885056189088,.478628670499366468,.568888888888888889,.478628670499366468,.236926885056189088};
    const int begin=mesh.faceOffsets[f]; Real result=0;
    for (int k=begin+1;k+1<mesh.faceOffsets[f+1];++k) {
        int a=mesh.facePoints[begin],b=mesh.facePoints[k],c=mesh.facePoints[k+1];
        Vec3 da=endpoint[a]-mesh.points[a],db=endpoint[b]-mesh.points[b],dc=endpoint[c]-mesh.points[c];
        for (int q=0;q<5;++q) {
            const Real t=(node[q]+1)/2;
            Vec3 ab=mesh.points[b]+db*t-mesh.points[a]-da*t;
            Vec3 ac=mesh.points[c]+dc*t-mesh.points[a]-da*t;
            result+=weight[q]*dot((da+db+dc)/3,cross(ab,ac))*.25;
        }
    }
    return result;
}
void nonlinearThreeDimensionalRecession() {
    const HostMesh mesh=tetrahedralBulk(); std::vector<int> faces;
    for (int f=0;f<static_cast<int>(mesh.owner.size());++f) if (mesh.neighbour[f]<0) faces.push_back(f);
    std::vector<Vec3> directions(mesh.points.size()),knownDisplacement(mesh.points.size()),zero(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1); fixed[4]=0;
    const Real amplitudes[4]={-.019,-.041,-.032,-.027};
    for (int p=0;p<4;++p) {
        directions[p]=normalized(mesh.points[p]-mesh.points[4]);
        knownDisplacement[p]=directions[p]*amplitudes[p];
    }
    std::vector<Vec3> known; std::string error;
    require(harmonicPointMotion(mesh,knownDisplacement,fixed,known,error),error);
    std::vector<Real> target;
    for (int f:faces) target.push_back(quadratureSweep(mesh,known,f));
    HostMesh output; std::vector<Real> sweeps; SweepConstraintReport report;
    require(constrainFaceSweeps(mesh,faces,target,zero,fixed,directions,.31,output,sweeps,report,error),error);
    require(report.iterations>=2,"nonlinear 3-D correction needs repeated exact sweep evaluation");
    require(mag(output.points[4]-mesh.points[4])>1e-5,"harmonic interior point actually moves");
    for (std::size_t i=0;i<faces.size();++i) {
        require(absValue(quadratureSweep(mesh,output.points,faces[i])-target[i])<1e-12,
            "nonuniform nonparallel 3-D sweeps match independent quadrature");
        require(absValue(sweeps[faces[i]]-target[i])<1e-12,"actual returned sweeps");
    }
    for (int p=0;p<4;++p)
        require(mag(cross(output.points[p]-mesh.points[p],directions[p]))<1e-13,"boundary correction stays normal");
    require(maximumGclResidual(mesh,output,sweeps)<1e-12,"tetrahedral bulk independent GCL");
}
void prescribedTrajectoryGclAndComposition() {
    const HostMesh mesh=tetrahedralBulk(); auto endpoint=mesh.points;
    for (std::size_t p=0;p<endpoint.size();++p) {
        Vec3 x=endpoint[p]; endpoint[p]+={.018*x.y-.009*x.z,.007*x.x+.011*x.z,-.023*x.y+.016*x.x};
    }
    HostMesh whole,mid,finish; std::vector<Real> all,first,second; std::string error;
    require(makeStageGeometry(mesh,endpoint,1,whole,all,error),error);
    auto halfway=mesh.points;
    for (std::size_t p=0;p<endpoint.size();++p) halfway[p]=mesh.points[p]*.63+endpoint[p]*.37;
    require(makeStageGeometry(mesh,halfway,.37,mid,first,error),error);
    require(makeStageGeometry(mid,endpoint,.63,finish,second,error),error);
    for (std::size_t f=0;f<all.size();++f) {
        require(absValue(all[f]-first[f]-second[f])<1e-13,"geometric sweeps compose across substeps");
        require(absValue(all[f]-quadratureSweep(mesh,endpoint,static_cast<int>(f)))<1e-13,"prescribed trajectory actual sweep");
    }
    require(maximumGclResidual(mesh,whole,all)<1e-13,"independent whole-step GCL");
    require(maximumGclResidual(mesh,mid,first)<1e-13,"independent first-substep GCL");
    require(maximumGclResidual(mid,finish,second)<1e-13,"independent second-substep GCL");
}
void rejectInvalidTrajectory() {
    const HostState base=coupledPair(); HostMesh gas=base.gasMesh,solid=base.solidMesh;
    SurfaceMesh surface=base.surface; gas.geometryVersion=77; solid.geometryVersion=88;
    SweepConstraintReport report; std::string error;
    require(!moveCoupledMeshesConstrained(base,base.filmAux,{-3,-3},1,gas,solid,surface,report,error),
        "impossible volume loss must not pass a GCL-only check");
    require(report.status==SweepConstraintStatus::InvalidTrajectory || report.status==SweepConstraintStatus::Nonconverged,
        "invalid trajectory diagnosis: "+error);
    require(gas.geometryVersion==77 && solid.geometryVersion==88,"coupled rejection is transactional");
}

void rebuildActualTrajectorySurface() {
    HostState base=coupledPair(); auto candidate=base.filmAux;
    candidate[0].solidFront=-.025; candidate[1].solidFront=-.09;
    HostMesh gas,solid; SurfaceMesh expected; std::string error;
    require(moveCoupledMeshes(base,candidate,1,gas,solid,expected,error),error);
    base.surface.area={-1,-1}; base.surface.normal={{99,99,99},{99,99,99}};
    base.surface.edgeLength={-99}; base.surface.sweptEdgeArea={-77};
    base.surface.baseVelocity={{.02,0,0},{.02,0,0}};
    base.surface.prescribedTopTraction={{1,2,3},{4,5,6}};
    SurfaceMesh result;
    require(rebuildTrajectorySurface(base,gas,solid,1,result,error),error);
    require(result.area==expected.area && result.edgeLength==expected.edgeLength
        && result.sweptEdgeArea==expected.sweptEdgeArea,"surface endpoint recomputes actual edge/area metrics");
    require(result.edgeOwner==expected.edgeOwner && result.edgeNeighbour==expected.edgeNeighbour,"surface edge topology rebuilt");
    require(result.baseVelocity[0].x==.02 && result.prescribedTopTraction[1].z==6,"nongeometric surface data preserved");
    HostMesh halfSolid,halfGas; auto sp=solid.points,gp=gas.points; std::vector<Real> sweep;
    for (std::size_t p=0;p<sp.size();++p) sp[p]=(sp[p]+base.solidMesh.points[p])*.5;
    for (std::size_t p=0;p<gp.size();++p) gp[p]=(gp[p]+base.gasMesh.points[p])*.5;
    require(makeStageGeometry(base.solidMesh,sp,.5,halfSolid,sweep,error),error);
    require(makeStageGeometry(base.gasMesh,gp,.5,halfGas,sweep,error),error);
    SurfaceMesh mid,second;
    require(rebuildTrajectorySurface(base,halfGas,halfSolid,.5,mid,error),error);
    require(rebuildTrajectorySurface(halfGas,halfSolid,mid,gas,solid,.5,second,error),error);
    require(absValue(mid.area[0]-result.area[0])>1e-6,"midpoint metrics differ from endpoint");
    for (std::size_t e=0;e<result.sweptEdgeArea.size();++e)
        require(absValue(result.sweptEdgeArea[e]-mid.sweptEdgeArea[e]-second.sweptEdgeArea[e])<1e-11,
            "actual surface edge sweep composes across clock subintervals");
    auto bad=solid; bad.faceIds[0]+=100000; bad.facePoints[0]=bad.facePoints[1];
    SurfaceMesh sentinel=result;
    require(!rebuildTrajectorySurface(base,gas,bad,1,sentinel,error),"invalid endpoint topology rejected");
    require(sentinel.area==result.area && sentinel.edgeLength==result.edgeLength,"surface rebuild rollback");
}
void rejectInteriorBoundaryDof() {
    const HostMesh mesh=tetrahedralBulk(); std::vector<Vec3> displacement(mesh.points.size()),direction(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1); direction[4]={0,0,1};
    HostMesh output; std::vector<Real> sweeps; SweepConstraintReport report; std::string error;
    require(!constrainFaceSweeps(mesh,{}, {},displacement,fixed,direction,1,output,sweeps,report,error),
        "interior point must not masquerade as boundary correction DOF");
    require(report.status==SweepConstraintStatus::InvalidInput,"interior DOF diagnostic");
}
HostState curvedOffsetPair() {
    HostState base=coupledPair();
    for (auto& p:base.solidMesh.points) if (p.z==1) p.z+=p.x==0?-.06:(p.x==2?-.13:0);
    std::string error; require(rebuildGeometry(base.solidMesh,error),error);
    std::vector<Vec3> normal(base.solidMesh.points.size());
    for (int f:base.surface.solidFace) for (int k=base.solidMesh.faceOffsets[f];k<base.solidMesh.faceOffsets[f+1];++k)
        normal[base.solidMesh.facePoints[k]]+=base.solidMesh.areaVectors[f];
    for (int sv=6;sv<12;++sv) base.gasMesh.points[sv-6]=base.solidMesh.points[sv]+normalized(normal[sv])*.02;
    for (auto& aux:base.filmAux) aux.thickness=.02;
    require(rebuildGeometry(base.gasMesh,error),error); return base;
}
void retainMovingOffsetRejection() {
    const HostState base=curvedOffsetPair(); HostMesh gas,solid; SurfaceMesh surface;
    SweepConstraintReport report; std::string error;
    require(moveCoupledMeshesConstrained(base,base.filmAux,{0,0},1,gas,solid,surface,report,error),
        "stationary curved offset interface retains supported geometry: "+error);
    auto candidate=base.filmAux; candidate[0].solidFront=-.01; candidate[1].solidFront=-.01;
    require(!moveCoupledMeshesConstrained(base,candidate,{-.01,-.01},1,gas,solid,surface,report,error),
        "moving curved offset must remain unsupported");
    require(error.find("unsupported")!=std::string::npos,"explicit moving-offset unsupported diagnosis: "+error);
}

void scaledRotatedDryMotion() {
    for (Real scale:std::vector<Real>{1e-4,1,1e3}) {
        HostState base=coupledPair();
        auto transform=[&](Vec3 x) {
            // Two independent rotations: the interface normal has all 3 components.
            const Real c=.8,t=.6; Vec3 y={c*x.x-t*x.y,t*x.x+c*x.y,x.z};
            return Vec3{.6*y.x+.8*y.z,y.y,-.8*y.x+.6*y.z}*scale;
        };
        for (auto& p:base.solidMesh.points) p=transform(p);
        for (auto& p:base.gasMesh.points) p=transform(p);
        std::string error; require(rebuildGeometry(base.solidMesh,error),error);
        require(rebuildGeometry(base.gasMesh,error),error);
        SweepConstraintControls controls; controls.absoluteVolumeTolerance=1e-14*scale*scale*scale;
        const std::vector<Real> target={-.04*scale*scale*scale,-.1*scale*scale*scale};
        HostMesh gas,solid,measured; SurfaceMesh surface; SweepConstraintReport report; std::vector<Real> swept;
        require(moveCoupledMeshesConstrained(base,base.filmAux,target,1,gas,solid,surface,report,error,controls),
            "rotated/rescaled 3-D constraint solve: "+error);
        require(makeStageGeometry(base.solidMesh,solid.points,1,measured,swept,error),error);
        for (int i=0;i<2;++i)
            require(absValue(swept[base.surface.solidFace[i]]-target[i])<1e-11*scale*scale*scale,
                "rotated/rescaled actual sweep target");
    }
}
void planarWetMotionStillSupported() {
    HostState base=coupledPair();
    for (int p=0;p<6;++p) base.gasMesh.points[p].z+=.02;
    for (auto& a:base.filmAux) a.thickness=.02;
    std::string error; require(rebuildGeometry(base.gasMesh,error),error);
    HostMesh gas,solid; SurfaceMesh surface; SweepConstraintReport report;
    require(moveCoupledMeshesConstrained(base,base.filmAux,{-.03,-.03},1,gas,solid,surface,report,error),error);
    for (int i=0;i<2;++i)
        require(absValue(mag(gas.areaVectors[base.surface.gasFace[i]])-surface.area[i])<1e-12,
            "planar conformal wet area retained");
}
void boundedSolveLimitAndNonfiniteInput() {
    const HostState base=coupledPair(); HostMesh gas=base.gasMesh,solid=base.solidMesh;
    SurfaceMesh surface=base.surface; SweepConstraintControls controls; controls.maximumDegreesOfFreedom=1;
    SweepConstraintReport report; std::string error;
    require(!moveCoupledMeshesConstrained(base,base.filmAux,{-.04,-.1},1,gas,solid,surface,report,error,controls),
        "bounded dense solve must reject oversized correction");
    require(report.status==SweepConstraintStatus::SizeLimit,"size-limit diagnostic");
    require(!moveCoupledMeshesConstrained(base,base.filmAux,{undefinedValue(),0},1,gas,solid,surface,report,error),
        "nonfinite target rejected");
    require(report.status==SweepConstraintStatus::InvalidInput,"nonfinite target diagnostic");
}

void stationaryCurvedOffsetsPreserveTranslatedPoints() {
    for (Real shift:std::vector<Real>{1,10,1000}) {
        auto base=curvedOffsetPair();
        for (auto& p:base.solidMesh.points) p+=Vec3{shift,shift,shift};
        for (auto& p:base.gasMesh.points) p+=Vec3{shift,shift,shift};
        std::string error; require(rebuildGeometry(base.solidMesh,error),error);
        require(rebuildGeometry(base.gasMesh,error),error);
        HostMesh gas,solid; SurfaceMesh surface; SweepConstraintReport report;
        const bool ok=moveCoupledMeshesConstrained(base,base.filmAux,{0,0},1,gas,solid,surface,report,error);
        require(ok,"stationary translated curved offset: "+error);
        for (std::size_t p=0;p<gas.points.size();++p)
            require(mag(gas.points[p]-base.gasMesh.points[p])==0,"zero target preserves accepted gas coordinates exactly");
        for (std::size_t p=0;p<solid.points.size();++p)
            require(mag(solid.points[p]-base.solidMesh.points[p])==0,"zero target preserves accepted solid coordinates exactly");
    }
}
void translatedRotatedNonuniformDryMotion() {
    int referenceConstraints=-1;
    for (Real shift:std::vector<Real>{0,200,500,1000,2000}) {
        auto base=coupledPair();
        auto transform=[&](Vec3 x) {
            Vec3 y={.8*x.x-.6*x.y,.6*x.x+.8*x.y,x.z};
            return Vec3{.6*y.x+.8*y.z,y.y,-.8*y.x+.6*y.z}+Vec3{shift,shift,shift};
        };
        for (auto& p:base.solidMesh.points) p=transform(p);
        for (auto& p:base.gasMesh.points) p=transform(p);
        std::string error; require(rebuildGeometry(base.solidMesh,error),error);
        require(rebuildGeometry(base.gasMesh,error),error);
        HostMesh gas,solid,measured; SurfaceMesh surface; SweepConstraintReport report;
        const std::vector<Real> targets={-.04,-.1};
        const bool ok=moveCoupledMeshesConstrained(base,base.filmAux,targets,1,gas,solid,surface,report,error);
        require(ok,"translated rotated nonuniform dry motion: "+error);
        if (referenceConstraints<0) referenceConstraints=report.constraints;
        require(report.constraints==referenceConstraints,"translation preserves planar constraint classification");
        std::vector<Real> solidSweeps,gasSweeps;
        require(makeStageGeometry(base.solidMesh,solid.points,1,measured,solidSweeps,error),error);
        require(makeStageGeometry(base.gasMesh,gas.points,1,measured,gasSweeps,error),error);
        for (int i=0;i<2;++i) {
            require(absValue(solidSweeps[base.surface.solidFace[i]]-targets[i])<1e-11,
                "translated actual solid sweeps meet physical target");
            require(absValue(gasSweeps[base.surface.gasFace[i]]+targets[i])<1e-11,
                "translated actual gas sweeps meet opposite physical target");
        }
        require(maximumGclResidual(base.solidMesh,solid,solidSweeps)<1e-11,"translated independent solid GCL");
        require(maximumGclResidual(base.gasMesh,gas,gasSweeps)<1e-11,"translated independent gas GCL");
    }
}

void rejectRoundedTrajectoryResidual() {
    auto base=coupledPair();
    for (auto& p:base.solidMesh.points) p+=Vec3{1e6,1e6,1e6};
    for (auto& p:base.gasMesh.points) p+=Vec3{1e6,1e6,1e6};
    std::string error; require(rebuildGeometry(base.solidMesh,error),error);
    require(rebuildGeometry(base.gasMesh,error),error);
    HostMesh gas=base.gasMesh,solid=base.solidMesh; SurfaceMesh surface=base.surface;
    gas.geometryVersion=77; solid.geometryVersion=88; SweepConstraintReport report;
    const bool ok=moveCoupledMeshesConstrained(base,base.filmAux,{-.04,-.1},1,gas,solid,surface,report,error);
    require(!ok,"local-frame convergence cannot substitute for actual world-coordinate target sweeps");
    require(error.find("actual target swept volume")!=std::string::npos,"rounded trajectory residual diagnosis: "+error);
    require(gas.geometryVersion==77 && solid.geometryVersion==88,"rounded-trajectory rejection preserves outputs");
}

void denseSmallCellSweepAccuracy() {
    for (Real scale:std::vector<Real>{1e-4,1e-3,1}) for (Real offset:std::vector<Real>{0,.25}) {
        auto base=coupledPair();
        for (auto& point:base.solidMesh.points) point=(point+Vec3{offset,offset,offset})*scale;
        for (auto& point:base.gasMesh.points) point=(point+Vec3{offset,offset,offset})*scale;
        std::string error; require(rebuildGeometry(base.solidMesh,error),error);
        require(rebuildGeometry(base.gasMesh,error),error);
        base.solid.resize(base.solidMesh.volumes.size());
        for (std::size_t c=0;c<base.solid.size();++c)base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];
        auto candidate=base.filmAux;candidate[0].solidFront=-.03e-8*scale;candidate[1].solidFront=-.12e-8*scale;
        const std::vector<Real> targets={-.04e-8*scale*scale*scale,-.1e-8*scale*scale*scale};
        HostMesh gas,solid,measured;SurfaceMesh surface;SweepConstraintReport report;std::vector<Real> sweeps;
        require(moveCoupledMeshesConstrained(base,candidate,targets,1,gas,solid,surface,report,error),
            "dense small-cell constrained motion: "+error);
        require(makeStageGeometry(base.solidMesh,solid.points,1,measured,sweeps,error),error);
        for (std::size_t f=0;f<targets.size();++f) {
            const Real allowed=8*std::numeric_limits<Real>::epsilon()*base.solidMesh.volumes[f];
            std::ostringstream detail;detail<<"dense small-cell sweep exceeds inventory budget scale="<<scale
                <<" offset="<<offset<<" residual="<<absValue(sweeps[base.surface.solidFace[f]]-targets[f])<<" allowed="<<allowed;
            require(absValue(sweeps[base.surface.solidFace[f]]-targets[f])<=allowed,detail.str());
            require(base.solid[f].condensed[0]==1000*base.solidMesh.volumes[f],"geometry may not repair physical mass");
        }
    }
}

void hardSweepCeilingsValidateAndRejectUnrepresentableMotion() {
    auto base=coupledPair();base.solid.resize(base.solidMesh.volumes.size());
    for (const auto& limits:std::vector<std::vector<Real>>{{1},{0,1},{-1,1},{undefinedValue(),1},
        {std::numeric_limits<Real>::infinity(),1}}) {
        HostMesh gas=base.gasMesh,solid=base.solidMesh;gas.geometryVersion=77;solid.geometryVersion=88;
        SurfaceMesh surface;SweepConstraintControls controls;controls.faceVolumeTolerance=limits;
        SweepConstraintReport report;std::string error;
        require(!moveCoupledMeshesConstrained(base,base.filmAux,{-.04,-.1},1,gas,solid,surface,report,error,controls),
            "invalid per-face material ceiling must fail");
        require(report.status==SweepConstraintStatus::InvalidInput,"invalid ceiling status");
        require(gas.geometryVersion==77&&solid.geometryVersion==88,"invalid ceilings preserve outputs");
    }
    for (auto& p:base.solidMesh.points)p+=Vec3{1000,1000,1000};
    for (auto& p:base.gasMesh.points)p+=Vec3{1000,1000,1000};
    std::string error;require(rebuildGeometry(base.solidMesh,error),error);require(rebuildGeometry(base.gasMesh,error),error);
    HostMesh gas=base.gasMesh,solid=base.solidMesh;gas.geometryVersion=77;solid.geometryVersion=88;
    SurfaceMesh surface;SweepConstraintReport report;
    require(!moveCoupledMeshesConstrained(base,base.filmAux,{-.04,-.1},1,gas,solid,surface,report,error),
        "world-coordinate quantization may not override the material inventory ceiling");
    require(report.status==SweepConstraintStatus::InvalidTrajectory||report.status==SweepConstraintStatus::Nonconverged,
        "unrepresentable inventory accuracy has a typed rejection");
    require(gas.geometryVersion==77&&solid.geometryVersion==88,"unrepresentable material motion preserves outputs");
}

}
int main() {
    hardSweepCeilingsValidateAndRejectUnrepresentableMotion();
    denseSmallCellSweepAccuracy();
    rejectRoundedTrajectoryResidual();
    stationaryCurvedOffsetsPreserveTranslatedPoints();
    translatedRotatedNonuniformDryMotion();
    actualNonuniformDrySweep();
    explicitIntegratedTargets();
    rejectIncompatibleTargets();
    stationaryNeedsNoDenseSolve();
    nonlinearThreeDimensionalRecession();
    prescribedTrajectoryGclAndComposition();
    rejectInvalidTrajectory();
    rebuildActualTrajectorySurface();
    rejectInteriorBoundaryDof();
    retainMovingOffsetRejection();
    scaledRotatedDryMotion();
    planarWetMotionStillSupported();
    boundedSolveLimitAndNonfiniteInput();
    std::cout << "PASS: actual 3-D constrained point motion and independent GCL\n";
    return 0;
}
