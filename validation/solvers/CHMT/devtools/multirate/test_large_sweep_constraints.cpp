// Reuse the production-entry fixtures and independent quadrature without
// maintaining a second copy of the existing small-geometry regression suite.
#define main inheritedSweepRegressionMain
#include "test_sweep_constraints.cpp"
#undef main
#include "mesh/SweepWorkspace.H"
#include <cstring>
#include "DiagnosticsHost.H"

namespace {
struct LargeFixture {
    HostMesh mesh;
    std::vector<int> faces;
    std::vector<Real> targets;
    std::vector<Vec3> zero,directions;
    std::vector<unsigned char> fixed;
};

// Disconnected valid finite-volume blocks isolate each former dense cap.
// The targets come from a real harmonic trajectory, not fabricated equations.
LargeFixture largeFixture(bool tetra,int blocks,int activeBlocks) {
    const HostMesh unit=tetra?tetrahedralBulk():twoHexahedra(0);
    LargeFixture x; x.mesh.faceOffsets.push_back(0);
    std::vector<Vec3> displacement;
    for (int block=0;block<blocks;++block) {
        const int pointBase=static_cast<int>(x.mesh.points.size());
        const int cellBase=block*static_cast<int>(unit.volumes.size());
        for (std::size_t p=0;p<unit.points.size();++p) {
            Vec3 point=unit.points[p]; point.x+=4*block;
            x.mesh.points.push_back(point); x.fixed.push_back(!tetra || p!=4);
            const bool active=block<activeBlocks && (tetra?p!=4:point.z==1);
            x.directions.push_back(active?Vec3{0,0,1}:Vec3{});
            displacement.push_back(active?Vec3{0,0,-.01}:Vec3{});
        }
        for (std::size_t f=0;f<unit.owner.size();++f) {
            const int face=static_cast<int>(x.mesh.owner.size());
            x.mesh.owner.push_back(cellBase+unit.owner[f]);
            x.mesh.neighbour.push_back(unit.neighbour[f]<0?-1:cellBase+unit.neighbour[f]);
            x.mesh.boundaryKind.push_back(unit.boundaryKind[f]);
            for (int k=unit.faceOffsets[f];k<unit.faceOffsets[f+1];++k)
                x.mesh.facePoints.push_back(pointBase+unit.facePoints[k]);
            x.mesh.faceOffsets.push_back(static_cast<int>(x.mesh.facePoints.size()));
            if (unit.neighbour[f]<0 && (tetra || (unit.faceCentres[f].z==1 && unit.areaVectors[f].z>0)))
                x.faces.push_back(face);
        }
    }
    std::string error; require(rebuildGeometry(x.mesh,error),error);
    std::vector<Vec3> known; require(harmonicPointMotion(x.mesh,displacement,x.fixed,known,error),error);
    // Five-point quadrature is independent of the production sweep polynomial.
    for (int face:x.faces) x.targets.push_back(quadratureSweep(x.mesh,known,face));
    x.zero.resize(x.mesh.points.size());
    return x;
}

struct SolveResult {
    bool accepted=false, warned=false;
    HostMesh output;
    std::vector<Real> sweeps;
    SweepConstraintReport report;
    std::string error;
};

SolveResult solve(const LargeFixture& x,const SweepConstraintControls& controls) {
    SolveResult r; r.output=x.mesh; r.output.geometryVersion=991;
    r.sweeps={123,456};
    ugkp::diagnostics::DeviceContext diagnostics{};ugkp::diagnostics::reset(diagnostics,1,34);
    Tolerances tolerances;tolerances.diagnostics=&diagnostics;
    r.accepted=constrainFaceSweeps(x.mesh,x.faces,x.targets,x.zero,x.fixed,x.directions,
        1,r.output,r.sweeps,r.report,r.error,controls,tolerances);
    r.warned=!ugkp::diagnostics::takeSnapshot(diagnostics).warnings.empty();
    return r;
}

void checkRejectedOutputs(const LargeFixture& x,const SolveResult& result) {
    require(result.output.geometryVersion==991 && result.output.points.size()==x.mesh.points.size()
        && result.sweeps==std::vector<Real>({123,456}),"rejected correction preserves output sentinels");
    for (std::size_t p=0;p<x.mesh.points.size();++p)
        require(mag(result.output.points[p]-x.mesh.points[p])==0,"rejected correction preserves all points");
    require(result.output.volumes==x.mesh.volumes && result.output.owner==x.mesh.owner
        && result.output.neighbour==x.mesh.neighbour && result.output.facePoints==x.mesh.facePoints,
        "rejected correction preserves mesh geometry/topology");
}

// Check all three nonconstant coefficients of the trajectory's signed
// tetrahedron volume. Endpoint-only planarity could miss cancellations.
void checkPlanarTrajectory(const HostMesh& old,const HostMesh& endpoint) {
    for (std::size_t f=0;f<old.owner.size();++f) {
        const int begin=old.faceOffsets[f],end=old.faceOffsets[f+1];
        const int a=old.facePoints[begin],b=old.facePoints[begin+1],c=old.facePoints[begin+2];
        const Vec3 ab=old.points[b]-old.points[a],ac=old.points[c]-old.points[a];
        const Vec3 da=endpoint.points[a]-old.points[a];
        const Vec3 db=endpoint.points[b]-old.points[b]-da,dc=endpoint.points[c]-old.points[c]-da;
        for (int k=begin+3;k<end;++k) {
            const int d=old.facePoints[k];
            const Vec3 ad=old.points[d]-old.points[a],dd=endpoint.points[d]-old.points[d]-da;
            const Real scale=mag(ab)*mag(ac)*mag(ad);
            const Real coefficient[3]={
                dot(db,cross(ac,ad))+dot(ab,cross(dc,ad))+dot(ab,cross(ac,dd)),
                dot(db,cross(dc,ad))+dot(db,cross(ac,dd))+dot(ab,cross(dc,dd)),
                dot(db,cross(dc,dd))};
            for (Real value:coefficient)
                require(absValue(value)<256*std::numeric_limits<Real>::epsilon()*scale,
                    "all three original trajectory planarity coefficients");
        }
    }
}

void checkPhysicalResult(const LargeFixture& x,const SolveResult& result,bool targetsMet=true) {
    require(result.accepted && result.report.status==SweepConstraintStatus::Success,
        "real geometry correction must succeed: "+result.error);
    require(result.error.empty(),"successful correction clears diagnostic");
    require(result.report.iterations>0 && result.report.rank>0,"actual Newton/QR correction ran");
    require(result.output.points.size()==x.mesh.points.size() && result.sweeps.size()==x.mesh.owner.size(),
        "complete geometry and all constrained/unconstrained sweeps returned");
    require(result.output.owner==x.mesh.owner && result.output.neighbour==x.mesh.neighbour
        && result.output.faceOffsets==x.mesh.faceOffsets && result.output.facePoints==x.mesh.facePoints,
        "correction preserves mesh topology");
    if(targetsMet)for (std::size_t i=0;i<x.faces.size();++i)
        require(absValue(result.sweeps[x.faces[i]]-x.targets[i])<1e-12,"all original target sweeps retained");
    for (std::size_t f=0;f<result.sweeps.size();++f)
        require(absValue(result.sweeps[f]-quadratureSweep(x.mesh,result.output.points,static_cast<int>(f)))<1e-12,
            "all face sweeps agree with independent quadrature");
    for (Real volume:result.output.volumes) require(chmt::finite(volume) && volume>0,"positive endpoint cell volumes");
    require(maximumGclResidual(x.mesh,result.output,result.sweeps)<1e-12,"independent large-geometry GCL");
    checkPlanarTrajectory(x.mesh,result.output);
}

void compareResults(const LargeFixture& x,const SolveResult& reference,const SolveResult& candidate,bool targetsMet=true) {
    require(reference.accepted==candidate.accepted && reference.report.status==candidate.report.status,
        "cached/reference acceptance and rejection classification agree");
    require(reference.report.rank==candidate.report.rank
        && reference.report.constraints==candidate.report.constraints
        && reference.report.degreesOfFreedom==candidate.report.degreesOfFreedom,
        "original numerical rank and full equation set are retained");
    if (!candidate.accepted) {
        checkRejectedOutputs(x,reference); checkRejectedOutputs(x,candidate); return;
    }
    checkPhysicalResult(x,reference,targetsMet); checkPhysicalResult(x,candidate,targetsMet);
    require(reference.output.geometryVersion==candidate.output.geometryVersion,"output geometry version agrees");
    for (std::size_t p=0;p<x.mesh.points.size();++p)
        require(mag(reference.output.points[p]-candidate.output.points[p])<2e-11,"weighted physical endpoint agrees");
    for (std::size_t f=0;f<reference.sweeps.size();++f)
        require(absValue(reference.sweeps[f]-candidate.sweeps[f])<2e-12,"all cached/reference face sweeps agree");
}

void checkWorkspaceReport(const SolveResult& result,const SweepConstraintControls&) {
    require(result.report.peakWorkspaceBytes>0,
        "actual solver workspace is accounted without a policy ceiling");
    const std::size_t dof=static_cast<std::size_t>(result.report.degreesOfFreedom);
    if (result.accepted) {
        const std::size_t expected=result.report.streamedModes?2*result.report.iterations*dof:dof;
        require(result.report.modeSolves==expected,"mode solve count exposes cached/streamed Newton recomputation");
    }
}

void checkLarge(const std::string& name) {
    const bool tetra=name!="hex130";
    const bool stress=name=="tet4500";
    const int blocks=stress?4500:name=="tet880"?880:130;
    const int active=name=="hex130"?1:(name=="tet880" || stress)?120:130;
    const LargeFixture x=largeFixture(tetra,blocks,active);
    const SolveResult candidate=solve(x,SweepConstraintControls{});
    std::cout << "case=" << name << " status=" << static_cast<int>(candidate.report.status)
        << " constraints=" << candidate.report.constraints << " dof=" << candidate.report.degreesOfFreedom
        << " mode_entries=" << x.mesh.points.size()*candidate.report.degreesOfFreedom
        << " rank=" << candidate.report.rank << " iterations=" << candidate.report.iterations << '\n';
    require(candidate.report.constraints==(stress?18000:name=="hex130"?4550:name=="tet130"?520:3520),
        "fixture retains its original complete constraint count");
    require(candidate.report.degreesOfFreedom==(name=="hex130"?6:name=="tet130"?520:480),
        "fixture retains its independent boundary DOF count");
    require(name!="tet880" || x.mesh.points.size()*candidate.report.degreesOfFreedom==2112000,
        "point-mode fixture independently exceeds the former two-million-entry cap");
    require(candidate.accepted,"default production entry must solve "+name+": "+candidate.error);
    require(candidate.report.streamedModes,"one streamed construction for every mesh size");
    checkWorkspaceReport(candidate,SweepConstraintControls{});
    if (stress) {
        checkPhysicalResult(x,candidate);
        require(candidate.report.rank==480,"default-budget stress fixture retains original full DOF rank");
        std::cout << "tet4500 default_streamed_peak_bytes=" << candidate.report.peakWorkspaceBytes
            << " mode_solves=" << candidate.report.modeSolves << '\n';
        return;
    }
    SweepConstraintControls referenceControls;
    referenceControls.maximumDegreesOfFreedom=1000000;
    referenceControls.maximumConstraints=1000000;
    referenceControls.maximumModeEntries=1000000000;
    const SolveResult reference=solve(x,referenceControls);
    compareResults(x,reference,candidate);
    require(candidate.report.rank==(name=="hex130"?4:name=="tet130"?520:480),"original QR rank for large fixture");
    if (name=="tet880") {
        SweepConstraintControls bounded; bounded.maximumWorkspaceBytes=32ull*1024*1024;
        const SolveResult streamed=solve(x,bounded);
        compareResults(x,reference,streamed);
        require(streamed.report.streamedModes,"32-MiB real large fixture regenerates modes instead of rejecting");
        checkWorkspaceReport(streamed,bounded);
        std::cout << "tet880 cached_peak_bytes=" << reference.report.peakWorkspaceBytes
            << " streamed_peak_bytes=" << streamed.report.peakWorkspaceBytes
            << " cached_mode_solves=" << reference.report.modeSolves
            << " streamed_mode_solves=" << streamed.report.modeSolves << '\n';
    }
}

LargeFixture nearRankFixture(Real sigma,Real secondTarget) {
    HostMesh unit=tetrahedralBulk();
    unit.points={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{.25,.25,.25}};
    std::string error; require(rebuildGeometry(unit,error),error);
    LargeFixture x; x.mesh.faceOffsets={0};
    for (int block=0;block<2;++block) {
        for (int p=0;p<4;++p) x.mesh.points.push_back(unit.points[p]+Vec3{Real(4*block),0,0});
        for (std::size_t f=0;f<unit.owner.size();++f) if (unit.neighbour[f]<0) {
            x.mesh.owner.push_back(block); x.mesh.neighbour.push_back(-1);
            x.mesh.boundaryKind.push_back(BoundaryKind::Slip);
            for (int k=unit.faceOffsets[f];k<unit.faceOffsets[f+1];++k)
                x.mesh.facePoints.push_back(4*block+unit.facePoints[k]);
            x.mesh.faceOffsets.push_back(static_cast<int>(x.mesh.facePoints.size()));
        }
    }
    require(rebuildGeometry(x.mesh,error),error);
    for (std::size_t f=0;f<x.mesh.owner.size();++f) if (x.mesh.faceCentres[f].z==0)
        x.faces.push_back(static_cast<int>(f));
    x.zero.resize(x.mesh.points.size()); x.directions.resize(x.mesh.points.size());
    x.fixed.assign(x.mesh.points.size(),1);
    for (std::size_t p=0;p<x.mesh.points.size();++p) if (x.mesh.points[p].z==0)
        x.directions[p]=p<4?Vec3{0,0,1}:Vec3{1,0,sigma};
    x.targets={-1e-6,-secondTarget}; return x;
}

void checkNearRank() {
    for (Real sigma:{1e-10,1e-11,1e-12}) for (Real target:{1e-12,1e-10,1e-8}) {
        const LargeFixture x=nearRankFixture(sigma,target);
        const SolveResult result=solve(x,SweepConstraintControls{});
        const bool expected=sigma==1e-10;
        require(result.accepted,"finite near-rank residual retains actual geometry: "+result.error);
        require(result.report.rank==(expected?2:1),"two-tetra original QR rank");
        checkPhysicalResult(x,result,expected);
        if(!expected)require(result.warned,"rank-deficient finite residual is diagnosed");
        require(result.report.streamedModes,"near-rank uses deterministic streamed modes");
        require(result.report.peakWorkspaceBytes>1,"reference reports actual workspace");
        SweepConstraintControls bounded; bounded.maximumWorkspaceBytes=result.report.peakWorkspaceBytes-1;
        const SolveResult streamed=solve(x,bounded);
        compareResults(x,result,streamed,expected);
        require(streamed.report.streamedModes,"near-rank real production fixture exercises regenerated modes");
        checkWorkspaceReport(streamed,bounded);
        std::cout << "near_rank sigma=" << sigma << " target2=" << target
            << " accepted=" << result.accepted << " rank=" << result.report.rank << '\n';
    }
}

void checkNonlinear() {
    LargeFixture x; x.mesh=tetrahedralBulk();
    for (std::size_t f=0;f<x.mesh.owner.size();++f) if (x.mesh.neighbour[f]<0)
        x.faces.push_back(static_cast<int>(f));
    x.zero.resize(x.mesh.points.size()); x.directions.resize(x.mesh.points.size());
    x.fixed.assign(x.mesh.points.size(),1); x.fixed[4]=0;
    std::vector<Vec3> displacement(x.mesh.points.size());
    const Real amplitudes[4]={-.019,-.041,-.032,-.027};
    for (int p=0;p<4;++p) {
        x.directions[p]=normalized(x.mesh.points[p]-x.mesh.points[4]);
        displacement[p]=x.directions[p]*amplitudes[p];
    }
    std::vector<Vec3> known; std::string error;
    require(harmonicPointMotion(x.mesh,displacement,x.fixed,known,error),error);
    for (int face:x.faces) x.targets.push_back(quadratureSweep(x.mesh,known,face));
    const SolveResult cached=solve(x,SweepConstraintControls{});
    checkPhysicalResult(x,cached); checkWorkspaceReport(cached,SweepConstraintControls{});
    require(cached.report.iterations>=2 && cached.report.streamedModes,"nonlinear streamed solve needs repeated Newton");
    SweepConstraintControls bounded; bounded.maximumWorkspaceBytes=cached.report.peakWorkspaceBytes-1;
    const SolveResult streamed=solve(x,bounded);
    compareResults(x,cached,streamed); checkWorkspaceReport(streamed,bounded);
    require(streamed.report.streamedModes,"nonlinear fixture regenerates each Newton's modes");
    require(streamed.report.modeSolves==cached.report.modeSolves,"legacy caps cannot alter mode construction or work");
}

void checkControls() {
    const LargeFixture x=largeFixture(true,1,1);
    const SweepConstraintControls defaults;
    require(defaults.maximumDegreesOfFreedom==0 && defaults.maximumConstraints==0
        && defaults.maximumModeEntries==0,"default legacy dimension caps are resource-budget driven");
    require(defaults.maximumWorkspaceBytes==0,"no default workspace policy cap");
    const SolveResult cached=solve(x,defaults);
    checkPhysicalResult(x,cached); checkWorkspaceReport(cached,defaults);
    const auto continueWith=[&](const SweepConstraintControls& controls) {
        const SolveResult result=solve(x,controls);
        compareResults(x,cached,result);
        require(result.report.modeSolves==cached.report.modeSolves,"legacy caps do not alter solve work");
    };
    SweepConstraintControls controls=defaults; controls.maximumDegreesOfFreedom=1;continueWith(controls);
    controls=defaults; controls.maximumConstraints=1;continueWith(controls);
    controls=defaults; controls.maximumModeEntries=1;continueWith(controls);
    controls=defaults; controls.maximumWorkspaceBytes=1;continueWith(controls);
    controls=defaults; controls.maximumWorkspaceBytes=0;continueWith(controls);
    controls=defaults; controls.maximumDegreesOfFreedom=-1;continueWith(controls);
    controls=defaults; controls.maximumConstraints=-1;continueWith(controls);
    LargeFixture invalid=x;invalid.faces[0]=static_cast<int>(x.mesh.owner.size());
    const SolveResult badIndex=solve(invalid,defaults);
    require(!badIndex.accepted && badIndex.report.status==SweepConstraintStatus::InvalidInput,
        "actual out-of-bounds face remains error");
    checkRejectedOutputs(invalid,badIndex);
    LargeFixture stationary=x; stationary.targets.assign(x.faces.size(),0);
    controls=defaults; controls.maximumDegreesOfFreedom=1; controls.maximumConstraints=1;
    controls.maximumModeEntries=1; controls.maximumWorkspaceBytes=1;
    const SolveResult unchanged=solve(stationary,controls);
    require(unchanged.accepted && unchanged.report.status==SweepConstraintStatus::Success
        && unchanged.report.iterations==0 && unchanged.report.modeSolves==0,
        "an already satisfied trajectory bypasses correction-only budgets and caps");
}

void checkOriginalTrajectoryPolynomials() {
    LargeFixture x; x.mesh=twoHexahedra(0);
    x.fixed.assign(x.mesh.points.size(),1); x.directions.resize(x.mesh.points.size());
    std::vector<Vec3> endpoint;
    // A projective map leaves every endpoint face planar and convex, but the
    // straight vertex paths between old and new faces are generally not planar.
    for (Vec3 point:x.mesh.points) {
        const Vec3 next=point/(1+dot(Vec3{.09,.13,.17},point));
        endpoint.push_back(next); x.zero.push_back(next-point);
    }
    HostMesh validEndpoint=x.mesh; validEndpoint.points=endpoint;
    std::string error; require(rebuildGeometry(validEndpoint,error),"projective endpoint is valid: "+error);
    bool exercisedAll=false;
    for (std::size_t f=0;f<x.mesh.owner.size();++f) {
        if (x.mesh.neighbour[f]<0) {
            x.faces.push_back(static_cast<int>(f));
            x.targets.push_back(quadratureSweep(x.mesh,endpoint,static_cast<int>(f)));
        }
        const int begin=x.mesh.faceOffsets[f];
        const int a=x.mesh.facePoints[begin],b=x.mesh.facePoints[begin+1];
        const int c=x.mesh.facePoints[begin+2],d=x.mesh.facePoints[begin+3];
        const Vec3 u=x.mesh.points[b]-x.mesh.points[a],v=x.mesh.points[c]-x.mesh.points[a];
        const Vec3 w=x.mesh.points[d]-x.mesh.points[a],du=x.zero[b]-x.zero[a];
        const Vec3 dv=x.zero[c]-x.zero[a],dw=x.zero[d]-x.zero[a];
        const Real c1=dot(du,cross(v,w))+dot(u,cross(dv,w))+dot(u,cross(v,dw));
        const Real c2=dot(du,cross(dv,w))+dot(du,cross(v,dw))+dot(u,cross(dv,dw));
        const Real c3=dot(du,cross(dv,dw));
        require(absValue(c1+c2+c3)<1e-12,"planarity terms cancel at the valid endpoint");
        exercisedAll=exercisedAll || (absValue(c1)>1e-6 && absValue(c2)>1e-6 && absValue(c3)>1e-6);
    }
    require(exercisedAll,"fixture exercises all three nonzero original trajectory coefficients");
    const SolveResult result=solve(x,SweepConstraintControls{});
    require(result.accepted && result.report.status==SweepConstraintStatus::Success
        && result.report.rank==0 && result.report.constraints==43 && result.warned,
        "finite frozen planarity residual warns with all three equations retained");
    for(std::size_t p=0;p<endpoint.size();++p)
        require(mag(result.output.points[p]-endpoint[p])<1e-14,"frozen finite endpoint is retained");
    for(std::size_t f=0;f<result.sweeps.size();++f)
        require(absValue(result.sweeps[f]-quadratureSweep(x.mesh,endpoint,static_cast<int>(f)))<1e-12,
            "finite planarity warning preserves independently integrated sweep");

}

bool samePointBits(const Vec3& a,const Vec3& b) {
    return std::memcmp(&a.x,&b.x,sizeof(Real))==0
        && std::memcmp(&a.y,&b.y,sizeof(Real))==0
        && std::memcmp(&a.z,&b.z,sizeof(Real))==0;
}

void checkHarmonicOutputRounding() {
    HostMesh mesh=tetrahedralBulk();
    // Preserve the existing graph, but make sub-ULP motion disappear in the
    // required (old + displacement) - old arithmetic at the interior point.
    const Real offset=std::ldexp(Real(1),40);
    for (Vec3& point:mesh.points) point+=Vec3{offset,-offset,offset};
    std::vector<unsigned char> fixed(mesh.points.size(),1); fixed[4]=0;
    std::size_t peak=0; sweep_detail::Budget budget(1024*1024,peak);
    sweep_detail::HarmonicModes modes(mesh,fixed,budget,1e-12);
    sweep_detail::Points boundary(mesh.points.size(),Vec3{},sweep_detail::Allocator<Vec3>(budget));
    sweep_detail::Points actual{ sweep_detail::Allocator<Vec3>(budget) };
    for (Real amplitude:{1e-7,.03}) {
        std::vector<Vec3> prescribed(mesh.points.size());
        for (int p=0;p<4;++p)
            prescribed[p]=boundary[p]=Vec3{.7,-.3,.2}*(amplitude*(p+1));
        prescribed[0].x=boundary[0].x=-Real(0);
        prescribed[1].z=boundary[1].z=-Real(0);
        std::vector<Vec3> expected; std::string error;
        require(harmonicPointMotion(mesh,prescribed,fixed,expected,error),error);
        // Keep the previous separate subtraction and fixed-overwrite passes
        // as the independent reference for the effective output mode.
        for (std::size_t p=0;p<expected.size();++p) expected[p]-=mesh.points[p];
        for (std::size_t p=0;p<expected.size();++p) if(fixed[p]) expected[p]=prescribed[p];
        require(modes.solve(boundary,actual),"rounding-sensitive harmonic graph converges");
        require(actual.size()==expected.size(),"rounding-sensitive output contains every point");
        for (std::size_t p=0;p<expected.size();++p) {
            require(samePointBits(actual[p],expected[p]),"effective output is bitwise equal to the old arithmetic");
            if(fixed[p]) require(samePointBits(actual[p],boundary[p]),"fixed overwrite preserves boundary bits, including negative zero");
        }
        Vec3 weighted{}; Real total=0;
        for (int p=0;p<4;++p) {
            const Real weight=1/mag(mesh.points[4]-mesh.points[p]);
            weighted+=boundary[p]*weight; total+=weight;
        }
        const Vec3 directDisplacement=weighted/total;
        require(samePointBits(actual[4],(mesh.points[4]+directDisplacement)-mesh.points[4]),
            "free point retains ordered add-then-subtract rounding");
        require(!samePointBits(actual[4],directDisplacement),"fixture detects replacing output by direct displacement");
        if(amplitude==1e-7) require(mag(actual[4])==0 && mag(directDisplacement)>0,
            "sub-ULP interior displacement cancels only through the old arithmetic");
    }
    std::cout << "PASS: harmonic output bitwise old arithmetic, cancellation and fixed signed zeros\n";
}

void checkHarmonicModes() {
    HostMesh mesh=tetrahedralBulk();
    for (Vec3& point:mesh.points) point={1.7*point.x+.13*point.y,.63*point.y+.21*point.z,1.2*point.z};
    std::string error; require(rebuildGeometry(mesh,error),error);
    std::vector<unsigned char> fixed(mesh.points.size(),1); fixed[4]=0;
    Vec3 low=mesh.points.front(),high=low;
    for (Vec3 point:mesh.points) {
        low.x=std::min(low.x,point.x); low.y=std::min(low.y,point.y); low.z=std::min(low.z,point.z);
        high.x=std::max(high.x,point.x); high.y=std::max(high.y,point.y); high.z=std::max(high.z,point.z);
    }
    std::size_t peak=0; sweep_detail::Budget budget(1024*1024,peak);
    {
        sweep_detail::HarmonicModes modes(mesh,fixed,budget,1e-12*mag(high-low));
        sweep_detail::Points boundary(mesh.points.size(),Vec3{},sweep_detail::Allocator<Vec3>(budget));
        sweep_detail::Points actual{ sweep_detail::Allocator<Vec3>(budget) };
        for (Real amplitude:{1e-14,1e-6,.03}) for (int active=0;active<5;++active) {
            std::fill(boundary.begin(),boundary.end(),Vec3{});
            std::vector<Vec3> prescribed(mesh.points.size());
            for (int p=0;p<4;++p) if (p==active || active==4)
                prescribed[p]=boundary[p]=Vec3{.7,-.3,.2}*(amplitude*(p+1));
            std::vector<Vec3> expected;
            require(harmonicPointMotion(mesh,prescribed,fixed,expected,error),error);
            for (std::size_t p=0;p<mesh.points.size();++p) {
                expected[p]-=mesh.points[p];
                if (fixed[p]) expected[p]=prescribed[p];
            }
            require(modes.solve(boundary,actual),"cached harmonic graph converges");
            require(actual.size()==expected.size(),"cached harmonic mode contains every point");
            for (std::size_t p=0;p<expected.size();++p)
                require(samePointBits(actual[p],expected[p]),
                    "sorted cached graph preserves the full public harmonic mode at every tested amplitude");
        }
        require(budget.live>0 && peak>=budget.live,"harmonic graph and scratch allocations are accounted");
    }
    require(budget.live==0,"destroyed harmonic cache and modes return all accounted storage");
}

void checkWorkspaceAllocator() {
    std::size_t peak=0; sweep_detail::Budget tiny(1,peak);
    bool rejected=false;
    {sweep_detail::Points point(1,Vec3{},sweep_detail::Allocator<Vec3>(tiny));
     require(tiny.live==sizeof(Vec3) && peak==sizeof(Vec3),"tiny legacy budget retains actual allocation");}
    require(tiny.live==0,"tiny allocation released");
    peak=0;sweep_detail::Budget budget(2*sizeof(Real),peak);
    {
        sweep_detail::Reals values(2,Real(7),sweep_detail::Allocator<Real>(budget));
        require(budget.live==2*sizeof(Real) && peak==budget.live,"live payload bytes and peak are recorded");
        values.reserve(3);
        require(values.size()==2 && values[0]==7 && values[1]==7 && budget.live>=3*sizeof(Real),
            "growth beyond policy cap retains original data");
        require(peak>=5*sizeof(Real),"simultaneously live old and new allocations are counted");
    }
    require(budget.live==0,"vector destruction returns complete allocation charge");

    const std::size_t maximum=std::numeric_limits<std::size_t>::max();
    rejected=false;
    try { (void)sweep_detail::product(maximum,2); }
    catch (const sweep_detail::WorkspaceLimit&) { rejected=true; }
    require(rejected,"workspace size product overflow is rejected");
    rejected=false;
    try { (void)sweep_detail::sum(maximum,1); }
    catch (const sweep_detail::WorkspaceLimit&) { rejected=true; }
    require(rejected,"workspace size sum overflow is rejected");
    rejected=false;
    try { (void)sweep_detail::Allocator<Vec3>(tiny).allocate(maximum/sizeof(Vec3)+1); }
    catch (const sweep_detail::WorkspaceLimit&) { rejected=true; }
    require(rejected && tiny.live==0,"allocator rejects element-byte overflow without acquiring storage");
    require(sweep_detail::product(maximum,0)==0 && sweep_detail::sum(maximum,0)==maximum,
        "exact zero and maximum boundary size arithmetic remains valid");
}
}

int main(int argc,char** argv) {
    require(argc==2,"usage: test_large_sweep_constraints hex130|tet130|tet880|tet4500|near-rank|nonlinear|controls|trajectory|harmonic|allocator");
    const std::string name=argv[1];
    if (name=="near-rank") checkNearRank();
    else if (name=="nonlinear") checkNonlinear();
    else if (name=="controls") checkControls();
    else if (name=="trajectory") checkOriginalTrajectoryPolynomials();
    else if (name=="harmonic") { checkHarmonicModes(); checkHarmonicOutputRounding(); }
    else if (name=="allocator") checkWorkspaceAllocator();
    else {
        require(name=="hex130" || name=="tet130" || name=="tet880" || name=="tet4500","unknown large fixture");
        checkLarge(name);
    }
    std::cout << "PASS: " << name << " production swept-volume constraints\n";
    return 0;
}
