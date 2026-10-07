// Numerical certificates from actual polyhedral trajectories, not mocked residuals.
#define main existing_sweep_fixture_main
#include "test_sweep_constraints.cpp"
#undef main
#include <limits>

namespace {
Tolerances geometryTolerance(Real absolute,Real relative) {
    Tolerances tolerance;
    tolerance.absoluteGeometry=absolute;
    tolerance.relativeGeometry=relative;
    return tolerance;
}
std::vector<Vec3> affineEndpoint(const HostMesh& mesh) {
    auto points=mesh.points;
    for(auto& x:points)x={1.1*x.x+.05*x.y+.1,.9*x.y+.07*x.z-.1,1.2*x.z+.02*x.x};
    return points;
}
Real relativeGclResidual(const HostMesh& old,const HostMesh& endpoint,const std::vector<Real>& sweep) {
    Real maximum=0;
    for(std::size_t cell=0;cell<old.volumes.size();++cell){Real sum=0;
        for(int k=old.cellFaceOffsets[cell];k<old.cellFaceOffsets[cell+1];++k)
            sum+=old.cellFaceSigns[k]*sweep[old.cellFaces[k]];
        maximum=maxValue(maximum,absValue(endpoint.volumes[cell]-old.volumes[cell]-sum)
            /maxValue(old.volumes[cell],endpoint.volumes[cell]));
    }
    return maximum;
}
void cellThresholds() {
    const HostMesh old=twoHexahedra(0);const auto points=affineEndpoint(old);
    HostMesh endpoint;std::vector<Real> sweep;std::string error;
    require(makeStageGeometry(old,points,1,endpoint,sweep,error),error);
    const Real absolute=maximumGclResidual(old,endpoint,sweep);
    const Real relative=relativeGclResidual(old,endpoint,sweep);
    require(absolute>0&&relative>0,"affine fixture exposes an actual nonzero GCL rounding residual");
    for(bool relativeOnly:{false,true}) {
        const Real threshold=relativeOnly?relative:absolute;
        for(bool accepts:{true,false}) {
            const Real limit=threshold*(accepts?2:.5);
            const Tolerances tolerance=geometryTolerance(relativeOnly?0:limit,relativeOnly?limit:0);
            HostMesh output=old;output.geometryVersion=91;std::vector<Real> measured{123};
            const bool ok=makeStageGeometry(old,points,1,output,measured,error,tolerance);
            require(ok==accepts,relativeOnly?"configured relative GCL threshold changes acceptance":"configured absolute GCL threshold changes acceptance");
            if(!accepts){require(error.find("independent stage GCL")!=std::string::npos,error);
                require(output.geometryVersion==91&&measured==std::vector<Real>{123},"GCL rejection preserves mesh and sweep outputs");}
        }
    }
}
void periodicThresholds() {
    HostMesh old=twoHexahedra(0);int left=-1,right=-1;
    for(int f=0;f<static_cast<int>(old.owner.size());++f){
        if(old.faceCentres[f].x==0)left=f;
        if(old.faceCentres[f].x==2)right=f;
    }
    require(left>=0&&right>=0,"periodic fixture opposing faces");
    old.boundaryKind[left]=old.boundaryKind[right]=BoundaryKind::Periodic;
    old.periodicPartner[left]=right;old.periodicPartner[right]=left;
    std::string error;require(rebuildGeometry(old,error),error);
    auto points=old.points;for(auto& point:points)point.x=1.0000001*point.x+.125;
    // Measure the same real face sweeps without enforcing periodic pairing.
    HostMesh open=old;open.boundaryKind[left]=open.boundaryKind[right]=BoundaryKind::Slip;
    require(rebuildGeometry(open,error),error);
    HostMesh measured;std::vector<Real> sweep;
    require(makeStageGeometry(open,points,1,measured,sweep,error),error);
    const Real mismatch=absValue(sweep[left]+sweep[right]);
    const Real relative=mismatch/maxValue(absValue(sweep[left]),absValue(sweep[right]));
    require(mismatch>100*maximumGclResidual(open,measured,sweep),"periodic defect separated from cell GCL roundoff");
    for(bool relativeOnly:{false,true})for(bool accepts:{false,true}) {
        const Real limit=(relativeOnly?relative:mismatch)*(accepts?2:.5);
        const Tolerances tolerance=geometryTolerance(relativeOnly?0:limit,relativeOnly?limit:0);
        HostMesh endpoint=old;endpoint.geometryVersion=91;std::vector<Real> actual{123};
        const bool ok=makeStageGeometry(old,points,1,endpoint,actual,error,tolerance);
        require(ok==accepts,relativeOnly?"configured relative periodic threshold changes acceptance":"configured absolute periodic threshold changes acceptance");
        if(!accepts){require(error.find("periodic paired sweep mismatch")!=std::string::npos,error);
            require(endpoint.geometryVersion==91&&actual==std::vector<Real>{123},"periodic rejection preserves outputs");}
        // Surface reconstruction independently certifies the periodic mesh too.
        SurfaceMesh surface;surface.area={123};const SurfaceMesh reference;
        const bool surfaceOk=rebuildTrajectorySurface(old,HostMesh{},reference,measured,HostMesh{},1,surface,error,tolerance);
        // The endpoint must preserve the periodic topology, unlike the measuring mesh.
        require(!surfaceOk,"surface rejects endpoint boundary topology mismatch regardless of tolerance");
        HostMesh periodicEndpoint=old;periodicEndpoint.points=points;
        require(rebuildGeometry(periodicEndpoint,error),error);
        require(rebuildTrajectorySurface(old,HostMesh{},reference,periodicEndpoint,HostMesh{},1,surface,error,tolerance)==accepts,
            "nested surface periodic certificate uses configured tolerance");
    }
}
void nestedCertificates() {
    const HostMesh old=twoHexahedra(0);const auto points=affineEndpoint(old);
    HostMesh endpoint=old;endpoint.points=points;std::string error;
    require(rebuildGeometry(endpoint,error),error);
    const Tolerances tolerance=geometryTolerance(0,0);
    SurfaceMesh output;output.area={123};const SurfaceMesh reference;
    require(!rebuildTrajectorySurface(old,HostMesh{},reference,endpoint,HostMesh{},1,output,error,tolerance),
        "explicit-mesh surface certificate respects strict configured tolerance");
    require(output.area==std::vector<Real>{123},"surface rejection preserves output");
    HostState base;base.gasMesh=old;
    require(!rebuildTrajectorySurface(base,endpoint,HostMesh{},1,output,error,tolerance),
        "state surface overload forwards configured tolerance");
    std::vector<Vec3> displacement(points.size()),directions(points.size());
    for(std::size_t p=0;p<points.size();++p)displacement[p]=points[p]-old.points[p];
    std::vector<unsigned char> fixed(points.size(),1);std::vector<Real> sweeps;
    SweepConstraintReport report;const SweepConstraintControls controls;
    require(constrainFaceSweeps(old,{}, {},displacement,fixed,directions,1,endpoint,sweeps,report,error,controls),error);
    require(!constrainFaceSweeps(old,{}, {},displacement,fixed,directions,1,endpoint,sweeps,report,error,controls,tolerance),
        "constraint certification forwards configured tolerance");
    require(report.status==SweepConstraintStatus::InvalidTrajectory,"strict constraint failure is a trajectory rejection");
    auto coupled=coupledPair();auto aux=coupled.filmAux;for(auto& film:aux)film.solidFront=-.04;
    HostMesh gas,solid;
    require(moveCoupledMeshes(coupled,aux,1,gas,solid,output,error),error);
    require(!moveCoupledMeshes(coupled,aux,1,gas,solid,output,error,tolerance),
        "legacy coupled wrapper forwards configured tolerance");
    require(error.find("independent stage GCL")!=std::string::npos,error);
    require(moveCoupledMeshesConstrained(coupled,aux,{-.04,-.04},1,gas,solid,output,report,error,controls),error);
    require(!moveCoupledMeshesConstrained(coupled,aux,{-.04,-.04},1,gas,solid,output,report,error,controls,tolerance),
        "material constrained path forwards configured tolerance");
    require(error.find("independent stage GCL")!=std::string::npos,error);
}
void invalidToleranceAndValidity() {
    const HostMesh old=twoHexahedra(0);const auto points=affineEndpoint(old);
    std::string error;
    for(Real invalid:{Real(-1),std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::quiet_NaN()})
        for(bool relative:{false,true}) {
            const Tolerances tolerance=geometryTolerance(relative?0:invalid,relative?invalid:0);
            HostMesh endpoint=old;endpoint.geometryVersion=91;std::vector<Real> sweep{123};
            require(!makeStageGeometry(old,points,1,endpoint,sweep,error,tolerance),"invalid configured geometry tolerance rejects");
            require(error.find("invalid stage geometry tolerances")!=std::string::npos,error);
            require(endpoint.geometryVersion==91&&sweep==std::vector<Real>{123},"invalid tolerance rejection preserves outputs");
        }
    const Tolerances tolerance=geometryTolerance(1e100,1e100);
    HostMesh endpoint;std::vector<Real> sweep;auto inverted=old.points;for(auto& point:inverted)point.x=-point.x;
    require(!makeStageGeometry(old,inverted,1,endpoint,sweep,error,tolerance),"loose residual tolerance cannot accept an inverted endpoint");
    auto crossed=old.points;for(auto& point:crossed){point.x=-2*point.x;point.y=-3*point.y;}
    require(!makeStageGeometry(old,crossed,1,endpoint,sweep,error,tolerance),"loose residual tolerance cannot accept an interior crossing");
}
}
int main() {
    cellThresholds();periodicThresholds();nestedCertificates();invalidToleranceAndValidity();
    std::cout<<"configured geometry tolerance regressions: real GCL/periodic abs+rel thresholds, nested certificates, transactional invalid controls and fixed validity passed\n";
}
