#define main original_sweep_main
#include "../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main

void samePoints(const std::vector<Vec3>& a,const std::vector<Vec3>& b) {
    require(a.size()==b.size(),"point count parity");
    for(std::size_t i=0;i<a.size();++i)
        require(a[i].x==b[i].x&&a[i].y==b[i].y&&a[i].z==b[i].z,"point parity");
}
void sameMesh(const HostMesh& a,const HostMesh& b) {
    samePoints(a.points,b.points);samePoints(a.areaVectors,b.areaVectors);
    samePoints(a.cellCentres,b.cellCentres);samePoints(a.faceCentres,b.faceCentres);
    samePoints(a.meshVelocity,b.meshVelocity);
    require(a.volumes==b.volumes&&a.oldVolumes==b.oldVolumes,"volume parity");
    require(a.owner==b.owner&&a.neighbour==b.neighbour,"topology parity");
}
void sameReport(const SweepConstraintReport& a,const SweepConstraintReport& b) {
    require(a.status==b.status&&a.iterations==b.iterations&&a.rank==b.rank
        &&a.constraints==b.constraints&&a.degreesOfFreedom==b.degreesOfFreedom
        &&a.lineSearchTrials==b.lineSearchTrials&&a.maximumSweepResidual==b.maximumSweepResidual
        &&a.maximumPlanarityResidual==b.maximumPlanarityResidual&&a.modeSolves==b.modeSolves,
        "physical residual and solve report parity");
}
int main() {
    int cases=0;
    for(Real scale:{1e-4,1.})for(Real offset:{0.,.25})for(int mode:{0,1,2,3}) {
        auto full=coupledPair();
        for(auto* mesh:{&full.gasMesh,&full.solidMesh}) {
            for(auto& point:mesh->points)point=(point+Vec3{offset,offset,offset})*scale;
            std::string error;require(rebuildGeometry(*mesh,error),error);
        }
        full.solid.resize(full.solidMesh.volumes.size());
        for(std::size_t c=0;c<full.solid.size();++c)full.solid[c].condensed[0]=1000*full.solidMesh.volumes[c];
        auto candidate=full.filmAux;
        const Real unit=scale*scale*scale;
        std::vector<Real> target=mode==0?std::vector<Real>{0,0}
            :mode==2?std::vector<Real>{.04e-8*unit,.1e-8*unit}
            :std::vector<Real>{-.04e-8*unit,-.1e-8*unit};
        candidate[0].solidFront=(mode==2?1:-1)*.03e-8*scale;
        candidate[1].solidFront=(mode==2?1:-1)*.12e-8*scale;
        if(mode==0)candidate=full.filmAux;
        if(mode==3) {
            std::vector<Real> limits;require(materialSweepResidualLimits(full.solidMesh,full.surface.solidFace,limits),"carry limits");
            full.solidSweepRemainder={.25*limits[0],-.25*limits[1]};
        }
        auto resident=full;resident.solid.clear();
        HostMesh gas[2],solid[2];SurfaceMesh surface[2];SweepConstraintReport report[2];
        ugkp::diagnostics::DeviceContext diagnostics[2]{};
        bool numerical[2]{false,false};
        Tolerances tolerance[2];tolerance[0].diagnostics=&diagnostics[0];tolerance[1].diagnostics=&diagnostics[1];
        std::string error[2];
        require(moveCoupledMeshesConstrained(full,candidate,target,1,gas[0],solid[0],surface[0],report[0],error[0],
            SweepConstraintControls{},tolerance[0],&numerical[0]),error[0]);
        require(moveCoupledMeshesConstrained(resident,candidate,target,full.solid.size(),1,gas[1],solid[1],surface[1],report[1],error[1],
            SweepConstraintControls{},tolerance[1],&numerical[1]),error[1]);
        sameMesh(gas[0],gas[1]);sameMesh(solid[0],solid[1]);sameReport(report[0],report[1]);
        require(surface[0].area==surface[1].area&&surface[0].sweptEdgeArea==surface[1].sweptEdgeArea,"surface parity");
        samePoints(surface[0].normal,surface[1].normal);samePoints(surface[0].meshVelocity,surface[1].meshVelocity);
        require(numerical[0]==numerical[1]&&error[0]==error[1],"error classification parity");
        const auto a=ugkp::diagnostics::takeSnapshot(diagnostics[0]),b=ugkp::diagnostics::takeSnapshot(diagnostics[1]);
        require(a.hasFatal==b.hasFatal&&a.warnings.size()==b.warnings.size(),"diagnostic parity");
        HostMesh measured;std::vector<Real> sweeps,remainder[2];
        require(makeStageGeometry(full.solidMesh,solid[0].points,1,measured,sweeps,error[0]),error[0]);
        require(updateMaterialSweepRemainder(full,target,sweeps,solid[0],remainder[0],error[0]),error[0]);
        require(updateMaterialSweepRemainder(resident,target,sweeps,solid[1],remainder[1],error[1]),error[1]);
        require(remainder[0]==remainder[1],"signed carry parity");
        ++cases;
    }
    // Multiple material faces can share a cell. Their inventory error budget
    // is divided identically even when no host material snapshot is present.
    {
        auto full=coupledPair();
        full.surface=SurfaceMesh{};full.filmAux.clear();
        for(int f=0;f<int(full.solidMesh.owner.size());++f)if(full.solidMesh.neighbour[f]<0) {
            full.surface.solidFace.push_back(f);full.surface.gasFace.push_back(-1);
            full.surface.solidCell.push_back(full.solidMesh.owner[f]);
            full.surface.persistentId.push_back(f);full.filmAux.emplace_back();
            full.solidMesh.boundaryKind[f]=BoundaryKind::Interface;
        }
        full.gasMesh=HostMesh{};full.solid.resize(full.solidMesh.volumes.size());
        auto resident=full;resident.solid.clear();
        std::vector<Real> target(full.filmAux.size(),0);
        HostMesh gas[2],solid[2];SurfaceMesh surface[2];SweepConstraintReport report[2];std::string error;
        require(moveCoupledMeshesConstrained(full,full.filmAux,target,1,gas[0],solid[0],surface[0],report[0],error),error);
        require(moveCoupledMeshesConstrained(resident,resident.filmAux,target,full.solid.size(),1,gas[1],solid[1],surface[1],report[1],error),error);
        sameMesh(solid[0],solid[1]);sameReport(report[0],report[1]);++cases;
    }
    std::cout<<"PASS material geometry explicit count: "<<cases<<" host/resident parity cases\n";
}
