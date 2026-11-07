#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "mesh_trajectory_probe.H"
#include "gpu/GasWindowProgram.H"
int main(){int count=0,samples=0;double maximum=0;
    for(int cells:{1,3})for(double scale:{1e-4,1.,1e4})for(bool empty:{false,true})for(bool gas:{false,true}){
        auto mesh=boxes(cells,scale,false);if(empty)for(int c=0;c<cells;++c)mesh.boundaryKind[6*c]=mesh.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
        std::string error;assert(chmt::rebuildGeometry(mesh,error));chmt::WallProgram program;program.interval.begin=5.;program.interval.end=6.;
        for(double t:{0.,.375,1.}){chmt::WallKnot knot;knot.time=5+t;knot.localTime=chmt::windowOffset(5.,knot.time);knot.gasPoints=mesh.points;knot.solidPoints=mesh.points;
            for(auto& p:knot.gasPoints){p.x+=scale*.03125*t;p.y+=scale*.015625*t;if(!empty)p.z+=scale*.0078125*t;}
            for(auto& p:knot.solidPoints){p.x-=scale*.015625*t;p.y+=scale*.03125*t;if(!empty)p.z-=scale*.0078125*t;}
            program.knots.push_back(knot);}
        const auto actual=trajectoryProbe::evaluate(mesh,program,gas,8,.125);auto previous=mesh.points;auto oldMesh=mesh;chmt::WindowTime elapsed{};
        const auto near=[&](chmt::Vec3 a,device::Point b){for(const auto pair:{std::pair<double,double>{a.x,b.x},{a.y,b.y},{a.z,b.z}}){
            const double denominator=std::max(std::abs(pair.first),std::abs(pair.second));const double relative=denominator?std::abs(pair.first-pair.second)/denominator:0;
            maximum=std::max(maximum,relative);assert(relative<=4e-15);}};
        for(const auto& step:actual){elapsed=elapsed+.125;chmt::WallKnot expected;assert(chmt::sampleGasWallProgramLocal(program,elapsed,expected,error));
            assert(step.status.failure==device::StageGeometryFailure::None);const auto& points=gas?expected.gasPoints:expected.solidPoints;
            for(std::size_t i=0;i<points.size();++i){near(previous[i],step.previous[i]);near(points[i],step.endpoint[i]);near(points[i],step.current[i]);}
            HostMesh endpoint;std::vector<double> sweep;
            assert(chmt::makeStageGeometry(oldMesh,points,.125,endpoint,sweep,error));
            assert(step.mesh.geometryVersion==endpoint.geometryVersion&&step.stage.interval==.125&&step.stage.topologyHash==endpoint.topologyHash);
            assert(step.mesh.volumes.size()==endpoint.volumes.size()&&step.stage.sweptVolume.size()==sweep.size());
            for(std::size_t c=0;c<endpoint.volumes.size();++c){exact(oldMesh.volumes[c],step.mesh.oldVolumes[c]);exact(endpoint.volumes[c],step.mesh.volumes[c]);
                exact(step.mesh.volumes[c],step.stage.newVolume[c]);exact(oldMesh.volumes[c],step.stage.oldVolume[c]);
                near(endpoint.cellCentres[c],{step.mesh.cellCentres[c].x,step.mesh.cellCentres[c].y,step.mesh.cellCentres[c].z});}
            for(std::size_t f=0;f<sweep.size();++f){exact(sweep[f],step.stage.sweptVolume[f]);
                near(endpoint.areaVectors[f],{step.mesh.areaVectors[f].x,step.mesh.areaVectors[f].y,step.mesh.areaVectors[f].z});
                near(oldMesh.faceCentres[f],{step.stage.faceCentre[f].x,step.stage.faceCentre[f].y,step.stage.faceCentre[f].z});}
            oldMesh=std::move(endpoint);previous=points;++samples;}
        ++count;
    }
    std::cout<<"PASS "<<count<<" CHMT mesh trajectory owners, "<<samples<<" captured advances; maximum scaled coordinate difference="<<maximum<<'\n';
}
