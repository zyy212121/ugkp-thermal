// Native CPU reference is selected from the frozen pre-migration include root.
#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "stage_geometry_probe.H"
int checkedStages=0,validStages=0,failedStages=0;unsigned long long conservationWarnings=0,accuracyWarnings=0;double maximumRelative=0;
void close(double a,double b){if(a==b)return;const double scale=std::max({std::abs(a),std::abs(b),1e-300});
    const double error=std::abs(a-b)/scale;maximumRelative=std::max(maximumRelative,error);
    if(!(error<=3e-12)){std::cerr.precision(17);std::cerr<<"stage quantity differs: "<<a<<' '<<b<<" relative="<<error<<'\n';std::abort();}}
void close(Vec3 a,device::Point b){close(a.x,b.x);close(a.y,b.y);close(a.z,b.z);}
void compareStage(HostMesh old,const std::vector<Vec3>& endpoint,chmt::Tolerances tolerance={},double interval=1e-3){
    std::string error;assert(chmt::rebuildGeometry(old,error));chmt::empty_detail::Extrusion empty;assert(chmt::empty_detail::build(old,empty,error));
    HostMesh native;std::vector<double> sweep;ugkp::diagnostics::DeviceContext diagnostics{};tolerance.diagnostics=&diagnostics;
    const bool expected=chmt::makeStageGeometry(old,endpoint,interval,native,sweep,error,tolerance);
    std::vector<int> walls;for(std::size_t f=0;f<old.owner.size();++f)if(old.boundaryKind[f]==chmt::BoundaryKind::Slip||old.boundaryKind[f]==chmt::BoundaryKind::NoSlip||old.boundaryKind[f]==chmt::BoundaryKind::Interface)walls.push_back(f);
    const auto result=stageProbe::evaluate(topology(old),points(old.points),points(endpoint),walls,old.periodicPartner,empty.axis,empty.mate,
        {interval,tolerance.absoluteGeometry,tolerance.relativeGeometry});
    const auto status=result.status;
    if(expected!=(status.failure==device::StageGeometryFailure::None)){std::cerr<<"stage "<<checkedStages<<" acceptance differs native="<<error<<" device="<<int(status.failure)<<" phase="<<int(status.phase)<<" index="<<status.index<<'\n';std::abort();}
    if(!expected){
        const char* messages[]={"","invalid stage geometry tolerances","invalid stage interval","nonfinite vertex","degenerate face area","nonfinite cell area closure",
            "inverted/nonpositive cell volume","nonfinite cell centroid","rotational/nonmatching periodic areas unsupported","rotational/nontranslational periodic vertices unsupported",
            "duplicate tangential point in one layer","nonfinite wall triangle distance","invalid wall distance","nonzero normal trajectory displacement","layer-paired tangential trajectory differs",
            "zero-radius cell fan","nonfinite cell trajectory determinant","moving cell tetrahedral fan collapses or inverts","cell fan is not a single star-shaped volume",
            "cell inversion inside linear stage trajectory","nonfinite stage area/swept volume","nonfinite stage area closure"};
        std::string actual;
        if(status.failure==device::StageGeometryFailure::EmptyDuplicate||status.failure==device::StageGeometryFailure::EmptyNormal||status.failure==device::StageGeometryFailure::EmptyTangential)
            chmt::empty_detail::reject(actual,messages[int(status.failure)]);
        else chmt::errorAt(actual,messages[int(status.failure)],status.index);
        if(actual!=error){std::cerr<<"stage "<<checkedStages<<" error differs "<<actual<<" != "<<error<<'\n';std::abort();}++failedStages;
    }else{
        std::vector<Vec3> averagedArea;std::vector<double> integratedSweep;std::string integrationError;
        assert(chmt::stageIntegrals(old,endpoint,averagedArea,integratedSweep,integrationError,empty.axis>=0?&empty:nullptr));
        for(std::size_t f=0;f<averagedArea.size();++f)close(averagedArea[f],result.sweep[f].area);
        for(std::size_t f=0;f<native.owner.size();++f){close(native.areaVectors[f],result.faces[f].area);close(native.faceCentres[f],result.faces[f].centre);
            close(native.meshVelocity[f],result.velocity[f]);close(sweep[f],result.sweep[f].volume);}
        for(std::size_t c=0;c<native.volumes.size();++c){close(native.volumes[c],result.cells[c].volume);close(native.cellCentres[c],result.cells[c].centre);close(native.wallDistance[c],result.distances[c].distance);}
        ++validStages;
    }
    for(int category=0;category<2;++category){const auto& a=diagnostics.warnings[category];const auto& b=result.diagnostics.warnings[category];
        if(a.count!=b.count){std::cerr<<"stage "<<checkedStages<<" warning count "<<category<<" differs "<<a.count<<' '<<b.count<<'\n';std::abort();}
        if(a.count){assert(a.sample.entityKind==b.sample.entityKind&&a.sample.entityIndex==b.sample.entityIndex&&a.sample.sourceSiteId==b.sample.sourceSiteId);
            close(a.sample.observed,b.sample.observed);close(a.sample.reference,b.sample.reference);}}
    conservationWarnings+=diagnostics.warnings[0].count;accuracyWarnings+=diagnostics.warnings[1].count;
    ++checkedStages;
}
int main(){
    for(int cells:{1,7,129})for(double scale:{.001,1.,1024.})for(int kind=0;kind<3;++kind){
        auto mesh=boxes(cells,scale,kind!=2);mesh.periodicPartner.assign(mesh.owner.size(),-1);
        if(kind==1)for(int c=0;c<cells;++c){mesh.periodicPartner[6*c+2]=6*c+3;mesh.periodicPartner[6*c+3]=6*c+2;
            mesh.boundaryKind[6*c+2]=mesh.boundaryKind[6*c+3]=chmt::BoundaryKind::Periodic;}
        if(kind==2)for(int c=0;c<cells;++c)mesh.boundaryKind[6*c]=mesh.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
        for(int mode=0;mode<7;++mode){auto end=mesh.points;
            for(auto& p:end){if(mode==1)p.x+=.03125*scale;if(mode==2)p.x*=1.02;
                if(mode==3){p.x=-p.x;p.y=-p.y;}if(mode==4)p.z=-p.z;
                if(mode==5)p.z*=2e-15;if(mode==6)p.z*=1e-20;}
            compareStage(mesh,end);
        }
    }
    auto mesh=boxes(7,1.,false);for(int c=0;c<7;++c)mesh.boundaryKind[6*c]=mesh.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
    auto endpoint=mesh.points;for(int p=0;p<8;++p)endpoint[8+p]=endpoint[p];compareStage(mesh,endpoint);
    // Two individually valid cells exchange places: the midpoint has duplicate
    // Empty coordinates although the endpoint and per-cell trajectories pass.
    endpoint=mesh.points;for(int p=0;p<8;++p){endpoint[p]=mesh.points[8+p];endpoint[8+p]=mesh.points[p];}compareStage(mesh,endpoint);
    auto simple=boxes(1,1.,false);chmt::Tolerances invalid;invalid.absoluteGeometry=-1;compareStage(simple,simple.points,invalid);
    compareStage(simple,simple.points,{},0);compareStage(simple,simple.points,{},std::numeric_limits<double>::quiet_NaN());
    for(int cells:{1,7,129})for(int kind=0;kind<3;++kind){
        auto m=boxes(cells,.73,kind!=2);m.periodicPartner.assign(m.owner.size(),-1);
        if(kind==1)for(int c=0;c<cells;++c){m.periodicPartner[6*c+2]=6*c+3;m.periodicPartner[6*c+3]=6*c+2;
            m.boundaryKind[6*c+2]=m.boundaryKind[6*c+3]=chmt::BoundaryKind::Periodic;}
        if(kind==2)for(int c=0;c<cells;++c)m.boundaryKind[6*c]=m.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
        auto end=m.points;for(auto& p:end){p.x+=.0013*p.y*p.z;p.y+=.00017*p.z;}
        if(kind==2)for(std::size_t p=0;p<end.size();++p){end[p]=m.points[p];end[p].x+=.0013*m.points[p].y*m.points[p].y;}
        chmt::Tolerances exactBudget;exactBudget.absoluteGeometry=exactBudget.relativeGeometry=0;compareStage(m,end,exactBudget);
    }
    for(double scale:{1e-15,1.,1e15}){
        HostMesh pyramid;pyramid.faceOffsets={0};pyramid.points={{0,0,0},{scale,0,0},{scale,scale,0},{0,scale,0},{.2*scale,.4*scale,scale}};
        for(const auto face:std::vector<std::vector<int>>{{0,3,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}}){
            pyramid.facePoints.insert(pyramid.facePoints.end(),face.begin(),face.end());pyramid.faceOffsets.push_back(pyramid.facePoints.size());
            pyramid.owner.push_back(0);pyramid.neighbour.push_back(-1);pyramid.boundaryKind.push_back(chmt::BoundaryKind::Slip);}
        compareStage(pyramid,pyramid.points);auto end=pyramid.points;end[4].x+=1.5*scale;compareStage(pyramid,end);
        end=pyramid.points;end[4].z=-scale;compareStage(pyramid,end);
    }
    HostMesh star;star.faceOffsets={0};const double pi=std::acos(-1.);
    for(int p=0;p<5;++p)star.points.push_back({std::cos(2*pi*p/5),std::sin(2*pi*p/5),0});
    star.points.push_back({0,0,1});star.points.push_back({0,0,-1});
    for(int p=0;p<5;++p)for(int side=0;side<2;++side){const int a=(2*p)%5,b=(2*p+2)%5;
        star.facePoints.insert(star.facePoints.end(),side?std::initializer_list<int>{b,a,6}:std::initializer_list<int>{a,b,5});
        star.faceOffsets.push_back(star.facePoints.size());star.owner.push_back(0);star.neighbour.push_back(-1);star.boundaryKind.push_back(chmt::BoundaryKind::Slip);}
    compareStage(star,star.points);endpoint=star.points;for(auto& p:endpoint)p.x+=.01;compareStage(star,endpoint);
    HostMesh empty;empty.faceOffsets={0};compareStage(empty,{});
    std::cout<<"PASS "<<checkedStages<<" complete native/GPU stage transactions: valid="<<validStages<<" invalid="<<failedStages<<" maximum relative field/warning difference="<<maximumRelative<<"; conservation warnings="<<conservationWarnings<<" accuracy warnings="<<accuracyWarnings<<"; captured execution and state reuse\n";return 0;
}
