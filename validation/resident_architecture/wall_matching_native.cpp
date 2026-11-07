// Generated fixture header uses the existing native geometry test builders.
// Its production includes resolve against a frozen pre-migration source root.
#include "wall_geometry_native_meshes.H"
#include "wall_matching_probe.H"
#include <stdexcept>
int configurations=0,validRows=0,invalidRows=0,invalidMeshes=0;double maximumError=0;
void compare(WallGeometryMesh<double> mesh,bool reconstruction){
    wallMatchingProbe::Mesh input{mesh.points,mesh.cellCentres,mesh.volumes,mesh.faceOffsets,mesh.facePoints,mesh.owner,mesh.neighbour,{},mesh.physicalWall,mesh.omegaConstrained,mesh.gasCell};
    // Preserve caller row order, including repeated requests for one wall.
    for(int f=int(mesh.owner.size())-1;f>=0;--f)if(mesh.neighbour[f]<0&&mesh.physicalWall[f])input.walls.push_back(f);
    if(input.walls.empty())input.walls.push_back(0);input.walls.push_back(input.walls.front());
    const auto device=wallMatchingProbe::evaluate(input,reconstruction);
    for(int replay=0;replay<3;++replay){auto nativeMesh=mesh;if(replay==1)nativeMesh.omegaConstrained.assign(mesh.cellCentres.size(),1);
        WallGeometryOptions<double> options;options.constructQuadrature=false;options.reconstructMatching=reconstruction;
        wallGeometryDetail::Context<double> context;WallStatus status;
        const bool ready=wallGeometryDetail::prepare(nativeMesh,context,status,options);
        const auto& got=device[replay];
        if(!ready){if(got.invalidFace!=status.node)throw std::runtime_error("wall polygon failure differs");++invalidMeshes;continue;}
        int first=input.walls.size(),face=-1;
        for(int row=0;row<int(input.walls.size());++row){WallGeometryDescriptor<double> expected;WallStatus state;
            const bool okay=wallGeometryDetail::build(nativeMesh,context,input.walls[row],expected,state,options);const auto& actual=got.row[row];
            if(okay!=(actual.failure<0)){std::cerr<<configurations<<" replay "<<replay<<" row "<<row<<" native "<<okay<<" failure "<<actual.failure<<'\n';throw std::runtime_error("wall ray acceptance differs");}
            if(!okay){if(state.node!=actual.failure)throw std::runtime_error("wall ray rejection face differs");if(first==int(input.walls.size())){first=row;face=state.node;}++invalidRows;continue;}
            if(actual.donor!=expected.matchingCells.front()||actual.owner!=expected.ownerCell||actual.wallFace!=expected.wallFace)throw std::runtime_error("wall donor identity differs");
            const double length=std::max(expected.matchingDistance,expected.maximumOwnerDistance);
            auto close=[&](double a,double b,double scale){const double relative=std::abs(a-b)/std::max({std::abs(a),std::abs(b),std::abs(scale),1e-300});maximumError=std::max(maximumError,relative);
                if(!(relative<2e-12)){std::cerr.precision(17);std::cerr<<a<<" != "<<b<<'\n';throw std::runtime_error("wall ray metric differs");}};
            const V* ev[]={&expected.origin,&expected.normal,&expected.tangent1,&expected.tangent2,&expected.matchingPoint,&expected.firstMomentVector};
            const V* av[]={&actual.origin,&actual.normal,&actual.tangent1,&actual.tangent2,&actual.matchingPoint,&actual.firstMomentVector};
            for(int k=0;k<6;++k)for(int d=0;d<3;++d)close((*ev[k])[d],(*av[k])[d],k>=1&&k<=3?1.:k==5?length*expected.volume:length);
            close(expected.ownerDistance,actual.ownerDistance,length);close(expected.matchingDistance,actual.matchingDistance,length);
            close(expected.maximumOwnerDistance,actual.maximumOwnerDistance,length);close(expected.matchingDonorDistance,actual.matchingDonorDistance,length);
            close(expected.volume,actual.volume,expected.volume);close(expected.geometryVolume,actual.geometryVolume,expected.volume);
            close(expected.firstMoment,actual.firstMoment,length*expected.volume);++validRows;
        }
        if(got.firstInvalidRow!=first||got.invalidFace!=face)throw std::runtime_error("wall first failed request differs");
    }
    ++configurations;
}
WallGeometryMesh<double> concave(){MeshBuilder b;const std::vector<V> base={{0,0,0},{4,0,0},{4,1,0},{1,1,0},{1,4,0},{0,4,0}};
    for(int layer=0;layer<3;++layer){std::vector<V> low,high;for(auto p:base){p[2]=layer;low.push_back(p);p[2]=layer+1;high.push_back(p);}
        std::vector<std::vector<V>> faces={low,high};for(std::size_t j=0;j<base.size();++j)faces.push_back({low[j],low[(j+1)%base.size()],high[(j+1)%base.size()],high[j]});
        b.cell(faces,{.5,.5,layer+.5},7);b.m.cellCentres.back()={19./14,19./14,layer+.5};}
    b.m.physicalWall[0]=1;return b.m;
}
int main(){
    for(bool reconstruct:{false,true})for(int shape=0;shape<4;++shape)for(double scale:{1e-6,1.,1e6})for(int mode=0;mode<6;++mode){
        auto m=shape==0?cubes():shape==1?cubes(4,.2):shape==2?tetra():concave();
        for(auto& p:m.points)for(auto& x:p)x*=scale;for(auto& p:m.cellCentres)for(auto& x:p)x*=scale;
        for(auto& v:m.volumes)v*=scale*scale*scale;
        if(mode==1){m.omegaConstrained.assign(m.cellCentres.size(),0);m.omegaConstrained[1]=1;}
        if(mode==2){m.gasCell.assign(m.cellCentres.size(),1);m.gasCell[1]=0;}
        if(mode==3){m.gasCell.assign(m.cellCentres.size(),1);m.gasCell[0]=0;}
        if(mode==4)for(int f=0;f<int(m.owner.size());++f)if(m.neighbour[f]<0)m.physicalWall[f]=1;
        if(mode==5){for(auto& p:m.points)p[2]*=1e-12;for(auto& p:m.cellCentres)p[2]*=1e-12;for(auto& v:m.volumes)v*=1e-12;}
        compare(m,reconstruct);
    }
    auto bad=cubes();const auto origin=bad.points[bad.facePoints[bad.faceOffsets[0]]];
    for(int k=bad.faceOffsets[0];k<bad.faceOffsets[1];++k)bad.points[bad.facePoints[k]]=origin;compare(bad,false);
    for(bool reconstruct:{false,true}){
        auto longRay=cubes(80);longRay.omegaConstrained.assign(80,1);longRay.omegaConstrained.back()=0;compare(longRay,reconstruct);
        auto thin=cubes(4);for(auto& p:thin.points)p[2]*=1e-30;for(auto& p:thin.cellCentres)p[2]*=1e-30;for(auto& v:thin.volumes)v*=1e-30;compare(thin,reconstruct);
        for(double volume:{0.,-1.,std::numeric_limits<double>::quiet_NaN()}){auto input=cubes();input.volumes[0]=volume;compare(input,reconstruct);}
    }
    wallMatchingProbe::Mesh empty;empty.offsets={0};for(const auto& r:wallMatchingProbe::evaluate(empty,false))if(r.invalidFace>=0||r.firstInvalidRow!=0||!r.row.empty())throw std::runtime_error("empty wall partition failed");
    std::cout.precision(17);std::cout<<"PASS "<<configurations<<" native/GPU wall matching configurations, three captured mask replays; valid rows="<<validRows<<" rejected rows="<<invalidRows
        <<" rejected mesh phases="<<invalidMeshes<<" maximum scaled metric error="<<maximumError<<"; empty partition passed\n";
}
