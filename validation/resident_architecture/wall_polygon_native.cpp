// Compile this file against the frozen pre-migration common include root.
#include "common/gasWall/WallGeometryBuilder.H"
#include "wall_polygon_probe.H"
#include <iostream>
#include <stdexcept>
#include <cstring>
using namespace ugkwp::gaswall;
using V=std::array<double,3>;
int main(){
    WallGeometryMesh<double> mesh;mesh.faceOffsets.push_back(0);
    auto add=[&](const std::vector<V>& p,const std::vector<int>& supplied=std::vector<int>{}){
        const int begin=mesh.points.size();mesh.points.insert(mesh.points.end(),p.begin(),p.end());
        if(supplied.empty())for(int i=0;i<int(p.size());++i)mesh.facePoints.push_back(begin+i);
        else for(int i:supplied)mesh.facePoints.push_back(begin+i);
        mesh.faceOffsets.push_back(mesh.facePoints.size());};
    const double pi=std::acos(-1.);
    for(double scale:{1e-7,1.,1e5})for(int axis=0;axis<3;++axis)for(int count:{3,4,5,8,17,64,129})for(int kind=0;kind<5;++kind){
        std::vector<V> p;
        for(int i=0;i<count;++i){const double angle=2*pi*i/count;
            const double radius=kind==1&&(i&1)?.3:1.;V x{};
            x[axis]=scale*(radius*std::cos(angle)+(kind==2?100.:0.));
            x[(axis+1)%3]=scale*radius*std::sin(angle)*(kind==3?1e-12:1.);
            x[(axis+2)%3]=kind==4?scale*.01*std::sin(3*angle):0.;p.push_back(x);}
        add(p);
    }
    for(double scale:{1e-7,1.,1e5}){
        auto scaled=[&](std::vector<V> p){for(auto&x:p)for(auto&v:x)v*=scale;return p;};
        add(scaled({{0,0,0},{1,0,0},{2,0,0},{2,1,0},{1,1,0},{0,1,0}}));
        add(scaled({{0,0,0},{1,0,0},{2,0,0},{3,0,0}}));
        add(scaled({{0,0,0},{1,1,0},{0,1,0},{1,0,0}}));
        add(scaled({{0,0,0},{1,0,0},{1,1,0},{0,1,0}}),{0,1,2,3,0});
        add(scaled({{0,0,0},{1,0,0},{1,1,0},{0,1,0}}),{0,1,2,1,3});
        add(scaled({{0,0,0},{2,0,0},{2,1,0},{1,1,0},{1,2,0},{0,2,0}}));
    }
    const auto result=wallPolygonProbe::evaluate(mesh.points,mesh.faceOffsets,mesh.facePoints);
    int valid=0,invalid=0;std::size_t triangles=0;double maxError=0;
    auto close=[&](double a,double b,double scale){const double error=std::abs(a-b)/std::max(scale,1e-300);maxError=std::max(maxError,error);
        if(!(error<5e-13)){std::cerr.precision(17);std::cerr<<a<<" != "<<b<<" scaled "<<error<<'\n';throw std::runtime_error("polygon metric differs");}};
    for(int replay=0;replay<3;++replay){auto sample=mesh;
        if(replay==1){for(auto& p:sample.points){const double x=p[0];p[0]=-p[1]*.5;p[1]=x*.5;p[2]*=.5;}
            for(int f=0;f+1<int(mesh.faceOffsets.size());f+=13){const V origin=sample.points[mesh.facePoints[mesh.faceOffsets[f]]];
                for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k)sample.points[mesh.facePoints[k]]=origin;}}
        for(int f=0;f+1<int(mesh.faceOffsets.size());++f){wallGeometryDetail::Face<double> native;
            const bool okay=wallGeometryDetail::triangulate(sample,f,native,WallGeometryOptions<double>{});const auto& device=result[replay][f];
            if(okay!=bool(device.valid)||native.triangles.size()!=device.triangles.size()){std::cerr<<"face "<<f<<" replay "<<replay<<'\n';throw std::runtime_error("polygon classification/count differs");}
            if(okay)++valid;else ++invalid;
            for(std::size_t t=0;t<native.triangles.size();++t){const auto a=native.triangles[t];const auto b=device.triangles[t];
                if(a.a!=b.a||a.b!=b.b||a.c!=b.c)throw std::runtime_error("ear ordering differs");++triangles;}
            if(okay){close(native.scale,device.scale,native.scale);
                for(int d=0;d<3;++d){close(native.normal[d],device.normal[d],1.);close(native.centre[d],device.centre[d],std::max(native.scale,std::abs(native.centre[d])));}}
        }
    }
    const auto empty=wallPolygonProbe::evaluate({},std::vector<int>{0},{});for(const auto& x:empty)if(!x.empty())throw std::runtime_error("nonempty empty partition");
    std::cout.precision(17);std::cout<<"PASS "<<result[0].size()<<" polygon configurations, 3 captured geometry replays: valid="<<valid<<" invalid="<<invalid
        <<" ordered triangles="<<triangles<<" max scaled metric error="<<maxError<<"; empty partition passed\n";
}
