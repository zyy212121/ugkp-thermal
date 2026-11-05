#include "mesh/Geometry.H"
#include "mesh/Motion.H"
#include <iomanip>
#include <iostream>
#include <string>
using namespace chmt;
HostMesh boxes(int cells,Real scale=1,Real offset=0,bool skew=false,bool empty=false,bool overlap=false) {
    HostMesh m;m.faceOffsets.push_back(0);
    for(int c=0;c<cells;++c) {
        Real x=2*c;if(overlap && c==61)x=120.5;
        const int first=m.points.size();
        for(Vec3 p:std::vector<Vec3>{{x,0,0},{x+1,0,0},{x+1,1,0},{x,1,0},
            {x,0,1},{x+1,0,1},{x+1,1,1},{x,1,1}})
            m.points.push_back({offset+scale*(p.x+(skew?.125*p.y:0)),
                -offset+scale*(p.y+(skew?.0625*p.z:0)),offset+scale*p.z});
        int face=0;
        for(const auto& f:std::vector<std::vector<int>>{{0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}}) {
            for(int p:f)m.facePoints.push_back(first+p);
            m.faceOffsets.push_back(m.facePoints.size());m.owner.push_back(c);m.neighbour.push_back(-1);
            m.boundaryKind.push_back(empty && face<2?BoundaryKind::Empty:BoundaryKind::Slip);++face;
        }
    }
    m.referencePoints=m.points;m.oldPoints={{-7,8,9}};m.oldVolumes={41};m.meshVelocity={{-0.,2,3}};
    m.geometryVersion=19;m.topologyHash=31;m.boundaryPrimitive.resize(m.owner.size());
    m.boundarySst.resize(m.owner.size());m.thermalBoundary.resize(m.owner.size());
    for(std::size_t f=0;f<m.owner.size();++f){m.boundaryPrimitive[f].temperature=f+273;m.boundarySst[f].k=f;}
    return m;
}
void scalarValue(double x){std::cout<<' '<<x;}
void scalarValue(Vec3 p){scalarValue(p.x);scalarValue(p.y);scalarValue(p.z);}
template<class T> void field(const char* name,const std::vector<T>& values){std::cout<<name;for(const auto& x:values)scalarValue(x);std::cout<<'\n';}
void output(int id,bool ok,bool numerical,const std::string& error,const HostMesh& m,const std::vector<Real>& sweeps){
    std::cout<<"STATUS "<<id<<' '<<ok<<' '<<numerical<<' '<<error<<'\n';
    field("points",m.points);field("oldPoints",m.oldPoints);field("areaVectors",m.areaVectors);field("faceCentres",m.faceCentres);
    field("cellCentres",m.cellCentres);field("volumes",m.volumes);field("oldVolumes",m.oldVolumes);field("wallDistance",m.wallDistance);
    field("meshVelocity",m.meshVelocity);field("sweeps",sweeps);std::cout<<"IDENTITY "<<m.geometryVersion<<' '<<m.topologyHash<<'\n';
}
int main(){
    std::cout<<std::setprecision(17)<<std::scientific;int id=0;
    for(int cells:{1,3,128})for(double scale:{.001,1.,1024.})for(bool skew:{false,true})for(bool empty:{false,true}){
        if(skew&&empty)continue;auto m=boxes(cells,scale,11.125,skew,empty);
        for(int moving=0;moving<2;++moving){auto end=m.points;if(moving)for(auto& p:end){p.x+=.03125*scale;p.y+=.0625*scale;if(!empty)p.z*=1.015625;}
            HostMesh result=boxes(1);std::vector<Real> sweep={-0.,7};std::string error="sentinel";bool numerical=false;
            const bool ok=makeStageGeometry(m,end,.125,result,sweep,error,Tolerances{},&numerical);output(id++,ok,numerical,error,result,sweep);}
    }
    for(int invalid=0;invalid<7;++invalid){auto m=boxes(2);auto end=m.points;double dt=.125;Tolerances tol;
        if(invalid==0)end.pop_back();if(invalid==1)end[0].x=undefinedValue();
        if(invalid==2){m.facePoints[2]=m.facePoints[1];}if(invalid==3)for(auto& p:end){p.x=-p.x;p.y=-p.y;}
        if(invalid==4)dt=0;if(invalid==5)tol.absoluteGeometry=-1;if(invalid==6)m.owner[0]=-1;
        HostMesh result=boxes(1);std::vector<Real> sweep={-0.,7};std::string error="sentinel";bool numerical=false;
        const bool ok=makeStageGeometry(m,end,dt,result,sweep,error,tol,&numerical);output(id++,ok,numerical,error,result,sweep);
    }
    std::cout<<"CASES "<<id<<'\n';
}
