#define main unused_geometry_test_main
#include "polyhedral_geometry_cuda.cpp"
#undef main
void sameScalar(double a,double b){if(std::isnan(a)&&std::isnan(b))return;exact(a,b);}
int main(){device::PolyhedralGeometry gpu;int queries=0,adaptive=0;
    for(int count:{1,2,7})for(double scale:{1e-4,1.,1e4})for(bool skew:{false,true}){
        auto mesh=boxes(count,scale,skew);std::string error;assert(chmt::rebuildGeometry(mesh,error));
        chmt::SurfaceMesh surface;for(std::size_t f=0;f<mesh.owner.size();++f)surface.solidFace.push_back(f);
        std::vector<chmt::SurfaceEdge> edges;assert(chmt::surfaceEdges(mesh,surface,edges,error));
        std::vector<device::SurfaceEdge> packed;for(auto e:edges)packed.push_back({e.a,e.b,surface.solidFace[e.owner],e.neighbour<0?-1:surface.solidFace[e.neighbour],e.neighbourA,e.neighbourB});
        for(int movement=0;movement<3;++movement){auto end=mesh.points;
            for(std::size_t p=0;p<end.size();++p){if(movement==1)end[p]+=Vec3{scale*.013,scale*-.007,scale*.003};
                if(movement==2)end[p]+=Vec3{scale*.001*(p%3),scale*-.002*(p%4),scale*.003*(p%5)};}
            auto endpoint=mesh;endpoint.points=end;assert(chmt::rebuildGeometry(endpoint,error));
            for(bool empty:{false,true}){
                for(int segments:{1,8,16,128}){const auto out=gpu.surfaceEdges(topology(mesh),points(mesh.points),points(end),packed,empty,segments);
                    for(std::size_t i=0;i<edges.size();++i){const double expected=chmt::edgeSweepQuadrature(mesh,end,surface,edges[i],segments,empty);sameScalar(expected,out[i].sweep);assert(out[i].invalid==(!std::isfinite(expected)?2:0));++queries;}}
                const double length=chmt::meshScale(mesh.points),scale2=length*length;
                const auto out=gpu.surfaceEdges(topology(mesh),points(mesh.points),points(end),packed,empty,0,length);
                for(std::size_t i=0;i<edges.size();++i){auto e=edges[i];Vec3 normal=chmt::normalized(endpoint.areaVectors[surface.solidFace[e.owner]]);
                    if(e.neighbour>=0)normal=chmt::normalized(normal+chmt::normalized(endpoint.areaVectors[surface.solidFace[e.neighbour]]));
                    const Vec3 tangent=end[e.b]-end[e.a],conormal=chmt::normalized(chmt::cross(tangent,normal));exact(conormal,out[i].conormal);exact(chmt::mag(tangent),out[i].length);
                    exact((end[e.a]+end[e.b])*.5-endpoint.faceCentres[surface.solidFace[e.owner]],out[i].ownerOffset);
                    if(e.neighbour>=0)exact((end[e.neighbourA]+end[e.neighbourB])*.5-endpoint.faceCentres[surface.solidFace[e.neighbour]],out[i].neighbourOffset);
                    double sweep=chmt::edgeSweepQuadrature(mesh,end,surface,e,8,empty),residual=0,tolerance=0;bool converged=false;
                    for(int segments=16;segments<=1024;segments*=2){const double refined=chmt::edgeSweepQuadrature(mesh,end,surface,e,segments,empty);
                        tolerance=256*std::numeric_limits<double>::epsilon()*scale2+1e-11*std::max(std::abs(refined),std::abs(sweep));
                        residual=std::abs(refined-sweep);sweep=refined;if(residual<=tolerance){converged=true;break;}}
                    exact(sweep,out[i].sweep);exact(residual,out[i].residual);exact(tolerance,out[i].tolerance);assert(out[i].converged==converged&&out[i].invalid==0);++adaptive;
                }
            }
        }
        auto collapsed=mesh.points;for(auto& p:collapsed)p={0,0,0};
        const auto invalid=gpu.surfaceEdges(topology(mesh),points(mesh.points),points(collapsed),packed,false,1);
        for(std::size_t i=0;i<edges.size();++i){double expected=chmt::edgeSweepQuadrature(mesh,collapsed,surface,edges[i],1,false);sameScalar(expected,invalid[i].sweep);}
    }
    std::cout<<"PASS "<<queries<<" frozen fixed-quadrature edges, "<<adaptive<<" adaptive edge geometry/decisions bitwise; collapsed endpoints classified\n";
}
