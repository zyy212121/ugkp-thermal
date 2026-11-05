// The oracle is the separately selected frozen Geometry.C.
#include "mesh/Geometry.C"
#include "gpu/PolyhedralGeometry.H"
#include <random>
#include <iostream>
#include <cstring>
#include <cstdlib>
namespace device=ugkp::gpu::geometry;
static void exact(double a,double b){if((std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(double))==0)return;
    std::cerr.precision(17);std::cerr<<"mismatch "<<a<<' '<<b<<'\n';std::exit(1);}
static std::vector<device::Point> pack(const std::vector<chmt::Vec3>& v){std::vector<device::Point> r;for(auto p:v)r.push_back({p.x,p.y,p.z});return r;}
int main(){std::mt19937_64 random(61427);std::uniform_real_distribution<double> value(-1,1);int cases=0;
    for(int n:{0,1,2,7,65})for(double scale:{1e-100,1e-3,1.,1e3,1e100}){
        chmt::HostMesh mesh;mesh.faceOffsets={0};std::vector<int> faces;std::vector<double> scales;
        std::vector<chmt::PlanarTriangleConstraint> planar;std::vector<device::PlanarConstraint> planes;
        for(int f=0;f<n;++f){int a=mesh.points.size();for(int j=0;j<3+f%5;++j){mesh.points.push_back({scale*value(random),scale*value(random),scale*value(random)});mesh.facePoints.push_back(a+j);}
            mesh.faceOffsets.push_back(mesh.facePoints.size());faces.push_back(f);scales.push_back(1.0+.125*f);
            for(int j=3;j<3+f%5;++j){planar.push_back({a,a+1,a+2,a+j,1.0+.0625*f,0});planes.push_back({a,a+1,a+2,a+j,1.0+.0625*f});}}
        device::Topology t;t.faceOffsets=mesh.faceOffsets;t.facePoints=mesh.facePoints;
        device::SweepLinearization gpu(t,pack(mesh.points),faces,scales,planes);
        for(int update=0;update<4;++update){std::vector<chmt::Vec3> end=mesh.points,correction(end.size());
            for(auto& p:end)p+=chmt::Vec3{.001*scale*value(random),.001*scale*value(random),.001*scale*value(random)};
            const auto polynomials=gpu.planarity(pack(end));
            for(std::size_t p=0;p<planar.size();++p){double expected[3];chmt::planarPolynomial(mesh,end,planar[p],expected);
                for(int j=0;j<3;++j)exact(expected[j],polynomials[p].value[j]);}
            gpu.endpoint(pack(end));gpu.resetCorrection();
            for(int m=0;m<11;++m){std::vector<chmt::Vec3> mode(end.size());
                for(auto& p:mode)p={scale*value(random),scale*value(random),scale*value(random)};
                const auto column=gpu.column(pack(mode));std::size_t row=0;
                for(int f=0;f<n;++f)exact(chmt::sweepDerivative(mesh,end,mode.data(),f)/scales[f],column[row++]);
                for(const auto& plane:planar){double derivative[3];chmt::planarPolynomial(mesh,end,plane,derivative,mode.data());
                    for(double d:derivative)exact(d/plane.scale,column[row++]);}
                const double coefficient=value(random);gpu.accumulate(pack(mode),coefficient);
                for(std::size_t p=0;p<mode.size();++p)correction[p]+=mode[p]*coefficient;
            }
            const auto result=gpu.correction();for(std::size_t p=0;p<result.size();++p){exact(correction[p].x,result[p].x);exact(correction[p].y,result[p].y);exact(correction[p].z,result[p].z);}
            ++cases;
        }
    }
    std::cout<<"PASS "<<cases<<" endpoint configurations with 11 streamed modes each: frozen sweep derivatives, planar coefficients/derivatives and ordered corrections bitwise\n";
}
