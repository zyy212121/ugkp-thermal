#include "mesh/Geometry.C"
#include "gpu/ExtendedGeometryMath.H"
#include <array>
#include <cassert>
#include <iostream>
#include <random>
namespace geometry=ugkp::gpu::geometry;
void extendedGeometryQuery(int,const geometry::Point*,double*);
int main(){
    using P=geometry::Point;std::vector<P> input;std::vector<double> expected;
    const auto add=[&](P p,P a,P b,P c){for(P x:{p,a,b,c})input.push_back(x);
        const auto v=[](P q)->chmt::Vec3{return {q.x,q.y,q.z};};expected.push_back(chmt::triangleDistance(v(p),v(a),v(b),v(c)));};
    add({2e99,2e99,1e99},{0,0,0},{1e100,0,0},{0,1e100,0});
    add({5e99,.25,.1},{0,0,0},{1e100,0,0},{1e100,1,0});
    add({.5,1,0},{0,0,0},{1,0,0},{2,0,0});add({.5,1,0},{0,0,0},{0,0,0},{2,0,0});
    add({3,4,0},{0,0,0},{0,0,0},{0,0,0});add({.25,.25,0},{0,0,0},{1,0,0},{0,1,0});
    add({5e299,.25,.1},{0,0,0},{1e300,0,0},{1e300,1,0});
    add({5e-301,2e-301,1e-301},{0,0,0},{1e-300,0,0},{0,1e-300,0});
    const double maximum=std::numeric_limits<double>::max(),nan=std::numeric_limits<double>::quiet_NaN(),infinity=std::numeric_limits<double>::infinity();
    add({maximum,maximum,maximum},{-maximum,-maximum,-maximum},{-maximum,-maximum,-maximum},{-maximum,-maximum,-maximum});
    add({nan,0,0},{0,0,0},{1,0,0},{0,1,0});add({infinity,0,0},{0,0,0},{1,0,0},{0,1,0});
    std::mt19937_64 random(0x7839abc);std::uniform_real_distribution<double> unit(-1,1);
    for(double scale:{1e-300,1e-150,1e-100,1.,1e100,1e150,1e300})for(int i=0;i<4096;++i){
        const auto p=[&]()->P{return {scale*unit(random),scale*unit(random),scale*unit(random)};};
        const P a=p(),b=p(),c=i%19==0?a:p(),q=p();add(q,a,b,c);}
    std::vector<double> gpu(expected.size());extendedGeometryQuery(expected.size(),input.data(),gpu.data());
    double maximumRelative=0;std::size_t finite=0;
    for(std::size_t i=0;i<expected.size();++i){const auto p=input.data()+4*i;
        const double host=geometry::extended::triangleDistance(p[0],p[1],p[2],p[3]);
        for(double actual:{host,gpu[i]}){const double reference=expected[i];
            if(std::isnan(reference)){assert(std::isnan(actual));continue;}
            if(!std::isfinite(reference)){assert(reference==actual);continue;}
            assert(std::isfinite(actual));const double relative=reference==0?std::abs(actual):std::abs((actual-reference)/reference);
            maximumRelative=std::max(maximumRelative,relative);
            if(relative>2e-13){std::cerr.precision(17);std::cerr<<"distance mismatch "<<i<<' '<<reference<<' '<<actual<<' '<<relative<<'\n';return 1;}
            ++finite;}
    }
    extendedGeometryQuery(0,nullptr,nullptr);
    std::cout.precision(17);std::cout<<"PASS "<<expected.size()<<" frozen extended-range triangle queries on host and CUDA; maximum relative error "<<maximumRelative<<"; finite comparisons "<<finite<<'\n';
}
