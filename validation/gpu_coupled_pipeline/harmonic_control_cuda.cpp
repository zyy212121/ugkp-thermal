#include "gpu/PolyhedralGeometry.H"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
namespace g=ugkp::gpu::geometry;
// Scalar Jacobi oracle follows the original Geometry.C point/adjacency order.
// Explicit caps exercise partial recurrences and the cooperative grid stride.
g::HarmonicResult reference(const std::vector<g::Point>& points,const std::vector<int>& offsets,const std::vector<int>& adjacent,
    const std::vector<unsigned char>& fixed,const std::vector<g::Point>& boundary,double tolerance,int cap,bool represented){
    std::vector<g::Point> old(points.size()),next(points.size());for(std::size_t p=0;p<points.size();++p)if(fixed[p])old[p]=boundary[p];
    g::HarmonicResult out;bool converged=false;
    for(int iteration=0;iteration<cap&&!converged;++iteration){double residual=0;
        for(std::size_t p=0;p<points.size();++p){if(fixed[p]){next[p]=old[p];continue;}g::Point sum;double total=0;
            for(int k=offsets[p];k<offsets[p+1];++k){const int q=adjacent[k];const double length=g::magnitude(points[p]-points[q]);
                if(!(length>0)){out.errorPoint=p;out.errorKind=1;return out;}const double weight=1/length;sum+=old[q]*weight;total+=weight;}
            next[p]=sum/total;if(!g::finite(next[p])){out.errorPoint=p;out.errorKind=2;return out;}
            residual=std::max(residual,g::magnitude(next[p]-old[p]));}
        old.swap(next);out.residual=residual;converged=residual<=tolerance;++out.iterations;
    }
    out.points=points;for(std::size_t p=0;p<points.size();++p){out.points[p]+=old[p];if(represented){out.points[p]=out.points[p]-points[p];if(fixed[p])out.points[p]=boundary[p];}}
    return out;
}
void exact(double a,double b){assert((std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0);}
int main(){int cases=0;for(int n:{3,17,1025,50001}){
    std::vector<g::Point> points(n),boundary(n);std::vector<int> offsets{0},adjacent;std::vector<unsigned char> fixed(n);
    for(int p=0;p<n;++p){points[p]={.125*p,.0625*(p%7),-.03125*(p%3)};fixed[p]=p%11==0||p==n-1;
        boundary[p]={.01*(p%3),-.03*(p%7),.002*(p%2)};if(p)adjacent.push_back(p-1);if(p+1<n)adjacent.push_back(p+1);offsets.push_back(adjacent.size());}
    for(int cap:{0,1,2,7,8,9,37})for(double tolerance:{0.,1e-12,std::numeric_limits<double>::quiet_NaN()}){
        g::HarmonicPointMotion gpu(points,offsets,adjacent,fixed,tolerance,cap);
        for(bool represented:{false,true}){const auto expected=reference(points,offsets,adjacent,fixed,boundary,tolerance,cap,represented),actual=gpu.solve(boundary,represented);
            assert(expected.errorPoint==actual.errorPoint&&expected.errorKind==actual.errorKind&&expected.iterations==actual.iterations);exact(expected.residual,actual.residual);
            for(int p=0;p<n;++p){exact(expected.points[p].x,actual.points[p].x);exact(expected.points[p].y,actual.points[p].y);exact(expected.points[p].z,actual.points[p].z);}++cases;}
    }
    g::HarmonicPointMotion gpu(points,offsets,adjacent,fixed,1e-12,37);auto bad=boundary;bad[0].x=std::numeric_limits<double>::infinity();
    const auto expected=reference(points,offsets,adjacent,fixed,bad,1e-12,37,false),actual=gpu.solve(bad);
    assert(expected.errorPoint==actual.errorPoint&&expected.errorKind==actual.errorKind);const auto recovered=gpu.solve(boundary),fresh=reference(points,offsets,adjacent,fixed,boundary,1e-12,37,false);
    assert(recovered.errorPoint<0&&recovered.iterations==fresh.iterations);for(int p=0;p<n;++p)exact(fresh.points[p].x,recovered.points[p].x);++cases;
    points[1]=points[0];g::HarmonicPointMotion zero(points,offsets,adjacent,fixed,1e-12,37);const auto invalid=zero.solve(boundary);assert(invalid.errorPoint==1&&invalid.errorKind==1);++cases;
    std::fill(fixed.begin(),fixed.end(),1);g::HarmonicPointMotion allFixed(points,offsets,adjacent,fixed,1e-12,37);const auto accepted=allFixed.solve(boundary);assert(accepted.errorPoint<0&&accepted.iterations==1&&accepted.residual==0);++cases;
}

std::cout<<"PASS "<<cases<<" harmonic caps, tolerances, large grid-stride fields, first errors and workspace recovery; finite states bitwise\n";
}
