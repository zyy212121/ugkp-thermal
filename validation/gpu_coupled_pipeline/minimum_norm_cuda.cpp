// Compile the independent oracle from the frozen checkout, not this branch.
#include "mesh/Geometry.C"
#include "gpu/PolyhedralGeometry.H"
#include <random>
#include <iostream>
#include <cstring>
#include <cstdlib>
namespace gpu=ugkp::gpu::geometry;
static void require(bool good,const char* what,int caseId){if(!good){std::cerr<<"case "<<caseId<<": "<<what<<'\n';std::exit(1);}}
static bool same(double a,double b){return (std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(double))==0;}
int main(){
    std::mt19937_64 random(981732);std::uniform_real_distribution<double> uniform(-1,1);
    gpu::DenseMinimumNorm solver;int count=0;
    auto test=[&](int nr,int nc,const std::vector<double>& matrix,const std::vector<double>& rhs,double tolerance){
        std::size_t peak=0;chmt::sweep_detail::Budget budget(0,peak);
        chmt::sweep_detail::Rows rows{chmt::sweep_detail::Allocator<chmt::sweep_detail::Reals>(budget)};
        for(int i=0;i<nr;++i)rows.emplace_back(matrix.begin()+std::size_t(i)*nc,matrix.begin()+std::size_t(i+1)*nc,chmt::sweep_detail::Allocator<double>(budget));
        chmt::sweep_detail::Reals b(rhs.begin(),rhs.end(),chmt::sweep_detail::Allocator<double>(budget)),expected(nc,27.25,chmt::sweep_detail::Allocator<double>(budget));
        int expectedRank=37;
        const bool expectedValid=chmt::minimumNormCorrection(rows,b,tolerance,1e-12,expected,expectedRank,nullptr);
        const auto result=solver.solve(nr,nc,matrix,rhs,tolerance);
        bool valid=result.valid;int rank=result.initialized?result.rank:37;
        if(valid){for(int i=rank;i<nr;++i)if(!std::isfinite(result.rhs[i]))valid=false;
            for(double x:result.delta)if(!std::isfinite(x))valid=false;}
        require(valid==expectedValid,"validity mismatch",count);require(rank==expectedRank,"rank mismatch",count);
        for(int j=0;j<nc;++j)require(same(expected[j],result.initialized?result.delta[j]:27.25),"delta mismatch",count);
        ++count;
    };
    for(int nr:{0,1,2,7,8,9,17,65,129})for(int nc:{0,1,3,8,19,73})for(double scale:{1e-150,1.0,1e150})for(int mode=0;mode<4;++mode){
        std::vector<double> a(std::size_t(nr)*nc),b(nr);
        for(auto& x:a)x=uniform(random)*scale;for(auto& x:b)x=uniform(random)*scale;
        if(mode==1&&nr>1)for(int j=0;j<nc;++j)a[(nr-1)*nc+j]=a[j];
        if(mode==2)for(int i=0;i<nr;++i)for(int j=0;j<nc;++j)a[i*nc+j]=(i+1)*(j+1)*scale;
        if(mode==3)std::fill(a.begin(),a.end(),0);
        test(nr,nc,a,b,1e-12);
    }
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(double bad:{nan,inf,-inf,1e308})for(int location=0;location<8;++location){
        std::vector<double> a={1,0,1,0,1,1},b={3,4};if(location<6)a[location]=bad;else b[location-6]=bad;
        test(2,3,a,b,1e-12);test(2,3,{1,0,1,0,1,1},{3,4},1e-12);
    }
    for(double tolerance:{0.0,1e-16,1e-3,.999,1.,inf,nan})test(3,3,{1,0,0,0,1e-10,0,0,0,1e-14},{1,1,1},tolerance);
    std::cout<<"PASS "<<count<<" frozen QR cases: rank, validity and finite delta bitwise; IEEE classifications; reuse after failure\n";
}
