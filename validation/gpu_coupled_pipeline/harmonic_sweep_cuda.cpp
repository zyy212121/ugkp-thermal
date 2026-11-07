// The separately selected frozen Geometry.C supplies the native Jacobi,
// sweep/planarity derivatives and pivoted twice-reorthogonalized QR oracle.
#include "gpu/HarmonicSweepCorrection.H"
#include "mesh/Geometry.C"
#include <random>
#include <iostream>
#include <cstring>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
#ifdef TEST_DEVICE_HARMONIC_SWEEP
g::HarmonicSweepResult harmonicSweepDeviceProbe(g::HarmonicSweepCorrection&,const g::HarmonicSweepDefinition&,
    const std::vector<g::Point>&,const std::vector<double>&,double,double);
#endif
static g::HarmonicSweepResult solve(g::HarmonicSweepCorrection& owner,const g::HarmonicSweepDefinition& definition,
    const std::vector<g::Point>& endpoint,const std::vector<double>& rhs,double rank,double consistency){
#ifdef TEST_DEVICE_HARMONIC_SWEEP
    return harmonicSweepDeviceProbe(owner,definition,endpoint,rhs,rank,consistency);
#else
    return owner.solve(endpoint,rhs,rank,consistency);
#endif
}

static void require(bool ok,const char* message,int test){if(!ok){std::cerr<<test<<": "<<message<<'\n';std::exit(1);}}
static bool same(double a,double b){return (std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0;}
static std::vector<chmt::Vec3> unpack(const std::vector<g::Point>& a){std::vector<chmt::Vec3> b;for(auto p:a)b.push_back({p.x,p.y,p.z});return b;}
static std::vector<g::Point> oracle(const g::HarmonicSweepDefinition& d,const std::vector<g::Point>& endpoint,
    const std::vector<double>& rhs,int& rank){
    chmt::HostMesh mesh;mesh.points=unpack(d.points);mesh.faceOffsets=d.topology.faceOffsets;
    mesh.facePoints=d.topology.facePoints;mesh.owner.resize(mesh.faceOffsets.size()-1);
    std::size_t peak=0;chmt::sweep_detail::Budget budget(0,peak);
    using namespace chmt::sweep_detail;
    HarmonicModes harmonic(mesh,d.fixed,budget,d.harmonicTolerance);
    const int rows=rhs.size(),columns=d.modes.size();Rows matrix{Allocator<Reals>(budget)};
    for(int r=0;r<rows;++r)matrix.emplace_back(columns,0,Allocator<double>(budget));
    Points boundary(d.points.size(),chmt::Vec3{},Allocator<chmt::Vec3>(budget)),mode(boundary);
    const auto end=unpack(endpoint);
    const auto generate=[&](int j){std::fill(boundary.begin(),boundary.end(),chmt::Vec3{});
        const auto seed=d.modes[j];boundary[seed.point]={seed.displacement.x,seed.displacement.y,seed.displacement.z};
        if(seed.partner>=0)boundary[seed.partner]=boundary[seed.point];
        require(harmonic.solve(boundary,mode),"native harmonic failed",j);};
    for(int j=0;j<columns;++j){generate(j);int row=0;
        for(std::size_t f=0;f<d.faces.size();++f)matrix[row++][j]=chmt::sweepDerivative(mesh,end,mode.data(),d.faces[f])/d.scales[f];
        for(const auto p:d.planar){double value[3];chmt::PlanarTriangleConstraint plane{p.a,p.b,p.c,p.d,p.scale,0};
            chmt::planarPolynomial(mesh,end,plane,value,mode.data());for(double x:value)matrix[row++][j]=x/p.scale;}
    }
    Reals b(rhs.begin(),rhs.end(),Allocator<double>(budget)),delta(columns,0,Allocator<double>(budget));
    require(chmt::minimumNormCorrection(matrix,b,1e-12,1e-12,delta,rank,nullptr),"native QR failed",0);
    std::vector<g::Point> correction(d.points.size());
    for(int j=0;j<columns;++j){generate(j);for(std::size_t p=0;p<mode.size();++p){
        correction[p].x+=mode[p].x*delta[j];correction[p].y+=mode[p].y*delta[j];correction[p].z+=mode[p].z*delta[j];}}
    return correction;
}
int main(){std::mt19937_64 random(513670);std::uniform_real_distribution<double> uniform(-1,1);int cases=0;
    for(int side:{3,5})for(double scale:{1e-3,1.,1e3})for(bool paired:{false,true})for(int columns:{0,1,5}){
        g::HarmonicSweepDefinition d;d.topology.faceOffsets={0};d.harmonicTolerance=scale*1e-12;
        std::vector<int> fixedPoints;
        for(int y=0;y<side;++y)for(int x=0;x<side;++x){
            d.points.push_back({scale*.2*x,scale*.3*y,scale*.01*((x+y)%3)});
            d.fixed.push_back(x==0||y==0||x==side-1||y==side-1);if(d.fixed.back())fixedPoints.push_back(d.points.size()-1);}
        std::set<std::pair<int,int>> edges;
        for(int y=0;y+1<side;++y)for(int x=0;x+1<side;++x){const int a=y*side+x,b=a+1,c=b+side,e=a+side;
            const int ids[]={a,b,c,e};for(int i=0;i<4;++i){d.topology.facePoints.push_back(ids[i]);edges.insert({ids[i],ids[(i+1)%4]});edges.insert({ids[(i+1)%4],ids[i]});}
            d.topology.faceOffsets.push_back(d.topology.facePoints.size());d.faces.push_back(d.faces.size());
            d.scales.push_back(scale*scale*scale);d.planar.push_back({a,b,c,e,scale*scale*scale});}
        d.offsets.assign(d.points.size()+1,0);for(auto e:edges){++d.offsets[e.first+1];d.neighbours.push_back(e.second);}
        for(std::size_t p=1;p<d.offsets.size();++p)d.offsets[p]+=d.offsets[p-1];
        for(int j=0;j<columns;++j)d.modes.push_back({fixedPoints[j%fixedPoints.size()],paired?fixedPoints[fixedPoints.size()-1-j%fixedPoints.size()]:-1,
            {scale*.03*(j%2),scale*.02,scale*.01*(j%3)}});
        g::HarmonicSweepCorrection solver(d);
        for(int update=0;update<2;++update){auto end=d.points;for(auto& p:end)p.z+=scale*.001*uniform(random);
            std::vector<double> rhs(d.faces.size()+3*d.planar.size());for(double& x:rhs)x=.001*uniform(random);
            int rank=0;const auto expected=oracle(d,end,rhs,rank);
            const auto got=solve(solver,d,end,rhs,1e-12,1e-12);
            require(!got.status.error,"device correction failed",cases);require(got.status.rank==rank,"rank mismatch",cases);
            require(got.status.modeSolves==2*columns,"mode count mismatch",cases);
            for(std::size_t p=0;p<expected.size();++p)require(same(expected[p].x,got.correction[p].x)&&same(expected[p].y,got.correction[p].y)&&same(expected[p].z,got.correction[p].z),"correction mismatch",cases);
            if(columns){auto bad=rhs;bad[0]=std::numeric_limits<double>::quiet_NaN();
                require(solve(solver,d,end,bad,1e-12,1e-12).status.error==3,"nonfinite RHS accepted",cases);
                const auto recovered=solve(solver,d,end,rhs,1e-12,1e-12);require(!recovered.status.error,"failure latched across solves",cases);
                for(std::size_t p=0;p<expected.size();++p)require(same(expected[p].z,recovered.correction[p].z),"reuse mismatch",cases);}
            ++cases;
        }
        if(columns){auto bad=d;bad.modes[0].displacement.x=std::numeric_limits<double>::quiet_NaN();
            g::HarmonicSweepCorrection invalid(bad);std::vector<double> rhs(d.faces.size()+3*d.planar.size());
            require(solve(invalid,bad,d.points,rhs,1e-12,1e-12).status.error==1,"nonfinite boundary mode accepted",cases);}
    }
    std::cout<<"PASS "<<cases<<" frozen native harmonic/Jacobian/QR/correction compositions, paired modes, dependent rows, failed RHS reuse and invalid seeds\n";
}
