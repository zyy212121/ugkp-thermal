#include "sweep_residual_probe.H"
#include "mesh/Geometry.C" // frozen native formula and geometry oracle
#include <random>
#include <iostream>
#include <cstring>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
static void require(bool ok,const char* message,int test){if(!ok){std::cerr<<test<<": "<<message<<'\n';std::exit(1);}}
static void exact(double a,double b,int test){require((std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0,"finite arithmetic mismatch",test);}
static void ratio(double a,double b,int test){
    if(std::isnan(a)){require(std::isnan(b),"NaN metric mismatch",test);return;}
    const double ulp=std::max(std::numeric_limits<double>::denorm_min(),std::fabs(a-std::nextafter(a,0.)));
    require(a==b||std::fabs(a-b)<=ulp,"dimensionless metric differs by more than one FP64 ULP",test);
}
static SweepResidualProbe reference(const g::SweepResidualDefinition& d,const std::vector<g::Point>& end){
    SweepResidualProbe out;chmt::HostMesh mesh;mesh.faceOffsets=d.topology.faceOffsets;mesh.facePoints=d.topology.facePoints;
    mesh.owner.resize(mesh.faceOffsets.size()-1);std::vector<chmt::Vec3> endpoints;
    for(auto p:d.old)mesh.points.push_back({p.x,p.y,p.z});for(auto p:end)endpoints.push_back({p.x,p.y,p.z});
    for(std::size_t p=0;p<end.size();++p)if(!chmt::finite(endpoints[p])){out.status.failure=g::SweepResidualFailure::Endpoint;out.status.index=p;return out;}
    if(d.emptyAxis>=0)for(std::size_t p=0;p<end.size();++p){const auto failure=g::emptyTrajectoryPoint(p,d.emptyMate[p],d.emptyAxis,d.old.data(),end.data());
        if(failure!=g::EmptyTrajectoryFailure::None){out.status.index=p;out.status.failure=failure==g::EmptyTrajectoryFailure::NormalDisplacement?g::SweepResidualFailure::EmptyNormal:g::SweepResidualFailure::EmptyTangential;return out;}}
    // Literal original moving-triangle integration, independent of faceSweep.
    for(std::size_t f=0;f<mesh.owner.size();++f){chmt::Vec3 area{};double volume=0;
        const int begin=mesh.faceOffsets[f],stop=mesh.faceOffsets[f+1],ia=mesh.facePoints[begin];
        for(int k=begin+1;k+1<stop;++k){const int ib=mesh.facePoints[k],ic=mesh.facePoints[k+1];
            const auto a=mesh.points[ia],b=mesh.points[ib],c=mesh.points[ic],da=endpoints[ia]-a,db=endpoints[ib]-b,dc=endpoints[ic]-c;
            const auto average=(chmt::triangleArea(a,b,c)+chmt::triangleArea(a+da*.5,b+db*.5,c+dc*.5)*4+chmt::triangleArea(a+da,b+db,c+dc))/6;
            area+=average;volume+=chmt::dot((da+db+dc)/3,average);}
        out.sweeps.push_back({{area.x,area.y,area.z},volume,0});
        if(!chmt::finite(area)||!std::isfinite(volume)){out.status.failure=g::SweepResidualFailure::Sweep;out.status.index=f;return out;}
    }
    for(std::size_t i=0;i<d.faces.size();++i){const double actual=out.sweeps[d.faces[i]].volume,r=actual-d.targets[i];
        if(!std::isfinite(r)){out.status.failure=g::SweepResidualFailure::FaceResidual;out.status.index=d.faces[i];return out;}
        out.residual.push_back(r/d.scales[i]);
        const double requested=d.absoluteTolerance+d.relativeTolerance*chmt::maxValue(chmt::absValue(d.targets[i]),chmt::absValue(actual));
        double tolerance=chmt::maxValue(requested,32*std::numeric_limits<double>::epsilon()*d.scales[i]);
        if(!d.faceCeiling.empty())tolerance=chmt::minValue(tolerance,d.faceCeiling[i]);
        out.status.metric=chmt::maxValue(out.status.metric,chmt::geometryResidualRatio(chmt::absValue(r),tolerance));out.status.maximumSweep=chmt::maxValue(out.status.maximumSweep,chmt::absValue(r));
    }
    for(std::size_t i=0;i<d.planar.size();++i){const auto p=d.planar[i];const chmt::PlanarTriangleConstraint plane{p.a,p.b,p.c,p.d,p.scale,d.planeTolerance[i]};double coefficients[3];
        chmt::planarPolynomial(mesh,endpoints,plane,coefficients);
        for(double coefficient:coefficients){const double r=coefficient/p.scale;
            if(!std::isfinite(r)){out.status.failure=g::SweepResidualFailure::PlanarResidual;out.status.index=p.a;return out;}
            out.residual.push_back(r);out.status.metric=chmt::maxValue(out.status.metric,chmt::geometryResidualRatio(chmt::absValue(r),plane.tolerance));out.status.maximumPlanarity=chmt::maxValue(out.status.maximumPlanarity,chmt::absValue(r));}
    }
    if(!std::isfinite(out.status.metric)){out.status.failure=g::SweepResidualFailure::Metric;return out;}
    out.rhs=out.residual;std::size_t row=d.faces.size();for(double tolerance:d.planeTolerance)for(int k=0;k<3;++k,++row)if(chmt::absValue(out.rhs[row])<=tolerance)out.rhs[row]=0;
    double maximum=0;for(double& r:out.rhs){maximum=chmt::maxValue(maximum,chmt::absValue(r));r=-r;}
    out.status.consistencyTolerance=chmt::maxValue(128*std::numeric_limits<double>::epsilon(),d.rankTolerance*maximum);return out;
}
static void compare(const SweepResidualProbe& a,const SweepResidualProbe& b,int test){
    require(a.status.failure==b.status.failure&&a.status.index==b.status.index,"first failure mismatch",test);
    if(a.status.failure!=g::SweepResidualFailure::None)return;
    ratio(a.status.metric,b.status.metric,test);exact(a.status.maximumSweep,b.status.maximumSweep,test);exact(a.status.maximumPlanarity,b.status.maximumPlanarity,test);exact(a.status.consistencyTolerance,b.status.consistencyTolerance,test);
    require(a.residual.size()==b.residual.size()&&a.sweeps.size()==b.sweeps.size(),"shape mismatch",test);
    for(std::size_t i=0;i<a.residual.size();++i){exact(a.residual[i],b.residual[i],test);exact(a.rhs[i],b.rhs[i],test);}
    for(std::size_t i=0;i<a.sweeps.size();++i){exact(a.sweeps[i].volume,b.sweeps[i].volume,test);exact(a.sweeps[i].area.x,b.sweeps[i].area.x,test);exact(a.sweeps[i].area.y,b.sweeps[i].area.y,test);exact(a.sweeps[i].area.z,b.sweeps[i].area.z,test);}
}
int main(){std::mt19937_64 random(194835);std::uniform_real_distribution<double> value(-1,1);int cases=0;
    for(int count:{0,1,3,47,130})for(double scale:{1e-100,1e-3,1.,1e3,1e100})for(bool ceiling:{false,true}){
        g::SweepResidualDefinition d;d.topology.faceOffsets={0};d.relativeTolerance=1e-10;d.rankTolerance=1e-12;
        for(int f=0;f<count;++f){const int a=d.old.size();for(int j=0;j<4+f%4;++j){d.old.push_back({scale*value(random),scale*value(random),scale*value(random)});d.topology.facePoints.push_back(a+j);}
            d.topology.faceOffsets.push_back(d.topology.facePoints.size());d.faces.push_back(f);d.targets.push_back(.00002*scale*scale*scale);d.scales.push_back(scale*scale*scale);
            if(ceiling)d.faceCeiling.push_back(1e-12*scale*scale*scale);
            for(int k=3;k<4+f%4;++k){d.planar.push_back({a,a+1,a+2,a+k,scale*scale*scale});d.planeTolerance.push_back((k%2)?1e-12:1.);}}
        g::SweepResidualEvaluation publication(d);
        for(int update=0;update<3;++update){auto end=d.old;for(auto& p:end)p+=g::Point{scale*.001*value(random),scale*.001*value(random),scale*.001*value(random)};
            const auto expected=reference(d,end),got=probeSweepResidual(d,end);compare(expected,got,cases);
            const auto status=publication.evaluate(end);require(status.failure==got.status.failure&&status.index==got.status.index,"publication status mismatch",cases);ratio(status.metric,got.status.metric,cases);++cases;}
        if(count){auto end=d.old;end.back().x=std::numeric_limits<double>::quiet_NaN();end.front().x=std::numeric_limits<double>::infinity();compare(reference(d,end),probeSweepResidual(d,end),cases++);
            auto invalid=d;invalid.targets[0]=std::numeric_limits<double>::quiet_NaN();compare(reference(invalid,d.old),probeSweepResidual(invalid,d.old),cases++);
            invalid=d;invalid.planar.front().scale=0;compare(reference(invalid,d.old),probeSweepResidual(invalid,d.old),cases++);
            invalid=d;invalid.planeTolerance.front()=0;compare(reference(invalid,d.old),probeSweepResidual(invalid,d.old),cases++);
            invalid=d;invalid.planeTolerance.back()=0;compare(reference(invalid,d.old),probeSweepResidual(invalid,d.old),cases++);
            invalid=d;invalid.scales.front()=0;invalid.targets.front()=0;compare(reference(invalid,d.old),probeSweepResidual(invalid,d.old),cases++);
            if(count==3&&scale==1e100){auto huge=d;for(auto& p:huge.old)p=p*1e100;compare(reference(huge,huge.old),probeSweepResidual(huge,huge.old),cases++);}
        }
    }
    g::SweepResidualDefinition empty;empty.topology.faceOffsets={0,4};empty.topology.facePoints={0,1,3,2};empty.old={{0,0,0},{1,0,0},{0,0,1},{1,0,1}};
    empty.faces={0};empty.targets={0};empty.scales={1};empty.relativeTolerance=1e-10;empty.rankTolerance=1e-12;empty.emptyAxis=2;empty.emptyMate={2,3,0,1};
    for(int mutation=0;mutation<3;++mutation){auto end=empty.old;if(mutation==1)end[1].z=.1;if(mutation==2)end[0].x=.2;compare(reference(empty,end),probeSweepResidual(empty,end),cases++);}
    const double special[]={0.,-0.,std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min(),1.,std::numeric_limits<double>::max(),std::numeric_limits<double>::infinity(),-1.,std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> residuals,tolerances;for(double a:special)for(double b:special){residuals.push_back(a);tolerances.push_back(b);}
    for(int i=0;i<4096;++i){residuals.push_back(std::ldexp(.5+.5*std::fabs(value(random)),int(random()%2098)-1073));tolerances.push_back(std::ldexp(.5+.5*std::fabs(value(random)),int(random()%2098)-1073));}
    const auto quotients=probeSweepRatios(residuals,tolerances);
    for(std::size_t i=0;i<quotients.size();++i)ratio(chmt::geometryResidualRatio(residuals[i],tolerances[i]),quotients[i],i);
    std::cout<<"PASS "<<quotients.size()<<" saturated extended-range quotient comparisons, within one FP64 ULP\n";
    std::cout<<"PASS "<<cases<<" frozen swept-volume/planarity residual and RHS configurations, device graph replay, failure ordering and one-way status publication\n";
}
