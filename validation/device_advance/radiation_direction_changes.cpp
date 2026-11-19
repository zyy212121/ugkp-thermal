#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/OrderedUpwindPlan.H"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ugkp::gpu;

template<class Report> auto hasValidBudget(const Report& result,int)->decltype(result.workLimit,bool()){
    const auto used=result.matvecs+result.totalLinearIterations;
    return used<=result.workLimit&&(result.sourceStop!=4||result.workLimit-used<5);
}
template<class Report> bool hasValidBudget(const Report&,long){return true;}

int main(int argc,char** argv){
    const bool tight=argc==2&&std::string(argv[1])=="tight";
    const bool streaming=argc==2&&std::string(argv[1])=="streaming";
    const bool positivity=argc==2&&std::string(argv[1])=="positivity";
    if(argc>2||(argc==2&&!tight&&!streaming&&!positivity))return 1;
    const int cells=positivity?64:streaming?16:257;
    RadiationGeometry g;
    g.cells=cells;g.directions=2;g.volume.assign(cells,1);
    g.weights.assign(2,2*std::acos(-1.));g.directionCosines={1,-1,-1,1};
    g.boundaryOwner={0,cells-1};g.boundaryPatch={0,1};g.boundaryFace={0,0};
    g.boundaryType={RadiationBoundary::Diffuse,RadiationBoundary::Black};
    g.boundaryArea={1,1};g.diffuseDenominator=g.weights;g.fixedIntensity={0,0};
    g.transportDiagonal.assign(2*cells,1);
    g.boundaryFlux={-1,1,1,-1};g.boundaryCosine=g.boundaryFlux;
    for(int cell=0;cell<cells;++cell){
        g.cellOffsets.push_back(int(g.cellFaces.size()));
        if(cell)g.cellFaces.push_back(cell-1);
        if(cell<cells-1){g.owner.push_back(cell);g.neighbour.push_back(cell+1);g.cellFaces.push_back(cell);}
    }
    g.cellOffsets.push_back(int(g.cellFaces.size()));
    g.internalFlux.assign(cells-1,1);g.internalFlux.resize(2*(cells-1),-1);
    MieOpticalTable table;table.temperature={100,2000};table.diameter={1e-4,1e-2};
    table.mu={-1,1};table.band={1,1};table.absorption.assign(4,1);table.scattering.assign(4,.3);
    table.extinction.assign(4,1.3);table.phase.assign(8,1);
    RadiationSnapshot input;
    input.epsilon.assign(cells,1e-6);input.diameter.assign(cells,1e-3);
    for(int i=0;i<cells;++i)input.temperature.push_back(600.+i*.25);
    input.wallTemperature={400,900};input.emissivity={.75,1};
    RadiationControls controls;controls.parallelAngles=true;controls.maximumConcurrentAngles=2;
    controls.sourceRelaxation=.75;controls.sourceTolerance=2e-5;
    controls.linearTolerance=2e-5;controls.sourceIterations=20;controls.linearIterations=tight?20:100;
    if(positivity){
        g.boundaryType={RadiationBoundary::Fixed,RadiationBoundary::Zero};
        g.fixedIntensity={1000,0};
        table.absorption.assign(4,.03);
        table.scattering.assign(4,.97);
        table.extinction.assign(4,1);
        input.temperature.assign(cells,100);
        for(int i=0;i<cells;++i)input.epsilon[i]=i%3==1?.001:0;
        controls.sourceRelaxation=1;
        controls.sourceTolerance=1e-5;
        controls.linearTolerance=1e-8;
        controls.sourceIterations=64;
        controls.linearIterations=200;
        DiscreteOrdinateRadiation transport(g,table,controls);
        const auto result=transport.solve(input,false,true);
        std::vector<double> x;
        transport.downloadIntensities(x);
        if(x.size()!=std::size_t(2*cells)||!hasValidBudget(result,0)||!result.sourceConverged)return 13;
        for(double value:x)if(!std::isfinite(value)||value<0)return 14;
        if constexpr(sizeof(RadiationReal)==sizeof(double)){
            if(x[2*cells-2]!=0||x[2*cells-1]!=0)return 15;
        }
        const double pi=std::acos(-1.);
        const double black=5.670374419e-8*std::pow(100.,4)/pi;
        double residual=0,scale=0,absorbed=0;
        for(int i=0;i<cells;++i){
            const double extinction=1500*input.epsilon[i];
            const double absorption=extinction*.03;
            const double mean=(x[i]+x[cells+i])*.5;
            absorbed+=4*pi*absorption*(mean-black);
            for(int direction=0;direction<2;++direction){
                const int offset=direction*cells;
                const int neighbour=i+(direction?1:-1);
                const double incoming=neighbour<0?1000:neighbour==cells?0:x[offset+neighbour];
                const double rhs=absorption*black+incoming;
                const double lhs=(1+extinction)*x[offset+i]-extinction*.97*mean;
                residual=std::max(residual,std::abs(lhs-rhs));
                scale=std::max({scale,std::abs(lhs),std::abs(rhs)});
            }
        }
        const double tolerance=radiationConvergenceTolerance(RadiationReal(1e-8));
        const double escaped=2*pi*(x[cells-1]+x[cells]-1000);
        if(residual>tolerance*scale||std::abs(absorbed+escaped)>4*pi*cells*tolerance*scale)return 16;
        std::cout<<"PASS full-domain positive intensity, residual "<<residual/scale
                 <<" and energy defect "<<std::abs(absorbed+escaped)<<'\n';
        return 0;
    }
    if(streaming){
        OrderedUpwindPlan plan;
        const std::vector<int> owners={0,1,2},neighbours={1,2,0},offsets={0,2,4,6},faces={0,2,0,1,1,2};
        plan.build(3,owners,neighbours,offsets,faces,{1,1,1},true);
        if(plan.allUpstreamReady||plan.levelCells!=std::vector<int>({0,1,2}))return 7;
        plan.build(3,owners,neighbours,offsets,faces,{1,-1,1},true);
        if(!plan.allUpstreamReady||plan.levelCells!=std::vector<int>({2,0,1}))return 8;
        input.epsilon.assign(cells,0);
        controls.linearIterations=4;
        controls.sourceIterations=64;
        controls.sourceRelaxation=1;
        controls.linearTolerance=1e-8;
        for(bool reverse:{false,true}){
            g.boundaryType=reverse?std::vector<RadiationBoundary>{RadiationBoundary::Zero,RadiationBoundary::Fixed}
                                  :std::vector<RadiationBoundary>{RadiationBoundary::Fixed,RadiationBoundary::Zero};
            g.fixedIntensity=reverse?std::vector<double>{0,1000}:std::vector<double>{1000,0};
            DiscreteOrdinateRadiation transport(g,table,controls);
            const auto result=transport.solve(input,false,true);
            std::vector<double> values;
            transport.downloadIntensities(values);
            if(values.size()!=std::size_t(2*cells)||!hasValidBudget(result,0))return 9;
            for(std::size_t i=0;i<values.size();++i){
                const double exact=(reverse==(i>=std::size_t(cells)))?1000:0;
                if(!std::isfinite(values[i])||values[i]<0||std::abs(values[i]-exact)>0.001)return 10;
            }
            if(result.sourceStop!=1||result.transportRelativeResidual>result.transportResidualTarget)return 11;
        }
        g.boundaryType.assign(2,RadiationBoundary::Fixed);
        g.fixedIntensity={1000,10};
        controls.sourceRelaxation=.7;
        DiscreteOrdinateRadiation relaxed(g,table,controls);
        const auto balanced=relaxed.solve(input,false,true);
        if(!hasValidBudget(balanced,0)||balanced.sourceStop!=1
            ||balanced.transportRelativeResidual>balanced.transportResidualTarget)return 12;
        return 0;
    }
    DiscreteOrdinateRadiation solver(g,table,controls);
    std::cout<<std::hexfloat;
    for(bool warm:{false,true}){
        const auto result=solver.solve(input,warm,true);
        if(!hasValidBudget(result,0))return 4;
        if(!tight&&result.sourceIterations<controls.sourceIterations&&result.maximumRelativeChange>controls.sourceTolerance)return 3;
        std::vector<double> intensity;solver.downloadIntensities(intensity);
        if(intensity.size()!=std::size_t(2*cells)||result.boundaryFlux.size()!=2)return 1;
        for(double value:intensity)if(!std::isfinite(value)||value<0)return 2;
        std::cout<<result.sourceIterations<<' '<<result.maximumRelativeChange<<' '<<result.transportRelativeResidual<<'\n';
        if(result.directionWarnings.size()!=std::size_t(g.directions))return 5;
        for(const auto& warning:result.directionWarnings){
            if(std::abs(warning.publishedTarget-controls.linearTolerance)>1e-7*controls.linearTolerance)return 6;
            std::cout<<warning.publishedResidual<<' '<<warning.publishedTarget<<'\n';
        }
        for(auto value:result.incident)std::cout<<value<<'\n';
        for(auto value:result.power)std::cout<<value<<'\n';
        for(auto value:result.boundaryFlux)std::cout<<value<<'\n';
        for(auto value:intensity)std::cout<<value<<'\n';
    }
}
