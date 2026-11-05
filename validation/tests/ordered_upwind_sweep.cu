#include "gpu/OrderedUpwindPlan.H"
#include "gpu/OrderedUpwindSweep.cuh"
#include "gpu/OrderedUpwindGraph.cuh"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ugkp::gpu;
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T> struct Device {
    T* ptr=nullptr;std::size_t count;
    explicit Device(std::size_t n):count(n){if(n)check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)));}
    explicit Device(const std::vector<T>& x):Device(x.size()){if(count)check(cudaMemcpy(ptr,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(ptr)cudaFree(ptr);}
    std::vector<T> read(){std::vector<T> x(count);if(count)check(cudaMemcpy(x.data(),ptr,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
void identical(const std::vector<double>& a,const std::vector<double>& b,const char* what){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))){
        for(std::size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(double))){std::cerr<<what<<" at "<<i<<' '<<std::hexfloat<<a[i]<<' '<<b[i]<<'\n';break;}
        throw std::runtime_error(what);}
}
int main(int argc,char** argv){
    std::mt19937 rng(47291);int solvedCases=0;
    for(int count:{1,2,17,128,257,4096})for(int pattern=0;pattern<3;++pattern){
        std::vector<std::pair<int,int>> edges;
        for(int c=0;c<count;++c)for(int distance:{1,7,23})if(c+distance<count)edges.push_back({c,c+distance});
        std::shuffle(edges.begin(),edges.end(),rng);
        std::vector<int> owner,neighbour;std::vector<double> flux;
        std::vector<std::vector<int>> cellFaces(count);
        for(std::size_t i=0;i<edges.size();++i){const auto [a,b]=edges[i];owner.push_back(a);neighbour.push_back(b);
            double f=(.01+double(rng()%97)/97.);if(pattern==1)f=-f;else if(pattern==2)f=(rng()%3==0)?0:(rng()%2?f:-f);
            flux.push_back(f);cellFaces[a].push_back(i);cellFaces[b].push_back(i);}
        std::vector<int> offsets(1,0),faces;
        for(auto& row:cellFaces){std::shuffle(row.begin(),row.end(),rng);faces.insert(faces.end(),row.begin(),row.end());offsets.push_back(faces.size());}
        std::vector<double> diagonal(count,2),source(count),expected(count);
        for(int c=0;c<count;++c){source[c]=.1+double(rng()%997)/997.;expected[c]=.2+double(rng()%991)/991.;}
        for(std::size_t i=0;i<flux.size();++i){if(flux[i]>0)diagonal[owner[i]]+=flux[i];else diagonal[neighbour[i]]-=flux[i];}
        if(argc>1&&count!=std::atoi(argv[1]))continue;
        if(argc>2&&pattern!=std::atoi(argv[2]))continue;
        std::cerr<<"UPWIND_CASE "<<count<<' '<<pattern<<std::endl;
        OrderedUpwindPlan plan;plan.build(count,owner,neighbour,offsets,faces,flux);
        Device<int> dOffsets(plan.offsets),dNeighbour(plan.neighbour),dSubtract(plan.subtract),dOrder(plan.residualOrder),dLevels(plan.levelCells);
        Device<double> dFlux(plan.flux);
        OrderedUpwindDevice matrix{count,dOffsets.ptr,dNeighbour.ptr,dSubtract.ptr,dOrder.ptr,dLevels.ptr,dFlux.ptr};
        Device<double> dDiagonal(diagonal),dSource(source),dCurrent(expected),dPrevious(count),dResidual(count),dMagnitude(count);
        Device<UpwindIntensityFailure> failures(count);
        for(int sweep=0;sweep<5;++sweep){
            // Independent literal ordering of the original FV upwind sweep.
            for(int cell=0;cell<count;++cell){double rhs=source[cell];for(int f:cellFaces[cell]){
                if(cell==owner[f]&&flux[f]<0)rhs-=flux[f]*expected[neighbour[f]];
                else if(cell==neighbour[f]&&flux[f]>0)rhs+=flux[f]*expected[owner[f]];}
                expected[cell]=rhs/diagonal[cell];}
            snapshotUpwind<<<(count+127)/128,128>>>(count,dCurrent.ptr,dPrevious.ptr);
            for(std::size_t level=0;level+1<plan.levelOffsets.size();++level){const int begin=plan.levelOffsets[level],end=plan.levelOffsets[level+1];
                orderedUpwindLevel<<<(end-begin+127)/128,128>>>(begin,end,matrix,dDiagonal.ptr,dSource.ptr,dPrevious.ptr,dCurrent.ptr,1.,failures.ptr);}
            check(cudaGetLastError());identical(expected,dCurrent.read(),"ordered sweep");
            for(auto failure:failures.read())if(failure.cell!=-1)throw std::runtime_error("unexpected intensity failure");
            std::vector<double> residual(count),magnitude(count);
            for(int cell=0;cell<count;++cell){residual[cell]=diagonal[cell]*expected[cell]-source[cell];magnitude[cell]=std::max(std::abs(diagonal[cell]*expected[cell]),std::abs(source[cell]));}
            for(std::size_t f=0;f<flux.size();++f){if(flux[f]>0)residual[neighbour[f]]-=flux[f]*expected[owner[f]];
                else if(flux[f]<0)residual[owner[f]]+=flux[f]*expected[neighbour[f]];}
            upwindResidual<<<(count+127)/128,128>>>(matrix,dDiagonal.ptr,dSource.ptr,dCurrent.ptr,dResidual.ptr,dMagnitude.ptr);
            identical(residual,dResidual.read(),"face-ordered residual");identical(magnitude,dMagnitude.read(),"cell normalization terms");
        }
        Device<int> dOwner(owner),dAdjacent(neighbour);Device<double> dNativeFlux(flux),dReference(std::vector<double>{1.});
        const auto start=expected;
        const auto norm=[&](const std::vector<double>& value){
            std::vector<double> residual(count);double magnitude=1e-15,maximum=0;
            for(int c=0;c<count;++c){residual[c]=diagonal[c]*value[c]-source[c];
                magnitude=std::max(magnitude,std::max(std::abs(diagonal[c]*value[c]),std::abs(source[c])));}
            for(std::size_t f=0;f<flux.size();++f){if(flux[f]>0){residual[neighbour[f]]-=flux[f]*value[owner[f]];magnitude=std::max(magnitude,std::abs(flux[f]*value[owner[f]]));}
                else if(flux[f]<0){residual[owner[f]]+=flux[f]*value[neighbour[f]];magnitude=std::max(magnitude,std::abs(flux[f]*value[neighbour[f]]));}}
            for(double x:residual)maximum=std::max(maximum,std::abs(x));return maximum/magnitude;
        };
        for(bool concurrent:{false,true})for(int budget:{0,1,100}){
            if(argc>3&&budget!=std::atoi(argv[3]))continue;
            std::cerr<<"UPWIND_GRAPH "<<count<<' '<<pattern<<' '<<budget<<' '<<concurrent<<std::endl;
            expected=start;check(cudaMemcpy(dCurrent.ptr,start.data(),count*sizeof(double),cudaMemcpyHostToDevice));
            const UpwindLinearControls controls{1e-12,budget==1?.25:0.,1e-15,budget};
            const double initial=norm(expected),target=std::max(controls.tolerance,controls.relativeTolerance*initial);
            double final=initial;int iteration=0;
            while((final>target||(iteration==0&&(budget<=0||!std::isfinite(initial)||std::isnan(target))))&&iteration<std::max(budget,1)){
                ++iteration;
                for(int cell=0;cell<count;++cell){double rhs=source[cell];for(int f:cellFaces[cell]){
                    if(cell==owner[f]&&flux[f]<0)rhs-=flux[f]*expected[neighbour[f]];
                    else if(cell==neighbour[f]&&flux[f]>0)rhs+=flux[f]*expected[owner[f]];}
                    expected[cell]=rhs/diagonal[cell];}
                final=norm(expected);
            }
            OrderedUpwindGraph graph(plan,matrix,flux.size(),dOwner.ptr,dAdjacent.ptr,dNativeFlux.ptr,
                dDiagonal.ptr,dSource.ptr,dCurrent.ptr,dPrevious.ptr,dReference.ptr,controls,nullptr,concurrent);
            graph.launch(nullptr);UpwindLinearReport report;check(cudaMemcpy(&report,graph.report(),sizeof(report),cudaMemcpyDeviceToHost));
            identical(expected,dCurrent.read(),"batched graph solution");
            if(report.iterations!=iteration||report.errorCell!=-1||std::memcmp(&initial,&report.initialResidual,sizeof(double))
                ||std::memcmp(&final,&report.finalResidual,sizeof(double))||std::memcmp(&target,&report.target,sizeof(double)))
                throw std::runtime_error("batched graph convergence mismatch");
        }
        ++solvedCases;
    }
    std::cout<<"PASS "<<solvedCases<<" permuted/cyclic-dependency upwind matrices, five cell-order sweeps, true residuals and three stopping variants in both grid and cooperative-block execution bitwise\n";
}
