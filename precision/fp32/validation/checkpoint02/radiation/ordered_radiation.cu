#include "recovery_counter.cuh"
#include "gpu/OrderedUpwindGraph.cuh"
#include "gpu/RadiationPrecision.H"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace ugkp::gpu;
static_assert(sizeof(RadiationReal)==4,"This fixture is for the FP32 overlay");
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<class T>struct Device{
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){check(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()){if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
    std::vector<T> read(){std::vector<T>v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
double publishedResidual(const OrderedUpwindPlan& plan,const std::vector<int>& owner,
    const std::vector<int>& adjacent,const std::vector<double>& flux,const std::vector<double>& diag,
    const std::vector<double>& source,const std::vector<float>& values,double origin,double small){
    double residual=0,magnitude=small;
    for(int c=0;c<plan.cells;++c){
        const double x=origin+double(values[c]);double r=diag[c]*x-source[c];
        magnitude=std::max(magnitude,std::max(std::abs(diag[c]*x),std::abs(source[c])));
        for(int j=plan.offsets[c];j<plan.offsets[c+1];++j){
            const int i=plan.residualOrder[j];const double upstream=origin+double(values[plan.neighbour[i]]);
            if(plan.subtract[i])r+=flux[plan.face[i]]*upstream;else r-=flux[plan.face[i]]*upstream;
        }
        residual=std::max(residual,std::abs(r));
    }
    for(size_t f=0;f<flux.size();++f)if(flux[f]!=0)
        magnitude=std::max(magnitude,std::abs(flux[f]*(origin+double(values[flux[f]>0?owner[f]:adjacent[f]]))));
    return residual/magnitude;
}
int main()try{
    std::cout<<std::unitbuf;
    for(int pattern=0;pattern<4;++pattern)for(bool zero:{false,true}){
        const int n=1025;const double origin=16384;std::vector<int> owner,adjacent;
        if(pattern==1||pattern==3)for(int i=0;i<n-1;++i){owner.push_back(i);adjacent.push_back(i+1);}
        if(pattern==2){
            for(int i=1;i<=300;++i){owner.push_back(0);adjacent.push_back(i);}
            for(int i=300;i<n-1;++i){owner.push_back(i);adjacent.push_back(i+1);}
            owner.push_back(n-1);adjacent.push_back(3);
        }
        if(pattern==3){owner.push_back(n-1);adjacent.push_back(0);}
        std::vector<double> flux(owner.size(),pattern==3?-.137:.137);
        std::vector<std::vector<int>> rows(n);
        for(int f=0;f<int(owner.size());++f){rows[owner[f]].push_back(f);rows[adjacent[f]].push_back(f);}
        std::vector<int> offsets{0},faces;
        for(const auto& row:rows){faces.insert(faces.end(),row.begin(),row.end());offsets.push_back(faces.size());}
        OrderedUpwindPlan plan;plan.build(n,owner,adjacent,offsets,faces,flux);
        std::vector<double> exact(n),diag(n,1.037),source(n);
        for(int c=0;c<n;++c)exact[c]=zero?-origin:(pattern==3?100.013:1e-3)*(1+c%17);
        for(int c=0;c<n;++c){
            for(int i=plan.offsets[c];i<plan.offsets[c+1];++i)diag[c]+=std::abs(plan.flux[i]);
            source[c]=diag[c]*(origin+exact[c]);
            for(int i=plan.offsets[c];i<plan.offsets[c+1];++i)
                source[c]+=(plan.subtract[i]?1:-1)*plan.flux[i]*(origin+exact[plan.neighbour[i]]);
        }
        Device<int> dOffsets(plan.offsets),dNeighbour(plan.neighbour),dSubtract(plan.subtract),
            dOrder(plan.residualOrder),dLevels(plan.levelCells),dMap(plan.face),dOwner(owner),dAdjacent(adjacent);
        Device<double>dNative(flux),dDiag(diag),dSource(source),dOffset(std::vector<double>{origin}),
            dReference(std::vector<double>{2*origin});
        Device<float>dFlux(std::vector<float>(plan.flux.begin(),plan.flux.end())),
            dFloatDiag(std::vector<float>(diag.begin(),diag.end())),dRhs(n),dValues(n),dLow(n),dPrevious(n);
        OrderedUpwindDeviceT<float> matrix{n,dOffsets.p,dNeighbour.p,dSubtract.p,dOrder.p,dLevels.p,
            dFlux.p,dNative.p,dMap.p};
        UpwindLinearControls controls;controls.tolerance=1e-12;controls.maximumIterations=500;
        OrderedUpwindGraphT<float> graph(plan,matrix,flux.size(),dOwner.p,dAdjacent.p,dNative.p,
            dFloatDiag.p,dRhs.p,dValues.p,dPrevious.p,dReference.p,controls,nullptr,false,true,
            dLow.p,dDiag.p,dSource.p,dOffset.p,dRhs.p);
        check(cudaMemset(dValues.p,0,n*sizeof(float)));check(cudaMemset(dLow.p,0,n*sizeof(float)));
        const auto before=checkpoint02RecoveryCount();
        const UpwindLinearReport refined=graph.launch(nullptr);
        const auto high=dValues.read(),low=dLow.read();double error=0,minimum=1e300;
        for(int c=0;c<n;++c){
            error=std::max(error,std::abs(double(high[c])+double(low[c])-exact[c]));
            minimum=std::min(minimum,origin+double(high[c])+double(low[c]));
        }
        require(refined.errorCell<0,"physical intensity failure");
        require(refined.finalResidual<=refined.target,"true refinement residual did not converge");
        require(error<1e-7,"solution error");require(minimum>=0,"negative reconstructed intensity");
        graph.measurePublished(matrix,dOwner.p,dAdjacent.p,dNative.p,dFloatDiag.p,dRhs.p,dValues.p,
            controls,nullptr,dDiag.p,dSource.p,dOffset.p);
        const UpwindLinearReport published=graph.collect(nullptr);
        const double independent=publishedResidual(plan,owner,adjacent,flux,diag,source,high,origin,controls.small);
        require(std::abs(published.finalResidual-independent)<=1e-15*std::max(1.,independent),
            "published residual does not measure actual single-float field");
        if(pattern==3&&!zero)require(published.finalResidual>published.target,
            "published quantization error was hidden by the temporary low component");
        std::cout<<std::setprecision(17)<<"SWEEP pattern="<<pattern<<" zero="<<zero
            <<" recoveryRows="<<(checkpoint02RecoveryCount()-before)
            <<" iterations="<<refined.iterations<<" refinedResidual="<<refined.finalResidual
            <<" publishedResidual="<<published.finalResidual<<" independentResidual="<<independent
            <<" target="<<published.target<<" maximumError="<<error<<" minimumIntensity="<<minimum<<'\n';
    }
    std::cout<<"PASS FP32 correction, original GS dependencies, physical checks and stored-field residual\n";
    return 0;
}catch(const std::exception&e){std::cout<<"FAIL "<<e.what()<<'\n';return 1;}
