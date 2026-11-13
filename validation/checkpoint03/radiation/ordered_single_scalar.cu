#include "gpu/OrderedUpwindGraph.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace ugkp::gpu;
using Real=RadiationReal;
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<class T>struct Device{
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){check(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()){if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
    std::vector<T> read(){std::vector<T>v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
Real independentResidual(const OrderedUpwindPlan& plan,const std::vector<int>& owner,
    const std::vector<int>& adjacent,const std::vector<Real>& flux,const std::vector<Real>& diag,
    const std::vector<Real>& source,const std::vector<Real>& values,Real small){
    Real residual=0,magnitude=small;
    for(int c=0;c<plan.cells;++c){
        Real r=diag[c]*values[c]-source[c];
        magnitude=std::max(magnitude,std::max(std::abs(diag[c]*values[c]),std::abs(source[c])));
        for(int j=plan.offsets[c];j<plan.offsets[c+1];++j){
            const int i=plan.residualOrder[j];const Real upstream=values[plan.neighbour[i]];
            if(plan.subtract[i])r+=flux[plan.face[i]]*upstream;else r-=flux[plan.face[i]]*upstream;
        }
        residual=std::max(residual,std::abs(r));
    }
    for(size_t f=0;f<flux.size();++f)if(flux[f]!=0)
        magnitude=std::max(magnitude,std::abs(flux[f]*values[flux[f]>0?owner[f]:adjacent[f]]));
    return residual/magnitude;
}
int main()try{
    std::cout<<std::unitbuf;
    for(int pattern=0;pattern<4;++pattern)for(bool zero:{false,true}){
        const int n=1025;std::vector<int> owner,adjacent;
        if(pattern==1||pattern==3)for(int i=0;i<n-1;++i){owner.push_back(i);adjacent.push_back(i+1);}
        if(pattern==2){
            for(int i=1;i<=300;++i){owner.push_back(0);adjacent.push_back(i);}
            for(int i=300;i<n-1;++i){owner.push_back(i);adjacent.push_back(i+1);}
            owner.push_back(n-1);adjacent.push_back(3);
        }
        if(pattern==3){owner.push_back(n-1);adjacent.push_back(0);}
        std::vector<Real> flux(owner.size(),pattern==3?Real(-.137):Real(.137));
        std::vector<std::vector<int>> rows(n);
        for(int f=0;f<int(owner.size());++f){rows[owner[f]].push_back(f);rows[adjacent[f]].push_back(f);}
        std::vector<int> offsets{0},faces;
        for(const auto& row:rows){faces.insert(faces.end(),row.begin(),row.end());offsets.push_back(faces.size());}
        OrderedUpwindPlan plan;plan.build(n,owner,adjacent,offsets,faces,flux);
        std::vector<Real> exact(n),diag(n,Real(1.037)),source(n);
        for(int c=0;c<n;++c)exact[c]=zero?0:(pattern==3?Real(100.013):Real(1e-3))*(1+c%17);
        for(int c=0;c<n;++c){
            for(int i=plan.offsets[c];i<plan.offsets[c+1];++i)diag[c]+=std::abs(plan.flux[i]);
            source[c]=diag[c]*exact[c];
            for(int i=plan.offsets[c];i<plan.offsets[c+1];++i)
                source[c]+=(plan.subtract[i]?1:-1)*plan.flux[i]*exact[plan.neighbour[i]];
        }
        Device<int> dOffsets(plan.offsets),dNeighbour(plan.neighbour),dSubtract(plan.subtract),
            dOrder(plan.residualOrder),dLevels(plan.levelCells),dOwner(owner),dAdjacent(adjacent);
        Device<Real>dNative(flux),dDiag(diag),dSource(source),dReference(std::vector<Real>{Real(40000)}),
            dFlux(plan.flux),dValues(n),dPrevious(n);
        OrderedUpwindDevice matrix{n,dOffsets.p,dNeighbour.p,dSubtract.p,dOrder.p,dLevels.p,dFlux.p};
        UpwindLinearControls controls;controls.tolerance=Real(1e-12);controls.maximumIterations=500;
        OrderedUpwindGraph graph(plan,matrix,flux.size(),dOwner.p,dAdjacent.p,dNative.p,
            dDiag.p,dSource.p,dValues.p,dPrevious.p,dReference.p,controls);
        check(cudaMemset(dValues.p,0,n*sizeof(Real)));
        const UpwindLinearReport solved=graph.launch(nullptr);
        const auto values=dValues.read();Real error=0,scale=1,minimum=values[0];
        for(int c=0;c<n;++c){
            error=std::max(error,std::abs(values[c]-exact[c]));scale=std::max(scale,std::abs(exact[c]));
            minimum=std::min(minimum,values[c]);
        }
        require(solved.errorCell<0,"physical intensity failure");
        require(solved.finalResidual<=solved.target,"single-scalar equation did not converge");
        const Real errorBound=std::max(Real(128)*radiationEpsilon,Real(8)*solved.target)*scale;
        require(error<=errorBound,"solution error exceeds preset single-scalar bound");
        require(minimum>=0,"negative physical intensity");
        graph.measurePublished(matrix,dOwner.p,dAdjacent.p,dNative.p,dDiag.p,dSource.p,dValues.p,controls,nullptr);
        const UpwindLinearReport measured=graph.collect(nullptr);
        const Real published=measured.maximumResidual/measured.maximumMagnitude;
        const Real independent=independentResidual(plan,owner,adjacent,flux,diag,source,values,controls.small);
        require(std::abs(published-independent)<=Real(8)*radiationEpsilon*std::max(std::abs(published),std::abs(independent)),
            "reported residual does not measure the actual stored scalar field");
        std::cout<<std::setprecision(17)<<"SWEEP bits="<<8*sizeof(Real)<<" pattern="<<pattern<<" zero="<<zero
            <<" iterations="<<solved.iterations<<" residual="<<solved.finalResidual
            <<" publishedResidual="<<published<<" independentResidual="<<independent
            <<" target="<<solved.target<<" maximumError="<<error<<" errorBound="<<errorBound<<" minimumIntensity="<<minimum<<'\n';
    }
    std::cout<<"PASS single-scalar physical intensity, ordered dependencies and stored-field residual\n";
    return 0;
}catch(const std::exception&e){std::cout<<"FAIL "<<e.what()<<'\n';return 1;}
