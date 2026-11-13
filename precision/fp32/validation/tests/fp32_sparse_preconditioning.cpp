#include "gpu/SparseConjugateGradient.H"
#include "gpu/FloatJacobiPreconditioner.H"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
using namespace ugkp::gpu;

// Pending checkpoint-2 batch: this fixture must not run during implementation.
static void encoding(){
    for(const auto& pair:{std::pair<double,double>{1e150,1e-150},
                         {1e-150,1e150},{1e-250,1e-250},{1e250,1e250}}){
        const double r=pair.first,d=pair.second;
        const float encoded=floatJacobi::encode(r,static_cast<float>(d));
        assert(std::isnan(encoded));
        assert(floatJacobi::decode(encoded,r,d)==r/d);
    }
    const float ordinary=floatJacobi::encode(7.,3.f);
    assert(std::isfinite(ordinary)&&ordinary==7.f/3.f);
    assert(floatJacobi::decode(ordinary,7.,3.)==double(ordinary));
    assert(std::isinf(floatJacobi::decode(floatJacobi::encode(1.,0.f),1.,0.)));
    const double nan=std::numeric_limits<double>::quiet_NaN();
    assert(std::isnan(floatJacobi::decode(floatJacobi::encode(nan,1.f),nan,1.)));
}

static void solve(int n,double scale,bool heterogeneous,bool interfacePath){
    std::vector<double> mass(n),conductivity(n-1),exact(n),initial(n),x(n),rhs(n),rowSum(n);
    for(int i=0;i<n;++i){
        mass[i]=scale*(heterogeneous?std::pow(10.,(i%7-3)*3.):1.+.1*(i%5));
        exact[i]=300.+std::sin(.17*i);initial[i]=x[i]=299.+.01*(i%11);
    }
    for(int i=0;i<n-1;++i)conductivity[i]=.15*std::sqrt(mass[i])*std::sqrt(mass[i+1]);
    std::vector<int> offsets(1,0),columns;std::vector<double> values;
    for(int i=0;i<n;++i){
        columns.push_back(i);values.push_back(mass[i]+(i?conductivity[i-1]:0)+(i+1<n?conductivity[i]:0));
        if(i){columns.push_back(i-1);values.push_back(-conductivity[i-1]);}
        if(i+1<n){columns.push_back(i+1);values.push_back(-conductivity[i]);}
        offsets.push_back(int(values.size()));
        for(int e=offsets[i];e<offsets[i+1];++e){rowSum[i]+=values[e];rhs[i]+=values[e]*exact[columns[e]];}
    }
    SparseMatrixView matrix{n,int(values.size()),offsets.data(),columns.data(),values.data(),rowSum.data()};
    SparseCommunication communication;
    if(interfacePath){communication.globalSum=[](void*,double*,int){};
        communication.addInterface=[](void*,const double*,double*,int){};}
    ConjugateGradientControls controls;controls.tolerance=1e-12;controls.maximumIterations=4000;
    // Scale the dimensional norm floor with A and b; the nondimensional
    // accuracy target stays identical across the range-continuation fixtures.
    controls.normalizationSmall=std::abs(scale)*1e-20;
    SparseConjugateGradient solver;const auto report=solver.solve(matrix,rhs.data(),x.data(),controls,communication);
    double mean=0;for(double value:initial)mean+=value;mean/=n;
    double norm=controls.normalizationSmall,trueL1=0,signedResidual=0,inventoryDefect=0,sourceL1=0;
    for(int i=0;i<n;++i){
        double before=0,after=0;
        for(int e=offsets[i];e<offsets[i+1];++e){before+=values[e]*initial[columns[e]];after+=values[e]*x[columns[e]];}
        norm+=std::abs(before-rowSum[i]*mean)+std::abs(rhs[i]-rowSum[i]*mean);
        trueL1+=std::abs(rhs[i]-after);signedResidual+=rhs[i]-after;
        inventoryDefect+=mass[i]*(exact[i]-x[i]);sourceL1+=std::abs(rhs[i]);
    }
    assert(report.converged&&!report.singular);
    assert(report.physicalResidualEvaluations>=2);
    // Independent physical b-A*x check, not the recursive Krylov residual.
    assert(trueL1/norm<controls.tolerance);
    assert(std::abs(report.finalResidual-trueL1/norm)<1e-15);
    // Interior face powers cancel; only diagonal inventory remains globally.
    assert(std::abs(signedResidual-inventoryDefect)<1e-12*sourceL1);
    if(scale<1e-40||scale>1e40)assert(report.preconditionerFallbackApplications>0);
    std::cout<<n<<' '<<scale<<' '<<heterogeneous<<' '<<interfacePath<<' '
             <<report.iterations<<' '<<report.finalResidual<<' '<<report.preconditionerFallbackApplications<<' '
             <<report.physicalResidualEvaluations<<' '<<report.residualRestarts<<'\n';
}
int main(){
    encoding();
    for(int n:{7,513})for(bool variable:{false,true})for(bool interfaces:{false,true})
        solve(n,1.,variable,interfaces);
    for(double scale:{1e-150,1e150})for(bool interfaces:{false,true})solve(7,scale,false,interfaces);
    std::cout<<"PASS: FP32 preconditioning, FP64 residual/conservation and arithmetic continuation\n";
}
