#include "gpu/SparseConjugateGradient.H"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace ugkp::gpu;
int main(){
    SparseConjugateGradient solver;
    for(int side:{1,2,5,13,32})for(bool periodic:{false,true})for(double scale:{1e-12,1.,1e12}){
        const int n=side*side*side;
        std::vector<int> offsets(1,0),columns;
        std::vector<double> values,rowSum(n),rhs(n),expected(n),x(n),old(n);
        for(int i=0;i<n;++i)expected[i]=300+10*std::sin(i*.17);
        for(int i=0;i<n;++i){
            const int position[3]={i%side,(i/side)%side,i/(side*side)},stride[3]={1,side,side*side};
            const int diagonal=values.size();columns.push_back(i);values.push_back(2*scale);
            for(int axis=0;axis<3;++axis)for(int sign:{-1,1}){
                int q=position[axis]+sign;
                if(!periodic&&(q<0||q>=side))continue;
                const int j=i+((q+side)%side-position[axis])*stride[axis];
                // Face-symmetric, heterogeneous conductivity; includes duplicate
                // periodic neighbours and single-cell self-periodic entries.
                const double coefficient=scale*(1+((i+j)%11));
                columns.push_back(j);values.push_back(-coefficient);values[diagonal]+=coefficient;
            }
            offsets.push_back(values.size());
            for(int e=offsets[i];e<offsets[i+1];++e){rowSum[i]+=values[e];rhs[i]+=values[e]*expected[columns[e]];}
            x[i]=old[i]=295+.1*(i%5);
        }
        SparseMatrixView matrix{n,int(values.size()),offsets.data(),columns.data(),values.data(),rowSum.data()};
        ConjugateGradientControls controls;controls.tolerance=1e-12;controls.maximumIterations=2000;
        auto report=solver.solve(matrix,rhs.data(),x.data(),controls);
        assert(report.converged&&!report.singular);
        // Identity communication uses the same recurrence with one host
        // notification per iteration. The local device loop must stop on the
        // identical iteration and preserve every solution/residual bit.
        SparseCommunication serial;serial.globalSum=[](void*,double*,int){};
        std::vector<double> unbatched=old;
        const auto serialReport=solver.solve(matrix,rhs.data(),unbatched.data(),controls,serial);
        assert(x==unbatched&&report.iterations==serialReport.iterations
            &&report.initialResidual==serialReport.initialResidual
            &&report.finalResidual==serialReport.finalResidual);
        double err=0,residual=0,source=0;
        for(int i=0;i<n;++i){err=std::max(err,std::abs(x[i]-expected[i]));double ax=0;
            for(int e=offsets[i];e<offsets[i+1];++e)ax+=values[e]*x[columns[e]];
            residual+=std::abs(rhs[i]-ax);source+=std::abs(rhs[i]);}
        assert(err<2e-9);assert(residual/source<1e-12);
        // An exact input must be accepted without a gratuitous iteration.
        x=expected;
        auto converged=solver.solve(matrix,rhs.data(),x.data(),controls);
        assert(converged.converged&&converged.iterations==0);
        // maxIter remains a report, not a new runtime stop gate.
        if(side>2){x=old;controls.maximumIterations=1;auto shortRun=solver.solve(matrix,rhs.data(),x.data(),controls);assert(shortRun.iterations==1&&!shortRun.converged);}
        if(side==5){for(int minimum:{0,3,9})for(int maximum:{1,7,11}){
            controls.minimumIterations=minimum;controls.maximumIterations=maximum;
            controls.relativeTolerance=.25;controls.tolerance=1e-12;
            x=old;unbatched=old;
            const auto batched=solver.solve(matrix,rhs.data(),x.data(),controls);
            const auto sequential=solver.solve(matrix,rhs.data(),unbatched.data(),controls,serial);
            assert(x==unbatched&&batched.iterations==sequential.iterations
                &&batched.finalResidual==sequential.finalResidual&&batched.converged==sequential.converged);
        }}
        std::cout<<side<<' '<<periodic<<' '<<scale<<' '<<report.iterations<<' '<<err<<' '<<residual/source<<'\n';
    }
    // The communication API also works on empty local partitions.
    int offset=0;SparseMatrixView empty{0,0,&offset,nullptr,nullptr,nullptr};ConjugateGradientControls controls;
    auto result=solver.solve(empty,nullptr,nullptr,controls);assert(result.converged&&result.iterations==0);
    std::cout<<"PASS: 30 scaled structured/periodic solves, reuse, true residual, iteration limit and empty partition\n";
}
