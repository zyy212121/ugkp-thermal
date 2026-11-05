#include "MieTable.H"
#include "OriginalMieTable.H"
#include "OriginalMieDoRadiationMath.H"
#include "gpu/MieOpticalProperties.H"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace ugkp::gpu;
namespace original=Foam::originalGpuThermal;
double asymmetry(const original::MieTable& table,double diameter,double temperature){
    const auto& mu=table.muValues();double denominator=0,moment=0;
    double previous=table.phase(diameter,temperature,mu[0]);
    for(int i=1;i<mu.size();++i){const double current=table.phase(diameter,temperature,mu[i]),delta=mu[i]-mu[i-1];
        denominator+=.5*(previous+current)*delta;moment+=.5*(mu[i-1]*previous+mu[i]*current)*delta;previous=current;}
    return Foam::max(Foam::scalar(-1),Foam::min(Foam::scalar(1),moment/denominator));
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    original::MieTable oracle(argv[1]);Foam::gpuThermal::MieTable table(argv[1]);MieOpticalProperties device(table.exportGpuTable());
    std::vector<MieOpticalQuery> input;
    for(int i=0;i<4099;++i)input.push_back({1e-6*std::exp((i%173)*.04),100+(i%107)*31.,-1.2+(i%89)*.027});
    const double infinity=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(double diameter:{-1.,0.,1e-5,2e-5,.0002,infinity,-infinity,nan})
    for(double temperature:{-1.,300.,700.,3000.,infinity,-infinity,nan})
    for(double mu:{-infinity,-1.,-.5,0.,.3,1.,infinity,nan})input.push_back({diameter,temperature,mu});
    double maximum=0;std::size_t compared=0;
    for(int repeat=0;repeat<3;++repeat){
        if(repeat==1)input.resize(257);if(repeat==2)input.clear();
        std::vector<MieOpticalResult> actual;device.evaluate(input,actual);
        if(actual.size()!=input.size())return 3;
        for(std::size_t i=0;i<input.size();++i){const auto q=input[i];const auto expected=oracle.query(q.diameter,q.temperature);
            const double reference[]={expected.Qabs,expected.Qsca,expected.Qext,expected.planckBandFraction,oracle.phase(q.diameter,q.temperature,q.mu),asymmetry(oracle,q.diameter,q.temperature)};
            const auto a=actual[i];const double values[]={a.absorption,a.scattering,a.extinction,a.band,a.phase,a.asymmetry};
            for(int j=0;j<6;++j){const double x=reference[j],y=values[j];
                if(!std::isfinite(x)||!std::isfinite(y)){if(!((std::isnan(x)&&std::isnan(y))||x==y)){std::cerr<<"IEEE mismatch "<<i<<' '<<j<<' '<<x<<' '<<y<<'\n';return 4;}continue;}
                const double error=std::abs(x-y)/std::max(1.,std::max(std::abs(x),std::abs(y)));maximum=std::max(maximum,error);
                if(error>5e-13){std::cerr<<"numeric mismatch "<<i<<' '<<j<<' '<<x<<' '<<y<<' '<<error<<'\n';return 5;}
                ++compared;
            }
        }
    }
    const double pi=std::acos(-1.);
    const auto equal=[](double a,double b){return (std::isnan(a)&&std::isnan(b))||a==b||
        (std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=5e-13*std::max(1.,std::max(std::abs(a),std::abs(b))));};
    int balanced=0;
    for(int directions:{4,24,260}){
        std::vector<double> weights(directions),cosines(directions*directions),x(directions),y(directions),z(directions);
        double sum=0;
        for(int i=0;i<directions;++i){const double theta=(i+.5)*pi/directions,phi=i*2*pi/1.618033988749895;
            x[i]=std::sin(theta)*std::cos(phi);y[i]=std::sin(theta)*std::sin(phi);z[i]=std::cos(theta);
            weights[i]=std::sin(theta);sum+=weights[i];}
        for(auto& w:weights)w*=original::fourPiConstant()/sum;
        for(int row=0;row<directions;++row)for(int col=0;col<directions;++col)
            cosines[row*directions+col]=Foam::max(-1.,Foam::min(1.,x[row]*x[col]+y[row]*y[col]+z[row]*z[col]));
        Foam::scalarField nativeWeights(directions);for(int i=0;i<directions;++i)nativeWeights[i]=weights[i];
        std::vector<MieOpticalQuery> queries={{1.5e-5,1200.,0},{.0001,700.,0}};
        for(int budget:{0,1,37}){
            const double tolerance=budget==1?nan:1e-12;
            std::vector<double> scaling;std::vector<MiePhaseMetrics> metrics;
            device.balance(queries,weights,cosines,tolerance,budget,scaling,metrics);
            for(std::size_t c=0;c<queries.size();++c){
                Foam::List<Foam::scalarField> kernel(directions);
                for(int row=0;row<directions;++row){kernel[row].setSize(directions);for(int col=0;col<directions;++col)
                    kernel[row][col]=oracle.phase(queries[c].diameter,queries[c].temperature,cosines[row*directions+col])/original::fourPiConstant();}
                Foam::scalarField reference,next;Foam::List<Foam::scalarField> probability;
                const auto expected=original::balanceSymmetricPhaseKernelInPlace(kernel,nativeWeights,tolerance,budget,reference,next,probability);
                const auto actual=metrics[c];
                if(expected.iterations!=actual.iterations||!equal(expected.maxRowError,actual.rowError)
                    ||!equal(expected.maxWeightedColumnRelativeError,actual.columnError)||!equal(expected.maxRelativeScalingCorrection,actual.scalingCorrection)){
                    std::cerr<<"phase metric mismatch directions="<<directions<<" budget="<<budget<<" cell="<<c
                        <<" iterations "<<expected.iterations<<' '<<actual.iterations<<" row "<<expected.maxRowError<<' '<<actual.rowError<<'\n';return 6;}
                for(int row=0;row<directions;++row){if(!equal(reference[row],scaling[c*directions+row]))return 7;
                    for(int col=0;col<directions;++col){const double reconstructed=weights[col]*scaling[c*directions+row]*kernel[row][col]*scaling[c*directions+col];
                        if(!equal(probability[row][col],reconstructed))return 8;}}
                ++balanced;
            }
        }
    }
    std::cout<<"PASS frozen phase balancing, ordered metrics and factored probabilities: "<<balanced<<" cases\n";
    std::cout<<"PASS frozen Mie table / CUDA queries, angular moment, IEEE edges and workspace reuse: "<<compared<<" values; max scaled error "<<maximum<<'\n';
}
