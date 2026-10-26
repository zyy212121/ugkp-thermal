#include "GpuSolidThermalProperties.H"
#include "gpu/SolidThermalProperties.H"
#include "IStringStream.H"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace ugkp::gpu;
int main(){
    for(int model=0;model<6;++model){
        double rx[]={100,300,900,1800},cx[]={200,500,1200,2000},kx[]={100,600,2000},ex[]={100,2000};
        double rv[]={900,1000,1100,1200},cv[]={500,650,800,950},kv[]={2,5,9},ev[]={.2,.8};
        ThermalPropertyTable tables[4];const char* names[] = {"rho","Cp","kappa","emissivity"};
        double* xs[]={rx,cx,kx,ex};double* ys[]={rv,cv,kv,ev};int counts[]={4,4,3,2};
        std::ostringstream text;text<<std::setprecision(17);
        for(int i=0;i<4;++i){
            const bool constant=model==0||(model==1&&i==0)||(model==2&&i==1);
            tables[i]={constant,model==4,i==3,ys[i][0],constant?0:counts[i],xs[i],ys[i]};
            text<<names[i]<<" { type "<<(constant?"constant; value ":"table; outOfBounds ");
            if(constant)text<<ys[i][0]<<"; }\n";
            else {text<<(model==4?"clamp":"error")<<"; values (";for(int k=0;k<counts[i];++k)text<<'('<<xs[i][k]<<' '<<ys[i][k]<<')';text<<"); }\n";}
        }
        Foam::IStringStream input(text.str());Foam::dictionary dictionary(input);Foam::gpuThermal::GpuSolidThermalProperties original(dictionary);
        std::vector<double> knots;
        for(int i=0;i<2;++i)if(!tables[i].constant)knots.insert(knots.end(),xs[i],xs[i]+counts[i]);
        std::sort(knots.begin(),knots.end());knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
        SolidThermalProperties device(tables,knots.data(),knots.size());
        std::vector<ThermalPropertyQuery> queries;
        for(double anchor:{250.,300.,500.,900.,1200.,1800.})for(double trial:{250.,300.,301.,500.,899.,900.,1200.,1700.,1800.})queries.push_back({trial,anchor,31,anchor+7});
        if(model>=3){queries.push_back({50,2100,15});queries.push_back({2100,50,15});}
        std::vector<ThermalPropertyResult> results;device.evaluate(queries,results);
        double maximum=0;
        for(std::size_t i=0;i<queries.size();++i){
            const auto q=queries[i];const auto& value=results[i];assert(value.status.error==ThermalPropertyError::None);
            const double expected[]={original.Csec(q.trial,q.anchor),original.kappa(q.trial),original.deltaHv(q.trial,q.anchor),original.emissivity(q.trial),std::abs(q.trial-q.previous)/std::max({std::abs(q.trial),std::abs(q.previous),1.})};
            const double actual[]={value.capacity,value.conductivity,value.energy,value.emissivity,value.relativeChange};
            for(int j=0;j<(q.fields&ThermalTemperatureChange?5:4);++j){double err=std::abs(expected[j]-actual[j])/std::max(1.,std::abs(expected[j]));maximum=std::max(maximum,err);assert(err<2e-14);}
        }
        queries={{-1,300,15},{300,300,0}};device.evaluate(queries,results);
        assert(results[0].status.error==ThermalPropertyError::Temperature);assert(results[1].status.error==ThermalPropertyError::None);
        std::vector<double> previous(257),history(257);double relaxation=.6;
        for(int iteration=0;iteration<8;++iteration){
            const bool havePrevious=iteration!=0&&iteration!=5;
            std::vector<double> current(257),residual(257),expected(257);
            for(int j=0;j<257;++j){previous[j]=300+.7*j;
                current[j]=previous[j]+(iteration==3?history[j]:(j%3-1)*(.5+.1*iteration));
                residual[j]=current[j]-previous[j];}
            double wanted=relaxation;
            if(havePrevious){double numerator=0,denominator=0;
                for(int j=0;j<257;++j){const double difference=residual[j]-history[j];
                    numerator+=history[j]*difference;denominator+=difference*difference;}
                if(std::isfinite(denominator)&&denominator>1e-300){const double value=-relaxation*numerator/denominator;
                    if(std::isfinite(value))wanted=value;}
                wanted=std::min(std::max(wanted,.1),.9);}
            for(int j=0;j<257;++j)expected[j]=previous[j]+wanted*residual[j];
            relaxation=device.relax(current.data(),previous.data(),257,havePrevious,relaxation,.1,.9,1e-300);
            assert(relaxation==wanted);assert(current==expected);history=residual;
        }
        std::cout<<"PROPERTY_RESULT "<<model<<' '<<maximum<<'\n';
    }
    std::cout<<"PASS independent original CHT properties, GPU tables, stable secants and integrated energy\n";
}
