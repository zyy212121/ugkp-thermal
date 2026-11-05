#ifndef UGKP_GPU_SOLID_THERMAL_MATH_H
#define UGKP_GPU_SOLID_THERMAL_MATH_H
#include <cmath>
#ifdef __CUDACC__
#define UGKP_THERMAL_HD __host__ __device__
#else
#define UGKP_THERMAL_HD
#endif
namespace ugkp { namespace gpu {
struct ThermalPropertyTable {
    bool constant=false,clamp=false,allowNonpositive=false;
    double constantValue=0;
    int count=0;
    const double* temperatures=nullptr;
    const double* values=nullptr;
};
enum class ThermalPropertyError { None,Temperature,Value,Product,Secant };
struct ThermalPropertyStatus {
    ThermalPropertyError error=ThermalPropertyError::None;
    int property=-1;
    double temperature=0,value=0;
    unsigned extrapolated=0;
};
UGKP_THERMAL_HD inline double thermalProperty(const ThermalPropertyTable& table,double T,int property,ThermalPropertyStatus& status){
    if(status.error!=ThermalPropertyError::None)return 0;
    if(!std::isfinite(T)||T<0){status.error=ThermalPropertyError::Temperature;status.property=property;status.temperature=T;return 0;}
    double result=table.constantValue;
    if(!table.constant){
        const double* x=table.temperatures;const double* y=table.values;const int n=table.count;
        if((T<x[0]||T>x[n-1])&&table.clamp)result=T<x[0]?y[0]:y[n-1];
        else if(T==x[n-1])result=y[n-1];
        else {
            if(T<x[0]||T>x[n-1])status.extrapolated|=1u<<property;
            int first=0,count=n;
            while(count){const int step=count/2,middle=first+step;if(!(T<x[middle])){first=middle+1;count-=step+1;}else count=step;}
            int lower=T<x[0]?0:T>x[n-1]?n-2:first-1;
            lower=lower<0?0:lower>n-2?n-2:lower;
            const double x0=x[lower],x1=x[lower+1];
            if(T==x0)result=y[lower];
            else if(T==x1)result=y[lower+1];
            else {
                const double width=x1-x0;double fraction=0;
                if(std::isfinite(width))fraction=(T-x0)/width;
                else {const double scale=std::abs(x0)>std::abs(x1)?std::abs(x0):std::abs(x1);fraction=(T/scale-x0/scale)/(x1/scale-x0/scale);}
                result=fraction<=.5?y[lower]+fraction*(y[lower+1]-y[lower]):y[lower+1]+(1-fraction)*(y[lower]-y[lower+1]);
            }
        }
    }
    if(!std::isfinite(result)||(!table.allowNonpositive&&result<=0)){
        status.error=ThermalPropertyError::Value;status.property=property;status.temperature=T;status.value=result;
    }
    return result;
}
UGKP_THERMAL_HD inline double thermalProduct(double a,double b,ThermalPropertyStatus& status){
    const double value=a*b;
    if(status.error==ThermalPropertyError::None&&(!std::isfinite(value)||value<=0)){
        status.error=ThermalPropertyError::Product;status.value=value;
    }
    return value;
}
UGKP_THERMAL_HD inline double thermalSegment(const ThermalPropertyTable* tables,double a,double b,ThermalPropertyStatus& status){
    const double width=b-a;
    const double rho0=thermalProperty(tables[0],a,0,status),rho1=thermalProperty(tables[0],b,0,status);
    const double cp0=thermalProperty(tables[1],a,1,status),cp1=thermalProperty(tables[1],b,1,status);
    const double aa=thermalProduct(rho0,cp0,status),ab=thermalProduct(rho0,cp1,status);
    const double ba=thermalProduct(rho1,cp0,status),bb=thermalProduct(rho1,cp1,status);
    const double weighted=2*aa+ab+ba+2*bb;
    return (width/6)*weighted;
}
UGKP_THERMAL_HD inline double thermalIntegral(const ThermalPropertyTable* tables,const double* knots,int count,
    double lower,double upper,ThermalPropertyStatus& status){
    if(lower==upper)return 0;
    const bool reverse=upper<lower;if(reverse){const double t=lower;lower=upper;upper=t;}
    double value=0,start=lower;
    for(int k=0;k<count;++k)if(knots[k]>lower&&knots[k]<upper){value+=thermalSegment(tables,start,knots[k],status);start=knots[k];}
    value+=thermalSegment(tables,start,upper,status);
    return reverse?-value:value;
}
UGKP_THERMAL_HD inline double thermalSecant(const ThermalPropertyTable* tables,const double* knots,int count,
    double trial,double anchor,ThermalPropertyStatus& status){
    if(trial==anchor)return thermalProduct(thermalProperty(tables[0],anchor,0,status),thermalProperty(tables[1],anchor,1,status),status);
    const double value=thermalIntegral(tables,knots,count,anchor,trial,status)/(trial-anchor);
    if(status.error==ThermalPropertyError::None&&(!std::isfinite(value)||value<=0)){
        status.error=ThermalPropertyError::Secant;status.temperature=trial;status.value=value;
    }
    return value;
}
enum ThermalQueryFields:unsigned { ThermalCapacity=1,ThermalConductivity=2,ThermalEnergy=4,ThermalEmissivity=8,ThermalTemperatureChange=16 };
struct ThermalPropertyQuery {double trial=0,anchor=0;unsigned fields=ThermalCapacity|ThermalConductivity;double previous=0;};
struct ThermalPropertyResult {double capacity=0,conductivity=0,energy=0,emissivity=0,relativeChange=0;ThermalPropertyStatus status;};
UGKP_THERMAL_HD inline ThermalPropertyResult thermalProperties(const ThermalPropertyTable* tables,const double* knots,int count,const ThermalPropertyQuery& query){
    ThermalPropertyResult result;
    if(query.fields&ThermalCapacity)result.capacity=thermalSecant(tables,knots,count,query.trial,query.anchor,result.status);
    if(query.fields&ThermalConductivity)result.conductivity=thermalProperty(tables[2],query.trial,2,result.status);
    if(query.fields&ThermalEnergy)result.energy=thermalIntegral(tables,knots,count,query.anchor,query.trial,result.status);
    if(query.fields&ThermalEmissivity)result.emissivity=thermalProperty(tables[3],query.trial,3,result.status);
    if(query.fields&ThermalTemperatureChange){
        const double a=std::abs(query.trial),b=std::abs(query.previous);
        const double maximum=a>b?a:b;
        result.relativeChange=std::abs(query.trial-query.previous)/(maximum>1?maximum:1);
    }
    return result;
}
}}
#undef UGKP_THERMAL_HD
#endif
