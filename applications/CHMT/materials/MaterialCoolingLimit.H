#pragma once
#include "materials/MaterialCaloric.H"

namespace chmt::implicitMaterial {
template<int Degree>
CHMT_HD inline EnergyReal capacityPolynomial(const EnergyReal* coefficients,EnergyReal x){
    EnergyReal value=coefficients[Degree];
    for(int i=Degree-1;i>=0;--i)value=value*x+coefficients[i];
    return value;
}

template<int Degree>
CHMT_HD inline int bracketedCapacityRoots(const EnergyReal* coefficients,EnergyReal* roots){
    if constexpr(Degree==1){
        if(coefficients[1]==0)return 0;
        const EnergyReal root=-coefficients[0]/coefficients[1];
        if(!(root>0&&root<1))return 0;
        roots[0]=root;return 1;
    }else{
        if(coefficients[Degree]==0)return bracketedCapacityRoots<Degree-1>(coefficients,roots);
        EnergyReal derivative[Degree],extrema[Degree-1];
        for(int i=0;i<Degree;++i)derivative[i]=EnergyReal(i+1)*coefficients[i+1];
        const int extremaCount=bracketedCapacityRoots<Degree-1>(derivative,extrema);
        int count=0;
        EnergyReal left=0,a=coefficients[0];
        for(int i=0;i<=extremaCount;++i){
            const EnergyReal right=i<extremaCount?extrema[i]:EnergyReal(1);
            const EnergyReal b=capacityPolynomial<Degree>(coefficients,right);
            if(a!=0&&b!=0&&((a<0)!=(b<0))){
                EnergyReal lo=left,hi=right;
                for(int j=0;j<std::numeric_limits<EnergyReal>::digits;++j){
                    const EnergyReal middle=lo+EnergyReal(.5)*(hi-lo);
                    if(middle==lo||middle==hi)break;
                    const EnergyReal c=capacityPolynomial<Degree>(coefficients,middle);
                    if(c==0){lo=hi=middle;break;}
                    if((c<0)==(a<0))lo=middle;else hi=middle;
                }
                roots[count++]=lo+EnergyReal(.5)*(hi-lo);
            }
            if(b==0&&i<extremaCount)roots[count++]=right;
            left=right;a=b;
        }
        return count;
    }
}

CHMT_HD inline EnergyReal lowestCoolingEnergy(const MaterialCaloric& law,EnergyReal temperature,EnergyReal capacity){
    bool nasa=false;
    for(int s=0;s<Ns;++s)nasa|=law.pore[s]>0&&law.gas[s].model==ugkwp::SpeciesThermoModel::NASA7;
    if(!nasa){
        const EnergyReal coldCapacity=law.capacity(EnergyReal(0));
        const EnergyReal scale=maxValue(absValue(coldCapacity),capacity);
        const EnergyReal limit=coldCapacity<0?temperature*((-coldCapacity/scale)/(capacity/scale-coldCapacity/scale)):EnergyReal(0);
        return law.energy(limit);
    }

    EnergyReal upper=temperature;
    for(int branch=0;branch<=Ns;++branch){
        EnergyReal lower=0;
        for(int s=0;s<Ns;++s){
            const auto& gas=law.gas[s];
            if(law.pore[s]>0&&gas.model==ugkwp::SpeciesThermoModel::NASA7
                &&gas.Tmid>lower&&gas.Tmid<upper)lower=EnergyReal(gas.Tmid);
        }
        EnergyReal power[5]={law.a1,law.a2,0,0,0};
        for(int s=0;s<Ns;++s){
            const EnergyReal mass=law.pore[s];if(mass==0)continue;
            const auto& gas=law.gas[s];
            if(gas.model==ugkwp::SpeciesThermoModel::LinearCp){
                power[0]+=mass*speciesCv(gas,EnergyReal(0));power[1]+=mass*EnergyReal(gas.cp1);
            }else{
                const auto* coefficients=gas.nasa7+(upper>gas.Tmid?7:0);
                const EnergyReal gasR=ugkwp::universalGasConstant<EnergyReal>();
                const EnergyReal R=gasR/(gasR/EnergyReal(gas.R));
                power[0]+=mass*R*(EnergyReal(coefficients[0])-EnergyReal(1));
                for(int i=1;i<=4;++i)power[i]+=mass*R*EnergyReal(coefficients[i]);
            }
        }
        const EnergyReal span=upper-lower;
        EnergyReal normalized[5]={power[4],0,0,0,0};
        for(int i=3;i>=0;--i){
            for(int j=4-i;j>0;--j)normalized[j]=lower*normalized[j]+span*normalized[j-1];
            normalized[0]=lower*normalized[0]+power[i];
        }
        if(!(capacityPolynomial<4>(normalized,EnergyReal(1))>0))
            return law.energy(std::nextafter(upper,temperature));
        EnergyReal roots[4];
        const int count=bracketedCapacityRoots<4>(normalized,roots);
        if(count)return law.energy(lower+span*roots[count-1]);
        if(lower==0)return law.energy(EnergyReal(0));
        if(normalized[0]<=0)return law.energy(std::nextafter(lower,upper));
        upper=lower;
    }
    return law.energy(EnergyReal(0));
}
}
