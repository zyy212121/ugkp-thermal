#pragma once
#include "WallModelTypes.H"
namespace ugkwp {namespace gaswall {namespace detail {
template<class R,class TableReal> UGKWP_GAS_HD inline R nasaCpPolynomial(const TableReal* c,R T){return c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4])));}
template<class R,class TableReal> UGKWP_GAS_HD inline R nasaCpIntegral(const TableReal* c,R start,R delta){
 const R middle=start+delta/R(2),offset=delta*R(.38729833462074168852);
 return delta*(R(5)*(nasaCpPolynomial(c,middle-offset)+nasaCpPolynomial(c,middle+offset))+R(8)*nasaCpPolynomial(c,middle))/R(18);
}
// Branch offsets are immutable species data. Widen only this cancellation-prone
// coefficient difference; callers cache it once per solve, not per residual.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline R speciesEnthalpyBranchJumpAs(int species,const SpeciesThermoView<TableReal,N>& thermo){
 const auto& sp=thermo.species[species];if(sp.model==SpeciesThermoModel::LinearCp)return R(0);
 const TableReal* coefficients=thermo.coefficients+sp.coefficientOffset;
 using Acc=typename std::conditional<(sizeof(R)<sizeof(double)),double,R>::type;
 Acc difference[6];for(int j=0;j<6;++j)difference[j]=Acc(coefficients[j+7])-Acc(coefficients[j]);
 const Acc mid=Acc(sp.midTemperature);
 return R(difference[5]+mid*(difference[0]+mid*(difference[1]/Acc(2)+mid*(difference[2]/Acc(3)+mid*(difference[3]/Acc(4)+mid*difference[4]/Acc(5))))));
}
// Exactly integrates the degree-four NASA7 Cp polynomial (three Gauss points),
// preserving any tabulated enthalpy jump at the branch boundary explicitly.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline R speciesEnthalpyIncrement(int species,R wallT,R theta,const SpeciesThermoView<TableReal,N>& thermo,R branchJump){
 const auto& sp=thermo.species[species];const TableReal* coefficients=thermo.coefficients+sp.coefficientOffset;
 if(sp.model==SpeciesThermoModel::LinearCp)return theta*(coefficients[0]+coefficients[1]*(wallT+theta/R(2)));
 const R gasR=universalGasConstant<R>()/sp.molarMass;
 const bool wallHigh=wallT>sp.midTemperature,targetHigh=wallT+theta>sp.midTemperature;
 if(wallHigh==targetHigh)return gasR*nasaCpIntegral(coefficients+(wallHigh?7:0),wallT,theta);
 const R mid=sp.midTemperature,first=mid-wallT,second=theta-first;
 return gasR*(nasaCpIntegral(coefficients+(wallHigh?7:0),wallT,first)+nasaCpIntegral(coefficients+(targetHigh?7:0),mid,second)+(targetHigh?branchJump:-branchJump));
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline R speciesEnthalpyIncrement(int species,R wallT,R theta,const SpeciesThermoView<TableReal,N>& thermo){
 return speciesEnthalpyIncrement(species,wallT,theta,thermo,speciesEnthalpyBranchJumpAs<R>(species,thermo));
}
template<class TableReal,int N> UGKWP_GAS_HD inline TableReal speciesEnthalpyBranchJump(int species,const SpeciesThermoView<TableReal,N>& thermo){
 return speciesEnthalpyBranchJumpAs<TableReal,N,TableReal>(species,thermo);
}
}}}
