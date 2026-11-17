#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace chmt_accept {
using E=decltype(chmt::GasQ{}.energy);
constexpr int bits=sizeof(E)*8;
struct Metric {std::string name,unit;double actual,expected,scale,oldBound,newBound;};
inline std::vector<Metric>& metrics(){static std::vector<Metric> m;return m;}
inline void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
inline double allowance(double scale,double oldBound,unsigned operations){
 return bits==32 ? oldBound+operations*double(std::numeric_limits<E>::epsilon())*scale : oldBound;
}
inline void near(const std::string& name,const std::string& unit,double actual,double expected,
 double scale,double oldBound,unsigned operations){
 const double bound=allowance(scale,oldBound,operations);
 metrics().push_back({name,unit,actual,expected,scale,oldBound,bound});
 std::cout<<std::setprecision(17)<<"CASE "<<name<<" unit="<<unit<<" actual="<<actual<<" expected="<<expected<<" scale="<<scale<<" old_bound="<<oldBound<<" new_bound="<<bound<<std::endl;
 require(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=bound,name);
}
template<class T> inline void exact(const std::string& name,const std::string& unit,const T& a,const T& b){
 metrics().push_back({name,unit,double(a),double(b),std::max(std::abs(double(a)),std::abs(double(b))),0,0});
 std::cout<<std::setprecision(17)<<"CASE "<<name<<" unit="<<unit<<" actual="<<double(a)<<" expected="<<double(b)<<" old_bound=0 new_bound=0"<<std::endl;
 require(std::isfinite(double(a))&&std::memcmp(&a,&b,sizeof(T))==0,name+" exact selected-type bits");
}
inline void finish(const char* family){
 std::cout<<std::setprecision(17)<<"RESULT_JSON={\"family\":\""<<family<<"\",\"precision\":"<<bits
 <<",\"mode\":\"correctness\",\"measurements\":[";
 bool first=true;for(const auto& m:metrics()){if(!first)std::cout<<',';first=false;
 std::cout<<"{\"case\":\""<<m.name<<"\",\"unit\":\""<<m.unit<<"\",\"actual\":"<<m.actual
 <<",\"expected\":"<<m.expected<<",\"scale\":"<<m.scale<<",\"old_bound\":"<<m.oldBound
 <<",\"new_bound\":"<<m.newBound<<'}';}
 std::cout<<"]}\nPASS\n";
}
template<class T,class=void> struct hasCarry:std::false_type{};
template<class T> struct hasCarry<T,std::void_t<decltype(std::declval<T>().specificEnthalpyResidual)>>:std::true_type{};
}
