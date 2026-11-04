#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ugkp { namespace gpu {
struct ReflectionEntry {int donor;double weight;};
// Conservative angular remap. Transfer the quadrature flux measure, not
// unweighted intensity; closest reflected directions receive capacity first.
inline std::vector<std::vector<ReflectionEntry>> specularReflection(
 const std::vector<std::array<double,3>>& directions,const std::vector<double>& weights,
 const std::array<double,3>& normal){
 const int n=int(directions.size());
 if(weights.size()!=directions.size())throw std::invalid_argument("reflection quadrature size mismatch");
 double normalSquared=0;
 for(double x:normal){if(!std::isfinite(x))throw std::invalid_argument("nonfinite reflection normal");normalSquared+=x*x;}
 if(std::abs(normalSquared-1)>1e-12)throw std::invalid_argument("reflection normal must be unit length");
 for(const auto& direction:directions){double squared=0;
  for(double x:direction){if(!std::isfinite(x))throw std::invalid_argument("nonfinite reflection direction");squared+=x*x;}
  if(std::abs(squared-1)>1e-12)throw std::invalid_argument("reflection direction must be unit length");
 }
 std::vector<double> cosine(n),demand(n),supply(n);double incoming=0,outgoing=0;
 for(int i=0;i<n;++i){
  for(int d=0;d<3;++d)cosine[i]+=directions[i][d]*normal[d];
  if(!std::isfinite(cosine[i])||!std::isfinite(weights[i])||weights[i]<=0)
   throw std::invalid_argument("invalid reflection quadrature");
  if(cosine[i]<0)incoming+=(demand[i]=-weights[i]*cosine[i]);
  else outgoing+=(supply[i]=weights[i]*cosine[i]);
 }
 if(!(incoming>0)||!(outgoing>0)||std::abs(incoming-outgoing)>1e-12*std::max(incoming,outgoing))
  throw std::invalid_argument("reflection quadrature hemisphere measures differ");
 // Antipodal quadrature balances both hemispheres analytically. Normalize only
 // the summation roundoff so both row and column capacity loops terminate.
 for(auto& x:supply)x*=incoming/outgoing;
 const auto originalDemand=demand;
 struct Edge {double cost;int receiver,donor;};std::vector<Edge> edges;
 for(int i=0;i<n;++i)if(demand[i]>0)for(int j=0;j<n;++j)if(supply[j]>0){
  double distance=0;for(int d=0;d<3;++d){double dx=directions[i][d]-2*cosine[i]*normal[d]-directions[j][d];distance+=dx*dx;}
  edges.push_back({distance,i,j});
 }
 std::stable_sort(edges.begin(),edges.end(),[](const Edge&a,const Edge&b){return a.cost<b.cost;});
 std::vector<std::vector<ReflectionEntry>> rows(n);
 for(const auto& e:edges){const double transfer=std::min(demand[e.receiver],supply[e.donor]);
  if(transfer<=0)continue;
  rows[e.receiver].push_back({e.donor,transfer/originalDemand[e.receiver]});
  demand[e.receiver]-=transfer;supply[e.donor]-=transfer;
 }
 for(int i=0;i<n;++i)if(demand[i]>1e-12*incoming||supply[i]>1e-12*incoming)
  throw std::runtime_error("reflection capacity allocation incomplete");
 return rows;
}
}}
