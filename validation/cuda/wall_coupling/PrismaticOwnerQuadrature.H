#pragma once
#include "common/gasWall/WallGeometryBuilder.H"
#include <numeric>
// Captured orthogonal prism only: the wall-normal section area is V/height.
// This is the production geometric quadrature specialized algebraically to a
// constant-area section, not a new source discretization or fitted measure.
inline ugkwp::gaswall::WallGeometryDescriptor<double> descriptorForPrism(
 const ugkwp::gaswall::WallModelConfig<double>&config,double matchingDistance,
 double ownerHeight,double ownerDistance,double ownerVolume){
 using namespace ugkwp::gaswall;namespace geom=wallGeometryDetail;
 WallGeometryDescriptor<double>d;d.normal={0,1,0};d.tangent1={1,0,0};d.tangent2={0,0,-1};
 d.matchingDistance=matchingDistance;d.ownerDistance=ownerDistance;d.maximumOwnerDistance=ownerHeight;
 d.volume=d.geometryVolume=ownerVolume;d.quadraturePointsAreInterior=false;
 std::vector<double>profile(config.nodes),cuts{0,ownerHeight};
 if(ownerDistance>0&&ownerDistance<ownerHeight)cuts.push_back(ownerDistance);
 for(int i=0;i<config.nodes;++i){profile[i]=matchingDistance*std::pow(double(i)/(config.nodes-1),config.stretch);if(profile[i]>0&&profile[i]<ownerHeight)cuts.push_back(profile[i]);}
 for(int i=1;i<config.nodes;++i){double y=(profile[i-1]+profile[i])/2;if(y>0&&y<ownerHeight)cuts.push_back(y);}
 std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
 std::vector<double>point,weight;geom::gaussRule(8,point,weight);const double first=profile[1],area=ownerVolume/ownerHeight;
 for(size_t i=1;i<cuts.size();++i){double a=cuts[i-1],b=cuts[i];bool near=b<=first;double lo=near?std::sqrt(a/first):a,hi=near?std::sqrt(b/first):b;
  for(int j=0;j<(near?8:2);++j){double x=near?point[j]:(j==0?-1.:1.)/std::sqrt(3.),t=(lo+hi)/2+(hi-lo)/2*x;
   double y=near?first*t*t:t,w=area*(hi-lo)/2*(near?weight[j]*2*first*t:1);
   d.distance.push_back(y);d.volumeWeight.push_back(w);d.quadraturePoints.push_back({0,y,0});}}
 const double sum=std::accumulate(d.volumeWeight.begin(),d.volumeWeight.end(),0.);
 for(size_t i=0;i<d.distance.size();++i){d.volumeWeight[i]*=ownerVolume/sum;d.firstMoment+=d.volumeWeight[i]*d.distance[i];}
 d.firstMomentVector={0,d.firstMoment,0};geom::compressProjectionQuadrature(d,profile);return d;
}
