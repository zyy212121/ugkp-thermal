#ifndef CHMT_TEST_OWNER_PROGRAM_H
#define CHMT_TEST_OWNER_PROGRAM_H
template<class T>void emit(T x){std::cout<<x<<' ';}void emit(Vec3 v){emit(v.x);emit(v.y);emit(v.z);}
template<class T>void values(const std::vector<T>& v){std::cout<<v.size()<<' ';for(auto x:v)emit(x);}
void programOut(const WallProgram& program){
 std::cout<<program.interval.identity.sequence<<' '<<program.interval.identity.epoch<<' ';emit(program.interval.begin);emit(program.interval.end);
 const auto& s=program.surface;
 values(s.prescribedTopTraction);
 values(s.prescribedPressureGradient);
 values(s.edgeOwnerOffset);
 values(s.edgeNeighbourOffset);
 values(s.gasDistance);
 values(s.solidDistance);
 values(s.gasFace);
 values(s.solidFace);
 values(s.solidCell);
 values(s.edgeOwner);
 values(s.edgeNeighbour);
 values(s.persistentId);
 values(s.edgeLength);
 values(s.area);
 values(s.oldArea);
 values(s.sweptEdgeArea);
 values(s.edgeConormal);
 values(s.centre);
 values(s.normal);
 values(s.meshVelocity);
 values(s.baseVelocity);
 values(s.gasMapOffsets);
 values(s.gasMapFaces);
 values(s.gasMapWeights);
 values(s.normalOffsets);
 values(s.normalCoordinates);
 values(s.normalLayerMass);
 values(s.normalLayerVolume);
 values(s.oldNormalLayerVolume);
 values(s.normalThickness);
 values(s.normalPressure);
 values(s.normalPressureRate);
 values(s.bottomTemperature);
 values(s.topTemperature);
 values(s.bottomHeatFlux);
 values(s.topHeatFlux);
 std::cout<<program.knots.size()<<' ';
 for(const auto& k:program.knots){emit(k.time);std::cout<<k.localTime.sum<<' '<<k.localTime.correction<<' ';values(k.gasPoints);values(k.solidPoints);std::cout<<k.faces.size()<<' ';
  for(const auto& f:k.faces){emit(f.temperature);emit(f.velocity);std::cout<<int(f.primaryKind)<<' ';for(auto x:f.speciesRate)emit(x);for(auto x:f.condensedRate)emit(x);for(auto x:f.poreRate)emit(x);for(auto x:f.poreSweepRate)emit(x);emit(f.radiationFlux);emit(f.solidNormalVelocity);emit(f.normalVelocity);}}
 std::cout<<program.donorPlan.interval.identity.sequence<<' '<<program.donorPlan.interval.identity.epoch<<' ';emit(program.donorPlan.interval.begin);emit(program.donorPlan.interval.end);std::cout<<program.donorPlan.knots.size()<<' ';
 for(const auto& k:program.donorPlan.knots){emit(k.time);std::cout<<k.localTime.sum<<' '<<k.localTime.correction<<' ';std::cout<<k.cumulativeSolid.size()<<' ';
  for(const auto& q:k.cumulativeSolid){for(auto x:q.condensed)emit(x);for(auto x:q.pore)emit(x);for(auto x:q.progress)emit(x);emit(q.energy);emit(q.porosity);}
  std::cout<<k.cumulativeFilm.size()<<' ';for(const auto& q:k.cumulativeFilm){emit(q.mass);emit(q.enthalpy);for(auto x:q.species)emit(x);}}
 std::cout<<'\n';
}

#endif
