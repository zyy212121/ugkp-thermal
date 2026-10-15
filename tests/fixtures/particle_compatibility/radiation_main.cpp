
int main(){
 DeviceState s;
 for(double cache:{100.,300.,4100.}){s.pT[0]=GpuReal(cache);s.pThermalEnthalpyResidual[0]=.03125;clearMobileParticleRadiationSumsKernel(&s);accumulateMobileParticleRadiationSumsAtomicKernel(&s);printf("snapshot %a %a %a %a carry=%a\n",cache,double(s.radiationMobileMass[0]),double(s.radiationMobileTemperatureMass[0]),double(s.radiationMobileDiameterMass[0]),s.pThermalEnthalpyResidual[0]);}
 const double rows[][4]={{3917.331787109375,4.4000000000000002e-7,.63645399999999996,1506.7985147669101},{834.55799999999999,-2.6e-7,.79153200000000001,3339.4206371441369},{790.82860000000005,-7.0999999999999998e-7,.33108700000000002,-61.833068688043909},{3000,.001,1,0},{3000,.001,.8,200}};
 for(const auto& row:rows){
  s.pT[0]=GpuReal(row[0]);s.pThermalEnthalpyResidual[0]=row[1];
  Foam::gpuThermal::RadiationAffineTemperatureUpdate update{row[2],row[3]};
#ifdef ORIGINAL_RADIATION_PREFLIGHT
  ParticleRadiationValidationError error;
  validateParticleRadiationAffineTemperatureKernel(&s,&update,&error);
  if(error.code!=0)std::abort(); // Every comparison row must really be old-success.
#endif
  applyParticleRadiationAffineTemperatureKernel(&s,&update);
  printf("affine %a %a %a\n",row[0],double(s.pT[0]),s.pThermalEnthalpyResidual[0]);
 }
}
