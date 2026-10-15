// Independent serial 8x8 reference for the production eight-lane explicit solve.
// Constant solid properties make the stability partition independently known.
double reference_sum8(const double* v) {
 double a[8];std::copy(v,v+8,a);
 for(int d:{4,2,1})for(int j=0;j<8-d;++j)a[j]+=a[j+d];
 return a[0];
}
void full_2d_interval(bool tinyTail,bool reverse=false,bool negativeGas=false,bool negativeResistance=false) {
 DeviceState s;seed(s);if(negativeResistance)s.coldWallSolidificationParameters.interfaceResistanceM2KW=GPU_R(-1);const auto c=law(s);
 const GpuReal volume=GPU_R(1.e-12),mass=GPU_R(3.e-9),maximum=GPU_R(2.2e-8),area=GPU_R(1.e-8);
 const GpuReal wallT=GPU_R(900),gasT=GPU_R(1100),areaFactor=GPU_R(1.e-5),gasG=negativeGas?GPU_R(-1.e-5):GPU_R(1.e-5);
 s.pm[0]=GPU_R(7)*mass;s.pT[0]=GPU_R(1000);s.pThermalEnthalpyResidual[0]=.07125;
 for(int r=0;r<8;++r){s.cold2DAge[r]=.002+r*.0001;for(int z=0;z<8;++z)s.cold2DH[8*r+z]=float(c.enthalpy(1000.+r+z));}
 const double oldCarry=s.pThermalEnthalpyResidual[0];
 const auto geometry=coldWall2DGeometry(volume,mass,maximum);
 const double k=s.coldWallSolidificationParameters.solidThermalConductivityWmK;
 const double cp=s.coldWallSolidificationParameters.solidSpecificHeatJkgK;
 const double axial=(GPU_R(2)*k*k/(k+k+GPU_REAL_MIN))*geometry.ringAreaM2/geometry.axialNodeThicknessM;
 double radial[7],wet[8],wallG[8],oldH[64],h[64],oldAge[8],wallE[8]{},gasE[8]{};
 for(int r=0;r<7;++r)radial[r]=coldWall2DRadialFaceConductanceWK(geometry,r,GpuReal(k),GpuReal(k));
 double stable=DBL_MAX;
 for(int r=0;r<8;++r){
  wet[r]=coldWall2DRingWetAreaM2(geometry,area,r);
  wallG[r]=wet[r]>0.?areaFactor*wet[r]/(GPU_R(.5)*geometry.axialNodeThicknessM/k+s.coldWallSolidificationParameters.interfaceResistanceM2KW+GPU_REAL_MIN):0.;
  oldAge[r]=s.cold2DAge[r];
  for(int z=0;z<8;++z){
   const double conductance=(z?axial:0.)+(z<7?axial:0.)+(r?radial[r-1]:0.)+(r<7?radial[r]:0.)+(z==0?wallG[r]:0.)+(z==7?gasG/8.:0.);
   stable=std::min(stable,GPU_R(.45)*geometry.nodeMassKg*cp/conductance);
   h[8*r+z]=oldH[8*r+z]=double(s.cold2DH[8*r+z])+oldCarry;
  }
 }
 const double dt=stable*(reverse?-.25:(tinyTail?300./(1.-5.e-13):320.25));
 // Long-double remaining time is independent bookkeeping, never a larger step.
 long double remaining=std::abs(dt);std::vector<double> referenceTimes;
 while(remaining>0.) {
  const double step=(reverse?-1.:1.)*(remaining<stable?double(remaining):stable);
  referenceTimes.push_back(step);double temperature[64],power[64]{};
  for(int j=0;j<64;++j)temperature[j]=c.temperature(h[j]);
  for(int r=0;r<8;++r)for(int z=0;z<8;++z) {
   const int j=8*r+z;
   if(z)power[j]+=axial*(temperature[j-1]-temperature[j]);
   if(z<7)power[j]+=axial*(temperature[j+1]-temperature[j]);
   if(r)power[j]+=radial[r-1]*(temperature[j-8]-temperature[j]);
   if(r<7)power[j]+=radial[r]*(temperature[j+8]-temperature[j]);
   if(z==0){const double q=wallG[r]*(temperature[j]-wallT);power[j]-=q;wallE[r]+=q*step;}
   if(z==7){const double q=(gasG/8.)*(gasT-temperature[j]);power[j]+=q;gasE[r]+=q*step;}
  }
  for(int j=0;j<64;++j)h[j]+=step*power[j]/geometry.nodeMassKg;
  remaining-=std::abs(step);
 }
 if(!reverse)require(referenceTimes.size()>256,"reference genuinely exceeds old cap");
 if(tinyTail)require(referenceTimes.size()==301&&referenceTimes.back()>0.&&referenceTimes.back()<1.e-12*dt,"reference includes real positive tail");
 const GpuReal refWall=GpuReal(reference_sum8(wallE));
 if(negativeResistance)require(refWall<0.,"finite signed resistance produces actual negative wall heat");
 const GpuReal representedWall=(s.pm[0]/mass)*refWall;
 const double increment=reference_sum8(gasE)/double(mass)-double(representedWall)/double(s.pm[0]);
 double localOld[8]{},localCandidate[8]{},localChange[8]{};
 for(int r=0;r<8;++r)for(int z=0;z<8;++z){localOld[r]+=double(s.cold2DH[8*r+z])/64.;localCandidate[r]+=h[8*r+z]/64.;}
 const double oldMean=reference_sum8(localOld)+oldCarry;
 const double correction=(oldMean-reference_sum8(localCandidate))+increment;
 float expectedNodes[64];
 for(int r=0;r<8;++r)for(int z=0;z<8;++z){int j=8*r+z;expectedNodes[j]=ugkwp::coldWallStoredEnthalpyFloor(h[j]+correction);localChange[r]+=(double(expectedNodes[j])-double(s.cold2DH[j]))/64.;}
 const double storageChange=reference_sum8(localChange);
 const double expectedCarry=(oldCarry-storageChange)+increment;
 const double expectedBudget=double(s.pm[0])*(storageChange+(expectedCarry-oldCarry))+double(representedWall);
 observedStepTimes.clear();GpuReal mean=-999.,surface=-999.,frozen=-999.,wall=-999.;
 require(launch8([&]{GpuReal m=-999.,t=-999.,f=-999.,w=-999.;
  if(!advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,volume,mass,maximum,area,dt,wallT,GPU_R(10000),areaFactor,gasT,gasG,m,t,f,w))throw DeviceTrap{};
  if(threadIdx.x==0){mean=m;surface=t;frozen=f;wall=w;}
 })==0,"actual eight-lane complete >256 interval");
 require(observedStepTimes.size()==referenceTimes.size(),"real stable steps plus final tail executed");
 for(std::size_t j=0;j<observedStepTimes.size();++j)near(observedStepTimes[j],referenceTimes[j],8.e-15*std::abs(dt),"same stability partition");
 for(int j=0;j<64;++j)near(s.cold2DH[j],expectedNodes[j],0.,"complete 64-node profile");
 near(s.pThermalEnthalpyResidual[0],expectedCarry,3.e-9,"full interval FP64 carry");
 for(int r=0;r<8;++r)near(s.cold2DAge[r],oldAge[r]+(wet[r]>0.?dt:0.),0.,"complete wetted ring ages");
 near(frozen,double(area),1.e-14,"solid connected frozen footprint");near(s.pCold2DFrozenArea[0],double(area),0.,"frozen footprint cache");
 double published=inventory(expectedNodes,64,expectedCarry),top=0.;for(int r=0;r<8;++r)top+=double(expectedNodes[8*r+7])/8.;
 near(mean,GpuReal(c.temperature(published)),1.e-9,"full mean temperature cache");
 near(surface,GpuReal(c.temperature(top+expectedCarry)),1.e-9,"full top temperature cache");
 near(wall,refWall,1.e-16,"full actual wall settlement");near(s.heatBudget[0],expectedBudget,1.e-16,"once-only actual gas settlement");
 near(double(s.pm[0])*(inventory(s.cold2DH,64,s.pThermalEnthalpyResidual[0])-oldMean)+double(representedWall),s.heatBudget[0],3.e-17,"inventory plus wall equals gas");
 std::printf("steps=%zu tail=%.17g dt=%.17g wall=%.17g gas=%.17g carry=%.17g\n",observedStepTimes.size(),observedStepTimes.back(),dt,double(wall),s.heatBudget[0],s.pThermalEnthalpyResidual[0]);
}
