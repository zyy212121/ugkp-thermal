int interval(double multiple){
DeviceState s;seed(s);const auto c=law(s);
 const GpuReal volume=GPU_R(1e-12),mass=GPU_R(3e-9),maximum=GPU_R(2.2e-8),area=GPU_R(1e-8),wallT=900,gasT=1100,areaFactor=GPU_R(1e-5),gasG=GPU_R(1e-5);
 s.pm[0]=GPU_R(7)*mass;s.pT[0]=1000;s.pThermalEnthalpyResidual[0]=.07125;
 for(int r=0;r<8;r++){s.cold2DAge[r]=.002+r*.0001;for(int z=0;z<8;z++)s.cold2DH[8*r+z]=float(c.enthalpy(1000.+r+z));}
 auto g=coldWall2DGeometry(volume,mass,maximum);double k=s.coldWallSolidificationParameters.solidThermalConductivityWmK,cp=s.coldWallSolidificationParameters.solidSpecificHeatJkgK;
 double axial=(GPU_R(2)*k*k/(k+k+GPU_REAL_MIN))*g.ringAreaM2/g.axialNodeThicknessM,radial[7],stable=DBL_MAX;
 for(int r=0;r<7;r++)radial[r]=coldWall2DRadialFaceConductanceWK(g,r,GpuReal(k),GpuReal(k));
 for(int r=0;r<8;r++){double wet=coldWall2DRingWetAreaM2(g,area,r),wg=wet>0?areaFactor*wet/(GPU_R(.5)*g.axialNodeThicknessM/k+s.coldWallSolidificationParameters.interfaceResistanceM2KW+GPU_REAL_MIN):0;
  for(int z=0;z<8;z++){double gs=(z?axial:0)+(z<7?axial:0)+(r?radial[r-1]:0)+(r<7?radial[r]:0)+(z==0?wg:0)+(z==7?gasG/8:0);stable=std::min(stable,GPU_R(.45)*g.nodeMassKg*cp/gs);}}
 const double dt=stable*multiple;GpuReal mean=-999,top=-999,frozen=-999,wall=-999;
 int errors=launch8([&]{GpuReal m=-999,t=-999,f=-999,w=-999;bool ok=advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,volume,mass,maximum,area,dt,wallT,GPU_R(10000),areaFactor,gasT,gasG,m,t,f,w);if(!ok)throw DeviceTrap{};if(threadIdx.x==0){mean=m;top=t;frozen=f;wall=w;}});
 printf("multiple=%.17g dt=%.17g errors=%d steps=%zu mean=%a top=%a frozen=%a wall=%a carry=%a gas=%a age=%a\
",multiple,dt,errors,observedStepTimes.size(),double(mean),double(top),double(frozen),double(wall),s.pThermalEnthalpyResidual[0],s.heatBudget[0],s.cold2DAge[0]);
 printf("TIMES");for(double t:observedStepTimes)printf(" %a",t);printf("\
NODES");for(int i=0;i<64;i++)printf(" %a",double(s.cold2DH[i]));printf("\
");
 for(int lane=0;lane<8;++lane){printf("TRACE%d",lane);for(double v:observedTrace[lane])printf(" %a",v);printf("\n");}
 return 0;
}
void dump(const DeviceState& s,int errors){
 printf("errors=%d status=%d stuck=%d face=%d T=%a carry=%a wall=%a gas=%a age=%a contact=%a damage=%a frozen=%a\n",errors,int(s.pStatus[0]),int(s.pStuck[0]),s.pStuckFaceId[0],double(s.pT[0]),s.pThermalEnthalpyResidual[0],double(s.deposited[0]+s.reflected[0]),s.heatBudget[0],s.cold2DAge[0],double(mechanicalAge(s)),double(s.pDepositionArea[0]),double(s.pCold2DFrozenArea[0]));
 for(int j=0;j<64;++j)printf(" %a",double(s.cold2DH[j]));printf("\n");
 for(int j=0;j<8;++j)printf(" %a",double(s.cold2DAge[j]));printf("\n");
 printf("velocity %a %a %a %a %a %a\n",double(s.pux[0]),double(s.puy[0]),double(s.puz[0]),double(s.puxOld[0]),double(s.puyOld[0]),double(s.puzOld[0]));
}
void maskedResistance(){DeviceState s;seed(s);s.coldWallSolidificationParameters.solidThermalConductivityWmK=std::numeric_limits<GpuReal>::denorm_min();const auto c=law(s);for(int j=0;j<64;j++)s.cold2DH[j]=float(c.enthalpy(1000.));for(int j=0;j<8;j++)s.cold2DAge[j]=.001;GpuReal mean=-999,wall=-999;int errors=launch8([&]{GpuReal m=-999,t=-999,f=-999,w=-999;if(!advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,GPU_R(1e-12),GPU_R(3e-9),GPU_R(2e-8),GPU_R(1e-8),1e-10,GPU_R(900),GPU_R(10000),GPU_R(1e-5),GPU_R(1100),GPU_R(0),m,t,f,w))throw DeviceTrap{};if(threadIdx.x==0){mean=m;wall=w;}});printf("group mean=%a wall=%a\n",double(mean),double(wall));dump(s,errors);}
void effusivityWrapper(){DeviceState s;seed(s);s.solveParticleTemperature=0;s.particleGasHeatTransferModelId=0;s.pStuck[0]=particleWallDeposited;s.particleStuckCandidateMask[0]=particleWallColdWall2D;s.pT[0]=GPU_R(1000);s.gasBoundaryT[0]=GPU_R(900);s.pDepositionArea[0]=1e-8f;s.particleWallDensityKgM3=GPU_R(1e200);s.particleWallSpecificHeatJkgK=GPU_R(1e200);s.particleWallConductivityWmK=GPU_R(1e200);for(int j=0;j<64;++j)s.cold2DH[j]=float(coldWallSpecificEnthalpyJkg(GPU_R(1000),s.coldWallSolidificationParameters));for(int j=0;j<8;++j)s.cold2DAge[j]=.001;const int errors=launch8([&]{relaxColdWall2DParticlesToResidentGasKernel(&s,1.e-9);});dump(s,errors);}
void adhesionFinalizer(){DeviceState s;seed(s);s.pT[0]=GPU_R(3000);s.pd[0]=GPU_R(1e-12);s.particleWallAdhesionEnergyScale=GPU_R(1e290);s.pContactMaximumArea[0]=1.f;s.pContactDuration[0]=1.f;s.pContactPeakFraction[0]=.5f;mechanicalAge(s)=.5;s.pStuck[0]=particleWallTransientDeposit;s.pDepositionArea[0]=0;s.pux[0]=1;s.puy[0]=2;s.puz[0]=3;int errors=0;try{finalizeOneThermalizedStuckParticle(s,0,GPU_R(1),GPU_R(0),GPU_R(0),GPU_R(0),GPU_R(0),GPU_R(0));}catch(const DeviceTrap&){errors=1;}dump(s,errors);}

void contactBounds(){
 for(int state:{int(particleWallTransientRebound),int(particleWallTransientDeposit)})for(float damage:{0.f,FLT_MAX/2}){
  DeviceState s;seed(s);s.solveParticleTemperature=0;s.particleGasHeatTransferModelId=0;s.particleWallHeatTransferEnabled=0;s.pStuck[0]=state;s.pContactMaximumArea[0]=FLT_MAX;s.pContactDuration[0]=1;s.pContactPeakFraction[0]=.5;s.pColdFrozenArea[0]=FLT_MAX;s.pDepositionArea[0]=damage;mechanicalAge(s)=1;int errors=0;threadIdx.x=0;try{relaxOneParticleToResidentGas(s,0,0.,0);}catch(const DeviceTrap&){errors=1;}dump(s,errors);
 }
 for(double age:{0.,.004,.01,std::numeric_limits<double>::quiet_NaN()}){
  DeviceState s;seed(s);mechanicalAge(s)=age;threadIdx.x=0;int errors=0;try{accumulateParticleWallRepresentedContactAreaKernel(&s);}catch(const DeviceTrap&){errors=1;}printf("area errors=%d age=%a sum=%a\n",errors,age,double(s.particleWallRepresentedContactArea[0]));
 }
}
void geometryOverflow(){const auto g=coldWall2DGeometry(GPU_REAL_MAX/GPU_R(2),GPU_R(1),GPU_R(1e-20));printf("geometry %d %a %a %a %a %a\n",g.valid,double(g.maximumRadiusM),double(g.filmThicknessM),double(g.axialNodeThicknessM),double(g.ringAreaM2),double(g.nodeMassKg));}

void multiplicityWrapper(bool two){DeviceState s;seed(s);s.solveParticleTemperature=0;s.particleGasHeatTransferModelId=0;s.pStuck[0]=particleWallDeposited;s.particleStuckCandidateMask[0]=two?particleWallColdWall2D:particleWallSolidifyingDeposition;s.pT[0]=GPU_R(1000);s.pd[0]=GPU_R(1e-12);s.pm[0]=GPU_R(1e300);s.gasBoundaryT[0]=GPU_R(900);s.pDepositionArea[0]=1e-8f;for(int j=0;j<64;++j)s.cold2DH[j]=float(coldWallSpecificEnthalpyJkg(GPU_R(1000),s.coldWallSolidificationParameters));for(int j=0;j<8;++j){s.cold2DAge[j]=.001;s.coldH[j]=s.cold2DH[j];}const int errors=launch8([&]{if(two)relaxColdWall2DParticlesToResidentGasKernel(&s,1.e-60);else relaxColdWall1DParticlesToResidentGasKernel(&s,1.e-60);});dump(s,errors);for(int j=0;j<8;++j)printf(" %a",double(s.coldH[j]));printf("\n");}
int main(int argc,char**argv){const std::string mode=argc>1?argv[1]:"interval";if(mode=="interval")return interval(std::atof(argv[2]));if(mode=="resistance")maskedResistance();else if(mode=="effusivity")effusivityWrapper();else if(mode=="adhesion")adhesionFinalizer();else if(mode=="contact")contactBounds();else if(mode=="geometry")geometryOverflow();else if(mode=="multiplicity1d")multiplicityWrapper(false);else if(mode=="multiplicity2d")multiplicityWrapper(true);else return 2;}
