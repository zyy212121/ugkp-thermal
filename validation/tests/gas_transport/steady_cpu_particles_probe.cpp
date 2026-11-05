// Compiled after production-header adapter by steady_cpu_particles.py selftest.
int main(){
 ParticleState s;initialise(s);initialiseParticles(s,8);
 s.gasBoundaryP[1]=s.gasBoundaryP[2]=100000;
 // A forced high-probability pool exercises production random draws/correction,
 // not only the zero-collision branch. Theta and selected momentum are retained.
 s.particleCount=2;
 for(int i=0;i<2;++i){s.pStatus[i]=1;s.pCellId[i]=0;s.pm[i]=10;s.pd[i]=.001;s.pux[i]=i?20:-20;s.pT[i]=300;s.pTheta[i]=100;s.pRng[i]=123+i;s.px[i]=0;}
 particleMoments(s);
 const double mass=s.momRhoP[0]*s.V[0],momentum=s.momRhoUPx[0]*s.V[0],energy=s.momRhoEP[0]*s.V[0];
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;clearPoissonThermalPoolKernel(&s);}
 threadIdx.x=0;accumulateParticlePoolAtomicKernel<true>(&s,100.);
 ck(s.poolThermalCount[0]==2,"nonzero production Poisson pool not exercised");
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;preparePoissonPoolSamplingKernel(&s);}
 threadIdx.x=0;samplePoissonPoolParticlesKernel(&s,0);
 for(int i=0;i<2;++i)correctOnePoissonThermalizedParticle(s,i,false);
 particleMoments(s);
 near(s.momRhoP[0]*s.V[0],mass,"pool mass");near(s.momRhoUPx[0]*s.V[0],momentum,"pool momentum");near(s.momRhoEP[0]*s.V[0],energy,"pool energy");
 ck(s.pRng[0]!=123&&s.pRng[1]!=124,"pool RNG not advanced");
 // Empty side patches perform the same geometric reflection, including travel remainder.
 s.particleCount=1;s.pStatus[0]=1;s.pCellId[0]=0;s.px[0]=.25;s.py[0]=.9;s.pz[0]=.5;
 s.pux[0]=s.puxOld[0]=0;s.puy[0]=s.puyOld[0]=2;s.puz[0]=s.puzOld[0]=0;
 trackOneParticleLocalFaceWalk(s,0,.1);
 ck(s.pStatus[0]==1&&s.puy[0]==-2,"empty side patch did not reflect");
 ck(std::abs(s.py[0]-.9)<1e-7,"reflected remainder travel differs");
 // Repeated injection/escape with tiny capacity must recycle survivor storage.
 ParticleState t;initialise(t);initialiseParticles(t,8);t.gasBoundaryP[1]=t.gasBoundaryP[2]=100000;
 t.sourceMassRate[0]=.01;t.injectionParcelMass=.001;t.sourceUx[0]=20;t.sourceT[0]=450;
 t.gasMu=1e-5;t.particleDiameterMin=t.particleDiameterMax=t.particleDiameterFallback=t.sourceD[0]=.01;
 for(int c=0;c<2;++c){t.rho[c]=1;t.rhoUx[c]=20;t.rhoE[c]=100000/(t.gammaGas-1)+200;threadIdx.x=c;recoverGasPrimitivesKernel(&t);}
 for(int n=0;n<100;++n)particleStep(t,.01,0.);
 ck(t.injected>8&&t.outflow>8,"injection exhausted capacity instead of recycling");
 ck(t.particleCount<8,"open-channel survivors overflowed");
 double resident=0;for(int i=0;i<t.particleCount;++i)resident+=t.pm[i];
 near(t.injectedMass-t.outflowMass,resident,"injected/exited/resident particle mass budget");
 near(t.injectedMaterialHeat,t.injectedMass*1000*450,"actual injected material heat budget");
 ck(t.gasMaterialHeat>0,"particle-to-gas heat feedback not exercised");
 if(t.particleCount){ck(t.pT[0]<450,"production particle cooling not exercised");ck(t.referenceAge[0]>0,"active residence age absent");}
 // Frozen-species finite bath: exact production steady paired source conserves
 // represented gas+parcel momentum and formation-inclusive total energy.
 ParticleState m;Input mix;mix.cells=2;mix.local=0;mix.mixture=1;mix.y0=.9;mix.perturbation=0;mix.mu=1e-5;initialise(m,mix);initialiseParticles(m,8);
 m.particleCount=2;
 for(int i=0;i<2;++i){m.pStatus[i]=1;m.pCellId[i]=0;m.pm[i]=.01;m.pd[i]=.001;m.pux[i]=i?30:10;m.pT[i]=i?1000:300;m.pTheta[i]=2;}
 particleMoments(m);m.epsGPrev[0]=m.epsG[0];
 const double vol=m.epsGPrev[0]*m.V[0],sp0=m.gasSpecies.rho[0],sp1=m.gasSpecies.rho[m.nCells];
 auto totalEnergy=[&](){double q=vol*m.rhoE[0];for(int i=0;i<2;++i)q+=m.pm[i]*(.5*sqr3(m.pux[i],m.puy[i],m.puz[i])+1.5*m.pTheta[i]+1000*m.pT[i]);return q;};
 auto totalMomentum=[&](){double q=vol*m.rhoUx[0];for(int i=0;i<2;++i)q+=m.pm[i]*m.pux[i];return q;};
 ck(m.rhoE[0]<0,"test must exercise negative formation-inclusive gas energy");
 const double e0=totalEnergy(),p0=totalMomentum();threadIdx.x=0;
 mixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&m,.01,true,false);
 near(totalEnergy(),e0,"paired mixture formation-inclusive total energy");near(totalMomentum(),p0,"paired mixture total momentum");
 near(m.gasSpecies.rho[0],sp0,"paired mixture species0 conservation");near(m.gasSpecies.rho[m.nCells],sp1,"paired mixture species1 conservation");
 ck(m.pT[0]>300&&m.pT[1]<1000,"paired nonlinear mixture heat path not active");
 ck(m.puxOld[0]==10&&m.puxOld[1]==30,"midpoint tracking old velocity overwritten");
 const double vx0=m.pux[0],vx1=m.pux[1];m.gravityX=2;m.gradPx[0]=0;threadIdx.x=0;
 mixtureParticleExternalKickKernel(&m,.25);
 near(m.pux[0]-vx0,.5,"external kick first active parcel");near(m.pux[1]-vx1,.5,"grid-stride kick second active parcel");
 ck(m.puxOld[0]==10&&m.puxOld[1]==30,"external kick changed midpoint history");
 std::printf("pool_selected=%d injected=%llu outflow=%llu active=%d max_collision_probability=%.17g\n",s.poolThermalCount[0],t.injected,t.outflow,t.particleCount,t.referenceMaxCollisionProbability);
}
