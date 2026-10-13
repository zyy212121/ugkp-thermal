from pathlib import Path
p=Path('common/operators/computeSstFaceFluxKernel.cuh');s=p.read_text();a='''            // A prior numerical certificate does not veto an evaluated face.
            // Missing consumed inventory/flux backing still cannot be used.
            if(!s.gasSpecies.rho || !s.gasSpecies.flux || !s.gasSpecies.faceStatus)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidStorage);return;}
''';assert a in s;s=s.replace(a,'')
a='''        if(ugkwp::mixtureGasActive(s))
        {
            if(s.gasSpecies.faceStatus[faceI]!=0)''';b='''        if(ugkwp::mixtureGasActive(s))
        {
            // Empty/invalid/unowned incidences consume no face storage. Check
            // each backing only when the original route reaches its read.
            if(!s.gasSpecies.faceStatus)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidStorage);return;}
            if(s.gasSpecies.faceStatus[faceI]!=0)''';assert a in s;s=s.replace(a,b)
a='''            {s.gasSpecies.cellStatus[c]=s.gasSpecies.faceStatus[faceI];if(nativeRefinement)return;}
            GPU_OPERATOR_REAL faceSum''';b='''            {s.gasSpecies.cellStatus[c]=s.gasSpecies.faceStatus[faceI];if(nativeRefinement)return;}
            if(!s.gasSpecies.flux)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidStorage);return;}
            GPU_OPERATOR_REAL faceSum''';assert a in s;s=s.replace(a,b)
a='''        if (ugkwp::mixtureGasActive(s))
        {
            for(int k=0;k<Ns;++k)
            {
                trial[k]=s.gasSpecies.rho''';b='''        if (ugkwp::mixtureGasActive(s))
        {
            if(!s.gasSpecies.rho)
            {s.gasSpecies.cellStatus[c]=int(ugkwp::GasTransportCode::InvalidStorage);return;}
            for(int k=0;k<Ns;++k)
            {
                trial[k]=s.gasSpecies.rho''';assert a in s;s=s.replace(a,b);p.write_text(s)
p=Path('tests/gas_transport/test_gas_advance_view.py');s=p.read_text()
s=s.replace('template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){trace.push_back("courant");}', 'int convectiveModeChecks=0;template<class S>void computeGasConvectiveCourantByCellKernel(S*,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);++convectiveModeChecks;trace.push_back("courant");}')
s=s.replace('template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){}', 'int convectiveModeChecks=0;template<class S>void computeGasConvectiveCourantByCellKernel(S*,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);++convectiveModeChecks;}')
s=s.replace('template<class S>void computeSstStabilityNumberKernel(S*,double,double){trace.push_back("sst-bound");}', 'int sstModeChecks=0;template<class S>void computeSstStabilityNumberKernel(S*,double,double,bool requireWallProfile=true,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);if(!requireWallProfile)std::abort();++sstModeChecks;trace.push_back("sst-bound");}')
s=s.replace('''trace.clear();Policy::reject=1;if(advanceGasTrial<Policy>(&h,.1,2.)!=0||trace.back()!="commit")return 3;
}''','''trace.clear();Policy::reject=1;if(advanceGasTrial<Policy>(&h,.1,2.)!=0||trace.back()!="commit")return 3;
// The original direct trace remains above; execute strict bounds separately.
trace.clear();Policy::reject=0;expectedNativeOwner=true;convectiveModeChecks=0;
if(prepareGasTrialTransport<Policy>(&h,.1,2.,true)!=0||convectiveModeChecks!=1)return 4;
}''')
s=s.replace('''if(position("recoverGasPrimitivesKernel")==trace.size())return 3;
}''','''if(position("recoverGasPrimitivesKernel")==trace.size())return 3;
// Strict post-source launches forward their actual owner/profile mode.
trace.clear();expectedNativeOwner=true;convectiveModeChecks=sstModeChecks=0;
if(prepareGasTrialTransport<Policy>(&h,.01,0.,true)!=0)return 4;
if(position("fresh-mass")==trace.size()||position("sst-bound")==trace.size()
    ||convectiveModeChecks!=1||sstModeChecks!=1)return 5;
}''');p.write_text(s)
p=Path('tests/gas_transport/test_boundary_layer_host.py');s=p.read_text();s=s.replace('template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){}','template<class S>void computeGasConvectiveCourantByCellKernel(S*,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);}')
s=s.replace('template<class S>void computeSstStabilityNumberKernel(S*,double,double){trace.push_back("sst-bound");}','template<class S>void computeSstStabilityNumberKernel(S*,double,double,bool requireWallProfile=true,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);if(!requireWallProfile)std::abort();trace.push_back("sst-bound");}');p.write_text(s)
p=Path('tests/gas_transport/test_legacy_ale_transport.py');s=p.read_text()
s=s.replace('template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*){trace.push_back("unexpected-chemistry");}','template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);trace.push_back("unexpected-chemistry");}')
s=s.replace('template<class S>void computeGasCourantFieldKernel(S*,double){trace.push_back("wave-speed");}','template<class S>void computeGasCourantFieldKernel(S*,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);trace.push_back("wave-speed");}')
s=s.replace('template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){trace.push_back("courant");}','template<class S>void computeGasConvectiveCourantByCellKernel(S*,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);trace.push_back("courant");}')
s=s.replace('template<class S>void computeGasDiffusionNumberKernel(S*,double,double){trace.push_back("diffusion-bound");}','template<class S>void computeGasDiffusionNumberKernel(S*,double,double,bool nativeRefinement=false){checkNativeOwner(nativeRefinement);trace.push_back("diffusion-bound");}');p.write_text(s)
