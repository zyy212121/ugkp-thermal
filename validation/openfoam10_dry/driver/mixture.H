void mixtureCase(Time& runTime,fvMesh& mesh,volScalarField& Y,std::ofstream& out) {
 const double d0=.003,d1=.0012,yl=.1,yr=.9;
 auto correctionDonor=[&](double left,double right){return (d1-d0)*(right-left)<=0?left:right;};
 auto potential=[&](double y){return d0*y+.5*(d1-d0)*y*y;};
 const double fl=potential(yl),fr=potential(yr),exactFlux=fl-fr;
 forAll(Y,i)Y[i]=yl+(yr-yl)*mesh.C()[i].x();Y.correctBoundaryConditions();
 surfaceScalarField gamma(IOobject("gamma",runTime.timeName(),mesh),mesh,dimensionedScalar("D",dimViscosity,d0));
 double change=1;label iter=0;
 for(;iter<500 && change>1e-13;++iter){scalarField old(Y.primitiveField());
  forAll(mesh.neighbour(),f){label l=mesh.owner()[f],r=mesh.neighbour()[f];double yf=correctionDonor(Y[l],Y[r]);gamma[f]=d0*(1-yf)+d1*yf;}
  forAll(mesh.boundary(),p)if(!isA<emptyFvPatch>(mesh.boundary()[p])){const labelUList& cells=mesh.boundary()[p].faceCells();forAll(cells,j){double yf=correctionDonor(Y[cells[j]],Y.boundaryField()[p][j]);gamma.boundaryFieldRef()[p][j]=d0*(1-yf)+d1*yf;}}
  solve(fvm::laplacian(gamma,Y));change=gMax(mag(Y.primitiveField()-old));
 }
 surfaceScalarField j(-gamma*fvc::snGrad(Y)*mesh.magSf());double err=0,fluxErr=0,coreErr=0,sumErr=0;
 forAll(Y,i){double x=mesh.C()[i].x();double dx=mesh.V()[i],b=2*(d1-d0)*(fr-fl),c=d0*d0+2*(d1-d0)*fl;double exact=(-d0+(std::pow(c+b*(x+dx/2),1.5)-std::pow(c+b*(x-dx/2),1.5))/(1.5*b*dx))/(d1-d0);err+=sqr(Y[i]-exact)*mesh.V()[i];}
 forAll(mesh.neighbour(),f){label l=mesh.owner()[f],r=mesh.neighbour()[f];
  dry::PoreFaceState<2> a,b;a.density=b.density=1;a.temperature=b.temperature=300;a.massFraction={{Y[l],1-Y[l]}};b.massFraction={{Y[r],1-Y[r]}};Gas gas;auto flux=dry::poreFaceFlux(a,b,1/mesh.deltaCoeffs()[f],0,{{d0,d1}},0,gas).diffusion;sumErr=max(sumErr,mag(flux[0]+flux[1]));coreErr=max(coreErr,mag(flux[0]*mesh.magSf()[f]-j[f]));fluxErr=max(fluxErr,mag(j[f]-exactFlux));}
 out<<"mode,N,iterations,L2_cellaverage,flux_error,zero_sum,core_difference\nmixture,"<<mesh.nCells()<<','<<iter<<','<<std::sqrt(err)<<','<<fluxErr<<','<<sumErr<<','<<coreErr<<'\n';Y.write();
}
