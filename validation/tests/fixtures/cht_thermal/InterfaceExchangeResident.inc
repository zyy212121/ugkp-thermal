// CPU oracle for the resident API boundary used by host-only owner tests.
// The CUDA implementation is independently compared with this expression
// order by test_cht_interface_exchange.py.
 labelList interfaceSolidSizes;
 List<labelList> interfaceDestinations;
 scalarField interfaceAreas;
 void configureInterfaceExchange(const labelList& sizes,const List<labelList>& destination,const scalarField& area)
 {interfaceSolidSizes=sizes;interfaceDestinations=destination;interfaceAreas=area;}
 void previewInterfaceExchange(scalar dt,const scalarField& radiation,List<scalarField>& solid,
     List<scalarField>& gasFlux,List<scalarField>& depositedFlux,List<scalarField>& reflectedFlux)const
 {
  const label pairs=interfaceSolidSizes.size();
  solid.setSize(pairs);gasFlux.setSize(pairs);depositedFlux.setSize(pairs);reflectedFlux.setSize(pairs);
  label offset=0;
  forAll(interfaceSolidSizes,pairI){
   const label size=interfaceDestinations[pairI].size();
   solid[pairI].setSize(interfaceSolidSizes[pairI]);solid[pairI]=0;
   gasFlux[pairI].setSize(size);depositedFlux[pairI].setSize(size);reflectedFlux[pairI].setSize(size);
   for(label i=0;i<size;++i){
    const label slot=offset+i,face=faces[slot];
    const scalar d=particleWallHeatTransferEnabled()?deposited[face]:0;
    const scalar r=particleWallHeatTransferEnabled()?reflected[face]:0;
    scalar combined=gas[face];if(!radiation.empty())combined+=radiation[slot];
    if(particleWallHeatTransferEnabled())combined+=d+r;
    solid[pairI][interfaceDestinations[pairI][i]]+=combined;
    const scalar denominator=interfaceAreas[slot]*dt;
    gasFlux[pairI][i]=gas[face]/denominator;
    depositedFlux[pairI][i]=d/denominator;reflectedFlux[pairI][i]=r/denominator;
   }
   offset+=size;
  }
 }
