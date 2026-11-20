namespace implicitWindowGpu {
using namespace implicitMaterial;
using ugkp::diagnostics::hasFatal;

struct IterationView {
    MaterialMeshView mesh;
    MaterialTransportWorkspace transport;
    const PhysicsConfig* physics;
    const SolidQ* base;
    SolidQ* state;
    const Real* baseVolumes;
    const Real* swept;
    const PhysicalReal* nativeVolume;
    const MaterialWindowCell* exchange;
    const int* incidentOffsets;
    const int* incident;
    const int* orderedFaces;
    Face* faces;
    int faceCount;
    Amount* fixedMass;
    Amount* mass;
    ImplicitWindowStorage::Extents* extents;
    EnergyReal* fixedEnergy;
    EnergyReal* target;
    EnergyReal* conducted;
    PhysicalReal* temperature;
    MaterialCaloric* caloric;
    CellThermal* thermal;
    GpuReal* capacity;
    GpuReal* source;
    GpuReal* conductivity;
    ImplicitWindowMetrics* cellMetrics;
    ImplicitWindowMetrics* faceMetrics;
    const ImplicitWindowControls* controls;
    ImplicitWindowReport* report;
    DeviceContext* diagnostics;
};

__global__ void startCells(IterationView v){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=v.mesh.cells||v.report->initialized||hasFatal(v.diagnostics))return;
    const auto& q=v.base[c];const auto& exchange=v.exchange[c];Amount fixed;
    for(int k=0;k<Nc+Ns;++k){
        const PhysicalReal before=k<Nc?q.condensed[k]:q.pore[k-Nc];
        auto received=exchange.mass[k];received.add(before);const PhysicalReal after=received.value();
        const PhysicalReal roundoff=PhysicalReal(8)*std::numeric_limits<PhysicalReal>::epsilon()*absValue(before);
        if(!finite(after)||after<-maxValue(roundoff,PhysicalReal(v.physics->tolerances.absoluteMass))){
            fail(v.diagnostics,38521,EntityKind::Cell,c,after);return;
        }
        fixed[k]=maxValue(PhysicalReal(0),after);
    }
    for(int r=0;r<Nr;++r)fixed[Nc+Ns+r]=q.progress[r];
    auto energy=exchange.energy;energy.add(q.energy);
    MaterialPrimitive initial;
    if(!finite(energy.value())||!recoverMaterial(q,PhysicalReal(v.baseVolumes[c]),*v.physics,initial)){
        fail(v.diagnostics,38521,EntityKind::Cell,c,energy.value());return;
    }
    v.fixedMass[c]=fixed;v.fixedEnergy[c]=energy.value();v.temperature[c]=initial.temperature;
    v.extents[c]={};v.conducted[c]=0;
}

__global__ void clearFaces(Face* faces,int n,const ImplicitWindowReport* report){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=n||report->initialized)return;
    const auto& old=faces[f];faces[f]=Face{old.left,old.right,old.geometry};
}

__global__ void startLoop(ImplicitWindowReport* report,const DeviceContext* diagnostics,cudaGraphConditionalHandle loop){
    if(blockIdx.x||threadIdx.x)return;
    report->initialized=true;report->capacity=false;
    cudaGraphSetConditional(loop,!report->complete&&!hasFatal(diagnostics));
}

__global__ void selectIteration(const ImplicitWindowReport* report,const DeviceContext* diagnostics,
    bool prepare,cudaGraphConditionalHandle allowed){
    if(!blockIdx.x&&!threadIdx.x)cudaGraphSetConditional(allowed,!hasFatal(diagnostics)
        &&(prepare?!report->linearPrepared:report->linearPrepared&&report->requiresLinearSolve));
}

__global__ void reconstructCells(IterationView v,bool transport,bool reactions){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    auto mass=v.fixedMass[c];
    if(transport)for(int i=v.incidentOffsets[c];i<v.incidentOffsets[c+1];++i){
        const auto& f=v.faces[v.incident[i]];
        if(f.left==c){
            for(int k=0;k<components;++k)mass[k]-=f.transfer[k];
        }else{
            for(int k=0;k<components;++k)mass[k]+=f.transfer[k];
        }
    }
    if(reactions)for(int r=0;r<v.physics->nReactions;++r)
        applyExtent(mass,v.physics->reactions[r],r,v.extents[c].values[r]);
    v.mass[c]=mass;v.cellMetrics[c]={};
}

__global__ void prepareState(IterationView v){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    if(prepareCell(v.mass[c],v.fixedMass[c],v.temperature[c],PhysicalReal(v.mesh.volumes[c]),*v.physics,
        v.state[c],v.caloric[c],v.transport.primitive[c]))
        fail(v.diagnostics,38521,EntityKind::Cell,c,v.temperature[c]);
}

__global__ void porePressure(IterationView v){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    const auto error=materialTransport::pressureCell(v.mesh,*v.physics,c,v.transport);
    if(error!=MaterialTransportError::None)fail(v.diagnostics,38522,EntityKind::Cell,c,int(error));
}

__device__ bool activeFace(const Face& face,const Real* swept){
    return face.right>=0||(swept&&swept[face.geometry]!=0);
}

__global__ void transportRates(IterationView v){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=v.faceCount||hasFatal(v.diagnostics))return;
    const auto& face=v.faces[f];
    if(!activeFace(face,v.swept))return;
    const int slot=face.geometry;
    v.transport.splitFlux[slot]={};
    const auto error=materialTransport::face(v.mesh,v.state,*v.physics,v.swept,v.controls->dt,slot,v.transport);
    if(error!=MaterialTransportError::None)fail(v.diagnostics,38522,EntityKind::Face,slot,int(error));
}

__global__ void massColor(IterationView v,int first,int count,bool darcy,bool diffusion){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||hasFatal(v.diagnostics))return;
    const int slot=v.orderedFaces[first+i];
    auto& f=v.faces[slot];const int physical=f.geometry;
    auto& metrics=v.faceMetrics[slot];metrics={};
    if(!activeFace(f,v.swept))return;
    PhysicalReal spacing=0,area=0;
    if(f.right>=0){
        const auto normal=v.mesh.areaVectors[physical];
        const auto distance=v.mesh.neighbour[physical]>=0
            ?v.mesh.cellCentres[f.right]-v.mesh.cellCentres[f.left]
            :v.mesh.cellCentres[f.right]-v.mesh.cellCentres[f.left]
                +v.mesh.faceCentres[physical]-v.mesh.faceCentres[v.mesh.periodicPartner[physical]];
        spacing=PhysicalReal(dot(distance,normalized(normal)));area=PhysicalReal(mag(normal));
    }
    if(updateMass(f,v.mass[f.left],f.right>=0?&v.mass[f.right]:nullptr,v.mesh.volumes,*v.physics,
        v.transport.primitive,v.transport.faceFlux[physical],v.transport.splitFlux[physical],
        v.transport.faceDiffusionLossLeft[physical]+v.transport.faceDiffusionLossRight[physical],spacing,area,
        v.swept?PhysicalReal(v.swept[physical]):PhysicalReal(0),v.controls->dt,darcy,diffusion,metrics.mass))
        fail(v.diagnostics,38523,EntityKind::Face,physical);
}

__global__ void reactCells(IterationView v){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    if(!updateReactions(v.mass[c],v.extents[c].values,PhysicalReal(v.mesh.volumes[c]),v.temperature[c],
        *v.physics,v.controls->dt,v.cellMetrics[c].mass))
        fail(v.diagnostics,38524,EntityKind::Cell,c);
}

__global__ void thermalCells(IterationView v,bool transport){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    const bool coolingLimit=transport&&v.incidentOffsets[c]!=v.incidentOffsets[c+1];
    if(!prepareThermalCell(v.caloric[c],v.temperature[c],v.thermal[c],coolingLimit)){
        fail(v.diagnostics,38525,EntityKind::Cell,c,v.temperature[c]);return;
    }
    auto target=v.fixedEnergy[c];
    if(transport){
        PhysicalReal poreMass=0;
        for(int s=0;s<Ns;++s)poreMass+=v.state[c].pore[s];
        target+=EnergyReal(v.controls->dt)*EnergyReal(poreMass)
            *EnergyReal(dot(v.physics->gravity,v.transport.poreVelocity[c]));
        for(int i=v.incidentOffsets[c];i<v.incidentOffsets[c+1];++i){
            const auto& f=v.faces[v.incident[i]];
            if(f.left==c)target-=f.energy;else target+=f.energy;
        }
    }
    v.target[c]=target;
}

__global__ void energyColor(IterationView v,int first,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||hasFatal(v.diagnostics))return;
    auto& f=v.faces[v.orderedFaces[first+i]];
    if(!activeFace(f,v.swept))return;
    if(updateEnergy(f,*v.physics,v.temperature,v.thermal,v.conducted,v.target))
        fail(v.diagnostics,38525,EntityKind::Face,f.geometry);
}

__global__ void coefficients(IterationView v,GpuReal* solution){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    const auto& thermal=v.thermal[c];const EnergyReal volume=v.nativeVolume[c];
    const EnergyReal cp=thermal.capacity,anchor=v.temperature[c];
    v.capacity[c]=cp/volume;v.conductivity[c]=v.transport.primitive[c].conductivity;
    v.source[c]=(v.target[c]-thermal.anchor+cp*anchor)/(volume*EnergyReal(v.controls->dt));
    solution[c]=anchor;
}

__global__ void recoverConservativeEnergy(IterationView v,const GpuReal* solution,const GpuReal* flux){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=v.mesh.cells||hasFatal(v.diagnostics))return;
    const PhysicalReal solved=solution[c];
    if(!finite(solved)||solved<=0){fail(v.diagnostics,38526,EntityKind::Cell,c,solved);return;}
    const EnergyReal dt=EnergyReal(v.controls->dt);
    EnergyReal conducted=0,boundary=0;
    for(int i=v.mesh.cellFaceOffsets[c];i<v.mesh.cellFaceOffsets[c+1];++i){
        const int f=v.mesh.cellFaces[i];
        conducted-=dt*EnergyReal(v.mesh.cellFaceSigns[i])*EnergyReal(flux[f]);
        if(v.mesh.neighbour[f]<0&&v.mesh.boundaryKind[f]!=BoundaryKind::Periodic)
            boundary-=dt*EnergyReal(flux[f]);
    }
    v.conducted[c]=conducted;const EnergyReal target=v.target[c],energy=target+conducted;
    auto& q=v.state[c];q.energy=energy;MaterialPrimitive endpoint;
    if(!finite(energy)||!recoverMaterial(q,PhysicalReal(v.mesh.volumes[c]),*v.physics,endpoint)
        ||!finite(endpoint.temperature)||endpoint.temperature<=0){
        fail(v.diagnostics,38526,EntityKind::Cell,c,energy);return;
    }
    q.porosity=endpoint.porosity;
    const EnergyReal defect=absValue(energy-EnergyReal(v.caloric[c].energy(solved)));
    if(!finite(defect)){fail(v.diagnostics,38526,EntityKind::Cell,c,defect);return;}
    const EnergyReal rounding=EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon()*maxValue(absValue(energy),absValue(target));
    const EnergyReal threshold=v.controls->absolute+v.controls->relative*maxValue(EnergyReal(1),absValue(conducted))+rounding;
    auto& metrics=v.cellMetrics[c];
    metrics.energy=defect/maxValue(threshold,std::numeric_limits<EnergyReal>::min());
    metrics.change=absValue(endpoint.temperature-v.temperature[c])/maxValue(PhysicalReal(1),absValue(endpoint.temperature));
    metrics.boundary=boundary;metrics.packets=v.exchange[c].packets;
    v.temperature[c]=endpoint.temperature;
}

__global__ void checkEnergyFaces(IterationView v){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=v.faceCount||hasFatal(v.diagnostics))return;
    const auto& face=v.faces[f];
    if(!activeFace(face,v.swept))return;
    const auto& controls=*v.controls;auto& metrics=v.faceMetrics[f];
    if(!checkFaceEnergy(face,*v.physics,v.temperature,v.state,controls.absolute,controls.relative,metrics.energy)){
        fail(v.diagnostics,38525,EntityKind::Face,face.geometry);return;
    }
    if(face.right<0)metrics.boundary=-face.energy;
}

__device__ void merge(ImplicitWindowMetrics& a,const ImplicitWindowMetrics& b){
    a.mass=maxValue(a.mass,b.mass);a.change=maxValue(a.change,b.change);
    a.energy=maxValue(a.energy,b.energy);a.boundary+=b.boundary;
    a.initialResidual+=b.initialResidual;a.scale+=b.scale;a.packets+=b.packets;
}
__device__ void reduceBlock(ImplicitWindowMetrics value,ImplicitWindowMetrics* result){
    extern __shared__ __align__(16) unsigned char scratch[];
    auto* shared=reinterpret_cast<ImplicitWindowMetrics*>(scratch);
    shared[threadIdx.x]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){
        const int stride=(active+1)/2;
        if(threadIdx.x<active-stride)merge(shared[threadIdx.x],shared[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)*result=shared[0];
}

__global__ void linearParts(IterationView v,const GpuReal* residual,ImplicitWindowMetrics* parts){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    ImplicitWindowMetrics part;
    if(c<v.mesh.cells){
        part.initialResidual=EnergyReal(v.controls->dt)*absValue(EnergyReal(residual[c]));
        part.scale=absValue(v.target[c]);
    }
    reduceBlock(part,parts+blockIdx.x);
}

__global__ void prepareLinear(IterationView v,ImplicitWindowMetrics* parts,int n,
    ImplicitWindowMetrics* total,ugkp::gpu::ConjugateGradientControls* linear){
    ImplicitWindowMetrics value;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)merge(value,parts[i]);
    reduceBlock(value,total);__syncthreads();
    if(threadIdx.x)return;
    const auto residual=total->initialResidual,scale=total->scale;
    if(!finite(residual)||!finite(scale)){fail(v.diagnostics,38527,EntityKind::None,-1,residual);return;}
    const auto& controls=*v.controls;
    const EnergyReal tolerance=controls.absolute+controls.relative*maxValue(EnergyReal(1),residual)
        +EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon()*scale;
    *linear=controls.linear;linear->tolerance=std::numeric_limits<EnergyReal>::min();
    linear->relativeTolerance=residual>0?minValue(EnergyReal(1),tolerance*controls.linearFraction/residual):EnergyReal(1);
    v.report->requiresLinearSolve=linear->relativeTolerance<1;
    v.report->linearPrepared=true;
}

__global__ void resultParts(IterationView v,ImplicitWindowMetrics* parts){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    ImplicitWindowMetrics result;
    if(i<v.mesh.cells)result=v.cellMetrics[i];
    if(i<v.faceCount)merge(result,v.faceMetrics[i]);
    reduceBlock(result,parts+blockIdx.x);
}

__global__ void finishIteration(IterationView v,const ImplicitWindowMetrics* parts,int n,ImplicitWindowMetrics* total,
    const ugkp::gpu::ConjugateGradientReport* linear){
    if(hasFatal(v.diagnostics))return;
    ImplicitWindowMetrics value;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)merge(value,parts[i]);
    reduceBlock(value,total);__syncthreads();if(threadIdx.x)return;
    using namespace ugkp::diagnostics;
    auto& report=*v.report;const auto& controls=*v.controls;
    if(!finite(total->boundary)){
        fail(v.diagnostics,38527,EntityKind::None,-1,total->boundary);return;
    }
    if(report.requiresLinearSolve){
        if(!finite(linear->initialResidual)||!finite(linear->finalResidual)){
            fail(v.diagnostics,38527,EntityKind::None,-1,linear->finalResidual);return;
        }
        ++report.linearSolves;report.linearIterations+=linear->iterations;
        if(linear->finalResidual>linear->targetResidual)recordWarning(v.diagnostics,makeRecord(Code::AccuracyResidual,
            38527,4,linear->iterations,EntityKind::None,-1,-1,linear->finalResidual,linear->targetResidual,0,Quantity::LinearSystem));
    }
    ++report.iterations;report.linearPrepared=false;
    report.massResidual=total->mass;report.energyResidual=total->energy;report.packets=total->packets;
    report.boundaryEnergy=total->boundary;report.temperatureResidual=total->change/controls.changeGoal;
    report.converged=total->mass<=1&&total->energy<=1&&total->change<=controls.changeGoal;
    report.complete=report.converged||report.iterations>=controls.iterations;
    if(report.complete&&!report.converged){
        const auto residual=maxValue(report.massResidual,maxValue(report.energyResidual,report.temperatureResidual));
        recordWarning(v.diagnostics,makeRecord(Code::AccuracyResidual,38528,4,report.iterations,
            EntityKind::None,-1,-1,residual,1));
    }
}

__global__ void selectRecovery(const ImplicitWindowReport* report,const int* pending,
    const DeviceContext* diagnostics,cudaGraphConditionalHandle allowed){
    if(!threadIdx.x)cudaGraphSetConditional(allowed,report->linearPrepared
        &&(!report->requiresLinearSolve||!pending||!*pending)&&!hasFatal(diagnostics));
}

__global__ void continueLoop(ImplicitWindowReport* report,const int* pending,const DeviceContext* diagnostics,
    cudaGraphConditionalHandle loop){
    if(!threadIdx.x){
        report->capacity=report->requiresLinearSolve&&pending&&*pending;
        cudaGraphSetConditional(loop,!report->capacity&&!report->complete&&!hasFatal(diagnostics));
    }
}
}
