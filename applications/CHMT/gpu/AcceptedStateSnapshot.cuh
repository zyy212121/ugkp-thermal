
struct AcceptedStateSnapshot::Storage {
    ugkp::gpu::FrozenReadback transfer;
    ugkp::gpu::SnapshotRange<HostGasPublication> gas;
    ugkp::gpu::SnapshotRange<GpuReal> temperature, fraction, pressure, residualMass;
    ugkp::gpu::SnapshotRange<int> sourceFaces;
    ugkp::gpu::SnapshotRange<ugkp::thermal::ParticleState<>> particles;
    bool resident=false, sst=false, particleOwner=false;
    std::uint64_t sequence=0, gasVersion=0, solidVersion=0;
};
AcceptedStateSnapshot::AcceptedStateSnapshot() : data_(new Storage) {}
AcceptedStateSnapshot::~AcceptedStateSnapshot()=default;

void AcceptedStateSnapshot::freeze(Backend& backend,const HostState& state) {
    auto& s=*data_;
    s.sequence=state.commitSequence;
    s.gasVersion=state.gasMesh.geometryVersion;
    s.solidVersion=state.solidMesh.geometryVersion;
    s.resident=!backend.hostEndpointCurrent;
    if (!s.resident) return;
    const auto& view=backend.storage.hostView();
    if (view.nCells<0 || static_cast<std::size_t>(view.nCells)!=state.gasMesh.volumes.size() ||
        view.nFaces<0 || static_cast<std::size_t>(view.nFaces)!=state.gasMesh.owner.size())
        ugkp::gpu::snapshotDataFailure("accepted gas geometry has incomplete cell or face dimensions");
    const auto stream=backend.gasCaptureStream;
    s.sst=view.sstConfigured;
    s.particleOwner=backend.particles.active();
    auto* owner=backend.particles.owner();
    const auto shape=s.particleOwner ? ugkp::thermal::particleSnapshotShape(owner,stream)
                                     : ugkp::thermal::ParticleSnapshotShape{};
    bool wave=false;
    if (state.gasMesh.gasBoundary) {
        const auto& boundary=state.gasMesh.gasBoundary->faces;
        if (boundary.size()!=state.gasMesh.owner.size())
            ugkp::gpu::snapshotDataFailure("accepted wave boundary face dimensions differ");
        wave=std::any_of(boundary.begin(),boundary.end(),[](const auto& face){return face.wave;});
        if (wave && state.gasMesh.boundaryPrimitive.size()!=boundary.size())
            ugkp::gpu::snapshotDataFailure("accepted wave boundary pressure field is incomplete");
    }
    ugkp::gpu::SnapshotLayout layout;
    s.gas=layout.allocate<HostGasPublication>(view.nCells);
    s.temperature=layout.allocate<GpuReal>(view.nCells);
    s.fraction=layout.allocate<GpuReal>(view.gasVolumeFraction ? view.nCells : 0);
    s.pressure=layout.allocate<GpuReal>(wave ? view.nFaces : 0);
    s.particles=layout.allocate<ugkp::thermal::ParticleState<>>(shape.count);
    s.sourceFaces=layout.allocate<int>(shape.sources);
    s.residualMass=layout.allocate<GpuReal>(shape.sources);
    s.transfer.freeze(layout.bytes,stream,[&](unsigned char* bytes) {
        if (view.nCells) {
            const auto width=ugkp::gpu::workPlan().cellBlockThreads;
            packGasWindowCells<<<(view.nCells+width-1)/width,width,0,stream>>>(
                backend.storage.deviceView(),nullptr,view.nCells,backend.model.physics.singleGasSpecies,s.gas.at(bytes));
            ugkp::gpu::checkSnapshotCopy(cudaGetLastError(),"pack accepted gas snapshot");
        }
        s.temperature.copy(bytes,view.Tgas,stream);
        s.fraction.copy(bytes,view.gasVolumeFraction,stream);
        s.pressure.copy(bytes,view.gasBoundaryP,stream);
        s.sourceFaces.copy(bytes,shape.sourceFaces,stream);
        s.residualMass.copy(bytes,shape.sourceResidualMass,stream);
        if (shape.count)
            ugkp::thermal::packParticleSnapshot(owner,shape.count,s.particles.at(bytes),stream);
    });
}

void AcceptedStateSnapshot::collect(HostState& state) {
    auto& s=*data_;
    if (state.commitSequence!=s.sequence || state.gasMesh.geometryVersion!=s.gasVersion ||
        state.solidMesh.geometryVersion!=s.solidVersion)
        throw std::logic_error("accepted snapshot destination belongs to another endpoint");
    if (!s.resident) return;
    const auto* bytes=s.transfer.collect();
    const auto* pressure=s.pressure.at(bytes);
    for (std::size_t f=0;f<s.pressure.count;++f)
        if (state.gasMesh.gasBoundary->faces[f].wave) state.gasMesh.boundaryPrimitive[f].pressure=pressure[f];
    if (s.fraction.count) {
        const auto* fraction=s.fraction.at(bytes);
        state.gasVoidFraction.assign(fraction,fraction+s.fraction.count);
    }
    const auto* gas=s.gas.at(bytes);
    state.gas.resize(s.gas.count);
    state.gasTemperature.resize(s.gas.count);
    if (s.sst) state.sst.resize(s.gas.count);
    for (std::size_t i=0;i<s.gas.count;++i) {
        const auto volume=static_cast<GpuReal>(gasPhaseVolume(state,i));
        state.gas[i]=gas[i].gas*volume;
        state.gasTemperature[i]=s.temperature.at(bytes)[i];
        if (s.sst) state.sst[i]={gas[i].sst.rhoK*volume,gas[i].sst.rhoOmega*volume};
    }
    if (s.particleOwner) {
        state.particles.resize(s.particles.count);
        const auto* raw=s.particles.at(bytes);
        for (std::size_t i=0;i<s.particles.count;++i)
            state.particles[i]=ParticleCouplingStorage::adaptSnapshot(raw[i]);
        state.particleSourceFaces.resize(s.sourceFaces.count);
        state.particleSourceResidualMass.resize(s.sourceFaces.count);
        for (std::size_t i=0;i<s.sourceFaces.count;++i) {
            state.particleSourceFaces[i]=s.sourceFaces.at(bytes)[i];
            state.particleSourceResidualMass[i]=s.residualMass.at(bytes)[i];
        }
    }
}
