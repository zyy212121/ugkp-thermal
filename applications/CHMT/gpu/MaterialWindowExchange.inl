bool collectMaterialWindowExchange(Backend& backend,MaterialWindowExchange& exchange,std::string& error){
    const auto& surface=backend.accepted.surface;
    const auto faces=surface.solidCell.size();
    if(!backend.history.complete()||!backend.historyArchive||faces!=exchange.removed.size()
        ||surface.solidFace.size()!=faces){
        error="material window requires complete accepted gas exchange";return false;
    }
    std::vector<unsigned> receivers(faces);
    bool valid=true;
    backend.historyArchive->forEachSurfaceFrame([&](const CoupledSurfaceDownload* rows,std::size_t count){
        if(!valid)return;
        if(count!=faces){error="accepted material face layout changed";valid=false;return;}
        for(std::size_t f=0;f<count;++f){
            const auto& row=rows[f];receivers[f]=row.radiationReceiver;
            if(!exchange.append(f,row.primary,surface,error)){
                valid=false;return;
            }
            if(hasPacketTransfer(row.pore)&&!exchange.append(f,row.pore,surface,error)){
                valid=false;return;
            }
            if(backend.particles.active()&&row.particleWallEnergy!=0
                &&!exchange.append(f,materialHistoryPacket(row,2),surface,error)){
                valid=false;return;
            }
            if(!exchange.radiation(f,row.radiationEnergy,row.radiationReceiver,surface,error)){
                valid=false;return;
            }
        }
    });
    if(!valid)return false;
    const auto external=backend.history.externalRadiationDevice();
    auto energy=backend.history.externalRadiationEnergy();
    if(external.faces){
        if(external.faces!=faces||!external.energy){
            error="accepted device radiation face layout mismatch";return false;
        }
        energy.resize(faces);
        const auto stream=static_cast<cudaStream_t>(external.producerStream);
        if(!backend.executionFault.check(cudaMemcpyAsync(energy.data(),external.energy,
                faces*sizeof(decltype(energy)::value_type),cudaMemcpyDeviceToHost,stream),"read accepted window radiation")
            ||!backend.executionFault.check(cudaStreamSynchronize(stream),"collect accepted window radiation")){
            error=backend.executionFault.message;return false;
        }
    }
    if(!energy.empty()&&energy.size()!=faces){error="accepted radiation face layout mismatch";return false;}
    for(std::size_t f=0;f<energy.size();++f)if(!exchange.radiation(f,energy[f],receivers[f],surface,error))return false;
    error.clear();return true;
}
