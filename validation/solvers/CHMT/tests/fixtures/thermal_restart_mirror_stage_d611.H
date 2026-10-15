    GpuThermalRestartMirrorMetadata stageThermalRestartMirrors
    (
        const Time& runTime,
        const bool allowThermalCouplingWrite = false
    ) const
    {
        requireInitialised("stageThermalRestartMirrors");
        const label nParticles = writeParticleRestartMirror
        (
            runTime,
            allowThermalCouplingWrite
        );
        int nSources = 0;
        check
        (
            ugkwpGpuResidentStrictDownloadSourceResidualMass
            (
                handle_, &nSources, 0, nullptr, nullptr
            ),
            "query source residual count for thermal restart manifest"
        );
        const fileName particlePath = particleRestartFile(runTime);
        const fileName epsPath = epsGPrevRestartFile(runTime);
        const fileName sourcePath = sourceResidualRestartFile(runTime);
        GpuThermalRestartMirrorMetadata result;
        result.particleFile = particlePath.name();
        result.epsGPrevFile = epsPath.name();
        result.sourceResidualFile = sourcePath.name();
        result.particleCount = nParticles;
        result.epsGPrevCount = hostMirrorCells_;
        result.sourceResidualCount = nSources;
        result.particleSha1 = sha1File(particlePath);
        result.epsGPrevSha1 = sha1File(epsPath);
        result.sourceResidualSha1 = sha1File(sourcePath);
        return result;
    }
