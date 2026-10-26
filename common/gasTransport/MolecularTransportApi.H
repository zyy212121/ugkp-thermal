#pragma once
#include "MolecularTransportIO.H"
// Typed in-process initialization; borrowed host configuration is copied once.
extern "C" int ugkwpGpuResidentStrictConfigureMolecularTransport(
    void*,const ugkwp::MolecularTransportConfiguration*);
