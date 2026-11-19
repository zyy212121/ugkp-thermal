#pragma once
#include "MolecularTransportIO.H"

extern "C" int ugkwpGpuResidentStrictConfigureMolecularTransport(
    void*,const ugkwp::MolecularTransportConfiguration*);
