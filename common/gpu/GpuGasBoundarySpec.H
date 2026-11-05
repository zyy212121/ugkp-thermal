#pragma once
#include "word.H"
#include "gasBoundary/Definition.H"
namespace Foam { namespace ugkpboundary {
using PatchGeometry=ugkp::boundary::Geometry;
using ScalarOperator=ugkp::boundary::InputScalar;
using VelocityOperator=ugkp::boundary::Velocity;
inline ScalarOperator scalarOperator(const word& type,bool fixes,bool wave) {
    return ugkp::boundary::scalarOperator(std::string(type),fixes,wave);
}
inline VelocityOperator velocityOperator(const word& type,bool fixes,bool inletOutlet) {
    return ugkp::boundary::velocityOperator(std::string(type),fixes,inletOutlet);
}
}}
