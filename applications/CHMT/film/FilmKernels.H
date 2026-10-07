#ifndef CHMT_FILM_FILMKERNELS_H
#define CHMT_FILM_FILMKERNELS_H
#include "gpu/DeviceViews.H"
#include "film/FilmMath.H"
namespace chmt {
int launchFilmRhs(FilmView, SurfaceView, PacketView, const PhysicsConfig&, Real stageDt,
    DeviceStatus*, void* stream);
int launchFilmValidate(FilmView, SurfaceView, const PhysicsConfig&, DeviceStatus*, void* stream);
}
#endif
