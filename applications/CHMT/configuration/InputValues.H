#ifndef CHMT_CONFIGURATION_INPUTVALUES_H
#define CHMT_CONFIGURATION_INPUTVALUES_H
#include "fvCFD.H"
#include "core/HostState.H"
#include "materials/Thermodynamics.H"
#include <stdexcept>
namespace chmt { namespace io {
        inline void require(bool condition, const std::string& message) {
            if (!condition) throw std::runtime_error(message);
        }
        inline Real readNumber(const Foam::dictionary& d, const char* key) {
            const Real v = d.lookup<Foam::scalar>(key);
            require(finite(v), std::string("nonfinite ")+key);
            return v;
        }
        inline Real readNumber(const Foam::dictionary& d, const char* key, Real fallback) {
            return d.found(key)?readNumber(d, key):fallback;
        }
        inline std::string text(const Foam::dictionary& d, const char* key) {
            Foam::token value(d.lookup(key));
            require(value.isWord() || value.isString(), std::string("expected word or quoted string for ")+key);
            return value.isWord()?std::string(value.wordToken()):std::string(value.stringToken());
        }
        inline std::string text(const Foam::dictionary& d, const char* key, const std::string& fallback) {
            return d.found(key)?text(d, key):fallback;
        }
        inline bool flag(const Foam::dictionary& d, const char* key, bool fallback = false) {
            return d.lookupOrDefault<Foam::Switch>(key, Foam::Switch(fallback));
        }
        inline Vec3 readVector(const Foam::dictionary& d, const char* key, Vec3 fallback = {}) {
            if (!d.found(key)) {
                return fallback;
            }
            const Foam::vector v = d.lookup<Foam::vector>(key);
            Vec3 out{v.x(), v.y(), v.z()};
            require(finite(out), std::string("nonfinite ")+key);
            return out;
        }
        inline std::vector<Real> scalars(const Foam::dictionary& d, const char* key) {
            const Foam::scalarList v(d.lookup(key));
            std::vector<Real> out(v.begin(), v.end());
            for (Real x:out)require(finite(x), std::string("nonfinite list ")+key);
            return out;
        }
        template<int N> inline void array(const Foam::dictionary& d, const char* key, Real (&out)[N],
            bool optional = false) {
            if (optional && !d.found(key)) {
                return;
            }
            const auto values = scalars(d, key);
            require(values.size() == N, std::string("wrong count for ")+key);
            std::copy(values.begin(), values.end(), out);
        }
        inline void elements(const Foam::dictionary& d, Real* out, int count) {
            const auto values = scalars(d, "element");
            require(values.size() == static_cast<std::size_t>(count), "element count mismatch");
            std::copy(values.begin(), values.end(), out);
        }
}}
#endif
