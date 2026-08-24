

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import math
from pathlib import Path


_HEADER = ["wavelength_m", "temperature_k", "n", "k"]


class TabulatedOpticalModel:
    

    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.source_sha1 = hashlib.sha1(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("tabulated optical CSV must be UTF-8") from error

        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        except csv.Error as error:
            raise ValueError("tabulated optical CSV syntax is malformed") from error
        if not rows or rows[0] != _HEADER:
            raise ValueError(
                "tabulated optical CSV header must be exactly " + ",".join(_HEADER)
            )
        values = {}
        wavelengths = set()
        temperatures = set()
        for row_index, row in enumerate(rows[1:], start=2):
            if len(row) != 4:
                raise ValueError(f"tabulated optical CSV row {row_index} must have four columns")
            try:
                wavelength_m, temperature_k, n_value, k_value = map(float, row)
            except ValueError as error:
                raise ValueError(
                    f"tabulated optical CSV row {row_index} contains a non-scalar"
                ) from error
            if not all(
                math.isfinite(value)
                for value in (wavelength_m, temperature_k, n_value, k_value)
            ):
                raise ValueError(
                    f"tabulated optical CSV row {row_index} contains a nonfinite value"
                )
            if wavelength_m <= 0.0 or temperature_k <= 0.0:
                raise ValueError("tabulated wavelength and temperature must be positive")
            if n_value <= 0.0 or k_value < 0.0:
                raise ValueError("tabulated optical values require n>0 and k>=0")
            key = (wavelength_m, temperature_k)
            if key in values:
                raise ValueError(f"duplicate tabulated optical grid point {key}")
            values[key] = (n_value, k_value)
            wavelengths.add(wavelength_m)
            temperatures.add(temperature_k)

        self.wavelengths = tuple(sorted(wavelengths))
        self.temperatures = tuple(sorted(temperatures))
        if len(self.wavelengths) < 2 or len(self.temperatures) < 2:
            raise ValueError("tabulated optical grid requires at least two values on each axis")
        expected = len(self.wavelengths) * len(self.temperatures)
        if len(values) != expected or any(
            (wavelength, temperature) not in values
            for wavelength in self.wavelengths
            for temperature in self.temperatures
        ):
            raise ValueError("tabulated optical grid must be complete and rectangular")
        self._values = values

    @staticmethod
    def _bracket(axis, value: float, name: str):
        if not math.isfinite(value) or value < axis[0] or value > axis[-1]:
            raise ValueError(
                f"tabulated optical {name} extrapolation is forbidden; "
                f"requested {value}, range [{axis[0]},{axis[-1]}]"
            )
        if value == axis[-1]:
            return len(axis) - 2, 1.0
        lower = bisect.bisect_right(axis, value) - 1
        weight = (value - axis[lower]) / (axis[lower + 1] - axis[lower])
        return lower, weight

    def validate_domain(
        self,
        wavelength_min_m: float,
        wavelength_max_m: float,
        temperature_min_k: float,
        temperature_max_k: float,
    ) -> None:
        self._bracket(self.wavelengths, wavelength_min_m, "wavelength")
        self._bracket(self.wavelengths, wavelength_max_m, "wavelength")
        self._bracket(self.temperatures, temperature_min_k, "temperature")
        self._bracket(self.temperatures, temperature_max_k, "temperature")

    def refractive_index(self, wavelength_m: float, temperature_k: float) -> complex:
        wavelength_i, wavelength_weight = self._bracket(
            self.wavelengths, wavelength_m, "wavelength"
        )
        temperature_i, temperature_weight = self._bracket(
            self.temperatures, temperature_k, "temperature"
        )

        def interpolate_component(component: int) -> float:
            lower_wavelength = (
                self._values[
                    (self.wavelengths[wavelength_i], self.temperatures[temperature_i])
                ][component]
                + temperature_weight
                * (
                    self._values[
                        (
                            self.wavelengths[wavelength_i],
                            self.temperatures[temperature_i + 1],
                        )
                    ][component]
                    - self._values[
                        (
                            self.wavelengths[wavelength_i],
                            self.temperatures[temperature_i],
                        )
                    ][component]
                )
            )
            upper_wavelength = (
                self._values[
                    (self.wavelengths[wavelength_i + 1], self.temperatures[temperature_i])
                ][component]
                + temperature_weight
                * (
                    self._values[
                        (
                            self.wavelengths[wavelength_i + 1],
                            self.temperatures[temperature_i + 1],
                        )
                    ][component]
                    - self._values[
                        (
                            self.wavelengths[wavelength_i + 1],
                            self.temperatures[temperature_i],
                        )
                    ][component]
                )
            )
            return lower_wavelength + wavelength_weight * (
                upper_wavelength - lower_wavelength
            )

        n_value = interpolate_component(0)
        k_value = interpolate_component(1)
        if not math.isfinite(n_value) or not math.isfinite(k_value) or n_value <= 0.0 or k_value < 0.0:
            raise ValueError("bilinear tabulated optical interpolation is invalid")
        return complex(n_value, -k_value)
