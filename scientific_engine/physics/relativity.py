from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import math

try:
    from scipy.constants import G as _G, c as _C
except Exception:
    _G = 6.67430e-11
    _C = 299_792_458.0

try:
    from astropy.constants import M_sun as _ASTROPY_M_SUN
    SOLAR_MASS_KG = float(_ASTROPY_M_SUN.value)
    ASTROPY_AVAILABLE = True
except Exception:
    # CODATA/IAU-compatible practical fallback for local calculations.
    SOLAR_MASS_KG = 1.98847e30
    ASTROPY_AVAILABLE = False

G = float(_G)
C = float(_C)


@dataclass(frozen=True)
class SchwarzschildStructure:
    event_horizon_radius_expression: str = "r_s = 2GM/c^2"
    singularity_radius_expression: str = "r = 0"
    event_horizon_is_singularity: bool = False
    horizon_description: str = "frontera causal del espacio-tiempo"
    singularity_description: str = "singularidad clásica en el centro"

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


def schwarzschild_radius(mass_kg: float) -> float:
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("mass_kg debe ser un número positivo y finito.")
    return 2.0 * G * mass / (C ** 2)


def solar_masses_to_kg(solar_masses: float) -> float:
    value = float(solar_masses)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("solar_masses debe ser positivo y finito.")
    return value * SOLAR_MASS_KG


def schwarzschild_radius_solar_masses(solar_masses: float) -> float:
    return schwarzschild_radius(solar_masses_to_kg(solar_masses))


def schwarzschild_structure() -> SchwarzschildStructure:
    return SchwarzschildStructure()


def format_schwarzschild_structure() -> str:
    return (
        "En un agujero negro de Schwarzschild, el horizonte de sucesos es una "
        "frontera causal situada en el radio de Schwarzschild, "
        "r_s = 2GM/c^2. La singularidad clásica está en el centro, en r = 0. "
        "Por tanto, el horizonte de sucesos y la singularidad son regiones "
        "distintas: el horizonte no es la singularidad."
    )


def format_schwarzschild_radius(*, mass_kg: float) -> str:
    radius_m = schwarzschild_radius(mass_kg)
    if radius_m >= 1000:
        return (
            f"Para M = {mass_kg:.6g} kg, el radio de Schwarzschild es "
            f"r_s = 2GM/c^2 ≈ {radius_m / 1000:.6g} km."
        )
    return (
        f"Para M = {mass_kg:.6g} kg, el radio de Schwarzschild es "
        f"r_s = 2GM/c^2 ≈ {radius_m:.6g} m."
    )
