from __future__ import annotations

from typing import Dict, Any

try:
    import astropy
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

from scientific_engine.physics.relativity import (
    SOLAR_MASS_KG,
    schwarzschild_radius_solar_masses,
)


def dependency_status() -> Dict[str, Any]:
    return {
        "astropy_available": ASTROPY_AVAILABLE,
        "solar_mass_kg": SOLAR_MASS_KG,
    }


def black_hole_radius_for_solar_masses(solar_masses: float) -> Dict[str, float]:
    radius_m = schwarzschild_radius_solar_masses(solar_masses)
    return {
        "solar_masses": float(solar_masses),
        "radius_m": radius_m,
        "radius_km": radius_m / 1000.0,
    }
