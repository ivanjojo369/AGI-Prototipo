from __future__ import annotations

import re
import unicodedata

try:
    import pint
    PINT_AVAILABLE = True
    _UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
except Exception:
    pint = None
    PINT_AVAILABLE = False
    _UREG = None


_UNIT_ALIASES = {
    # longitud
    "m": "meter", "meter": "meter", "meters": "meter", "metre": "meter", "metres": "meter",
    "metro": "meter", "metros": "meter",
    "km": "kilometer", "kilometer": "kilometer", "kilometers": "kilometer",
    "kilometre": "kilometer", "kilometres": "kilometer", "kilometro": "kilometer", "kilometros": "kilometer",
    "cm": "centimeter", "centimeter": "centimeter", "centimeters": "centimeter", "centimetro": "centimeter", "centimetros": "centimeter",
    "mm": "millimeter", "millimeter": "millimeter", "millimeters": "millimeter", "milimetro": "millimeter", "milimetros": "millimeter",
    # tiempo
    "s": "second", "sec": "second", "second": "second", "seconds": "second", "segundo": "second", "segundos": "second",
    "min": "minute", "minute": "minute", "minutes": "minute", "minuto": "minute", "minutos": "minute",
    "h": "hour", "hr": "hour", "hour": "hour", "hours": "hour", "hora": "hour", "horas": "hour",
    # masa
    "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram", "kilogramo": "kilogram", "kilogramos": "kilogram",
    "g": "gram", "gram": "gram", "grams": "gram", "gramo": "gram", "gramos": "gram",
    "mg": "milligram", "milligram": "milligram", "milligrams": "milligram", "miligramo": "milligram", "miligramos": "milligram",
    "t": "tonne", "tonne": "tonne", "tonnes": "tonne", "tonelada": "tonne", "toneladas": "tonne",
    # volumen
    "l": "liter", "liter": "liter", "liters": "liter", "litre": "liter", "litres": "liter", "litro": "liter", "litros": "liter",
    "ml": "milliliter", "milliliter": "milliliter", "milliliters": "milliliter", "mililitro": "milliliter", "mililitros": "milliliter",
}

# factor respecto a una unidad base por dimensión.
_FALLBACK_UNITS = {
    "meter": ("length", 1.0),
    "kilometer": ("length", 1000.0),
    "centimeter": ("length", 0.01),
    "millimeter": ("length", 0.001),
    "second": ("time", 1.0),
    "minute": ("time", 60.0),
    "hour": ("time", 3600.0),
    "kilogram": ("mass", 1.0),
    "gram": ("mass", 0.001),
    "milligram": ("mass", 1e-6),
    "tonne": ("mass", 1000.0),
    "liter": ("volume", 1.0),
    "milliliter": ("volume", 0.001),
}


def _ascii(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").strip().lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def normalize_unit(unit: str) -> str:
    value = _ascii(unit)
    value = value.replace(" por ", "/").replace(" per ", "/")
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*\*\s*", "*", value)
    value = value.strip(" .?¿!¡")

    # Normaliza cada componente de unidades compuestas simples, p. ej. km/h.
    pieces = re.split(r"([/*])", value)
    normalized = []
    for piece in pieces:
        if piece in {"/", "*"}:
            normalized.append(piece)
            continue
        token = piece.strip()
        normalized.append(_UNIT_ALIASES.get(token, token))
    return "".join(normalized)


def _fallback_signature(unit: str) -> tuple[str, float]:
    normalized = normalize_unit(unit)

    if "/" in normalized:
        numerator, denominator = normalized.split("/", 1)
        num_dim, num_factor = _fallback_signature(numerator)
        den_dim, den_factor = _fallback_signature(denominator)
        return f"{num_dim}/{den_dim}", num_factor / den_factor

    if "*" in normalized:
        left, right = normalized.split("*", 1)
        left_dim, left_factor = _fallback_signature(left)
        right_dim, right_factor = _fallback_signature(right)
        return f"{left_dim}*{right_dim}", left_factor * right_factor

    if normalized not in _FALLBACK_UNITS:
        raise RuntimeError(
            f"La unidad '{unit}' requiere Pint o no está soportada por el fallback SI."
        )
    return _FALLBACK_UNITS[normalized]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)

    if PINT_AVAILABLE and _UREG is not None:
        quantity = float(value) * _UREG(source)
        return float(quantity.to(target).magnitude)

    source_dim, source_factor = _fallback_signature(source)
    target_dim, target_factor = _fallback_signature(target)
    if source_dim != target_dim:
        raise ValueError(
            f"Unidades incompatibles: {from_unit} -> {to_unit}."
        )
    return float(value) * source_factor / target_factor
