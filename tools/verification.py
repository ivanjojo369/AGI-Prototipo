# tools/verification.py
from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple


_REASONING_HINTS = (
    "calcula",
    "cuánto",
    "cuanto",
    "determina",
    "regla",
    "restricción",
    "restriccion",
    "razona",
    "razonamiento",
    "analiza",
    "verifica",
    "comprueba",
    "compara",
    "orden",
    "lógica",
    "logica",
    "antes",
    "después",
    "despues",
    "inmediatamente",
    "exactamente",
    "únicamente",
    "unicamente",
    "solo",
    "procedimiento",
    "método",
    "metodo",
    "pasos",
    "explica",
    "explicación",
    "explicacion",
    "demuestra",
    "justifica",
    "por qué",
    "por que",
    "cómo",
    "como",
    "revisa",
    "corrige",
    "muestra",
    "descompón",
    "descompon",
    "transitividad",
    "premisa",
    "conclusión",
    "conclusion",
    "directa",
    "directas",
    "deducida",
    "deducidas",
    "inferencia",
    "inferencias",
)


_EQUATION_RE = re.compile(
    r"(?<![\w.])"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*([+\-*/×xX÷])\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*=\s*"
    r"(-?\d+(?:\.\d+)?)"
)


_NAME = r"[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9_-]*"

# Especificaciones de comparativos transitivos.
# Cada tupla: (patrón positivo, patrón inverso, tipo canónico)
#
# Ejemplo:
#   "Leo es más rápido que Nora" -> ("Leo", "speed_gt", "Nora")
#   "Paula es más lenta que Marta" -> ("Marta", "speed_gt", "Paula")
_COMPARATIVE_SPECS = (
    (
        r"es\s+m[aá]s\s+r[aá]pid[oa]\s+que",
        r"es\s+m[aá]s\s+lent[oa]\s+que",
        "speed_gt",
    ),
    (
        r"es\s+m[aá]s\s+alt[oa]\s+que",
        r"es\s+m[aá]s\s+baj[oa]\s+que",
        "height_gt",
    ),
    (
        r"es\s+m[aá]s\s+pesad[oa]\s+que",
        r"es\s+m[aá]s\s+liger[oa]\s+que",
        "weight_gt",
    ),
    (
        r"es\s+m[aá]s\s+grand[ea]\s+que",
        r"es\s+m[aá]s\s+pequeñ[oa]\s+que",
        "size_gt",
    ),
    (
        r"es\s+m[aá]s\s+larg[oa]\s+que",
        r"es\s+m[aá]s\s+cort[oa]\s+que",
        "length_gt",
    ),
    (
        r"es\s+m[aá]s\s+car[oa]\s+que",
        r"es\s+m[aá]s\s+barat[oa]\s+que",
        "price_gt",
    ),
    (
        r"es\s+m[aá]s\s+viej[oa]\s+que",
        r"es\s+m[aá]s\s+joven\s+que",
        "age_gt",
    ),
    # Inglés básico para los mismos dominios.
    (
        r"is\s+faster\s+than",
        r"is\s+slower\s+than",
        "speed_gt",
    ),
    (
        r"is\s+taller\s+than",
        r"is\s+shorter\s+than",
        "height_gt",
    ),
    (
        r"is\s+heavier\s+than",
        r"is\s+lighter\s+than",
        "weight_gt",
    ),
    (
        r"is\s+larger\s+than",
        r"is\s+smaller\s+than",
        "size_gt",
    ),
    (
        r"is\s+longer\s+than",
        r"is\s+shorter\s+than",
        "length_gt",
    ),
    (
        r"is\s+more\s+expensive\s+than",
        r"is\s+cheaper\s+than",
        "price_gt",
    ),
    (
        r"is\s+older\s+than",
        r"is\s+younger\s+than",
        "age_gt",
    ),
)

_RELATION_PATTERNS = [
    # Español genérico.
    (
        re.compile(
            rf"\b({_NAME})\s+es\s+mayor\s+que\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "gt",
        False,
    ),
    (
        re.compile(
            rf"\b({_NAME})\s+es\s+menor\s+que\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "gt",
        True,
    ),
    # Inglés genérico.
    (
        re.compile(
            rf"\b({_NAME})\s+(?:is|must\s+be)\s+greater\s+than\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "gt",
        False,
    ),
    (
        re.compile(
            rf"\b({_NAME})\s+(?:is|must\s+be)\s+less\s+than\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "gt",
        True,
    ),
    # Símbolos. Se reinterpretan después si el problema tiene
    # un dominio relacional específico (speed_gt, height_gt, etc.).
    (
        re.compile(rf"\b({_NAME})\s*>\s*({_NAME})\b"),
        "gt",
        False,
    ),
    (
        re.compile(rf"\b({_NAME})\s*<\s*({_NAME})\b"),
        "gt",
        True,
    ),
    # Orden temporal/posicional.
    (
        re.compile(
            rf"\b({_NAME})\s+(?:está\s+)?antes\s+(?:que|de)\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "before",
        False,
    ),
    (
        re.compile(
            rf"\b({_NAME})\s+(?:está\s+)?despu[eé]s\s+(?:que|de)\s+({_NAME})\b",
            re.IGNORECASE,
        ),
        "before",
        True,
    ),
]

# Añade automáticamente comparativos de dominio.
for _positive, _negative, _rel_type in _COMPARATIVE_SPECS:
    _RELATION_PATTERNS.append(
        (
            re.compile(
                rf"\b({_NAME})\s+{_positive}\s+({_NAME})\b",
                re.IGNORECASE,
            ),
            _rel_type,
            False,
        )
    )
    _RELATION_PATTERNS.append(
        (
            re.compile(
                rf"\b({_NAME})\s+{_negative}\s+({_NAME})\b",
                re.IGNORECASE,
            ),
            _rel_type,
            True,
        )
    )

_RELATION_PATTERNS = tuple(_RELATION_PATTERNS)


_CHAIN_RE = re.compile(
    rf"({_NAME}(?:\s*[<>]\s*{_NAME}){{1,}})"
)

_NEGATION_MARKERS = (
    "no ",
    "no es",
    "no se",
    "falso",
    "incorrect",
    "contradic",
    "error",
    "inválid",
    "invalid",
    "not ",
    "false",
    "wrong",
)

Relation = Tuple[str, str, str]  # (izquierda, tipo, derecha)


class Verifier:
    """
    Verifier 2.9.2 para QuipuLoop.

    Capacidades:
    1. Detecta contradicciones aritméticas simples de forma determinista.
    2. Detecta errores básicos de clasificación premisa/inferencia
       en relaciones transitivas (p. ej. "Ana es mayor que Diego").
    3. Comprueba si una relación marcada como deducida es realmente
       alcanzable por transitividad desde las premisas explícitas.
    4. Detecta relaciones invertidas respecto a las premisas.
    5. Valida órdenes finales como A > B > C contra el cierre transitivo.
    6. Reconoce cuándo un problema relacional tiene solución total única.
    7. Normaliza comparativos de distintos dominios: velocidad, altura, peso,
       tamaño, longitud, precio y edad, además de mayor/menor y antes/después.
    8. Interpreta símbolos > y < dentro del dominio relacional dominante
       del problema (por ejemplo, velocidad).
    9. Da autoridad al verificador determinista sobre el crítico probabilístico:
       si el problema relacional es completamente verificable y la respuesta
       pasa todas las comprobaciones, NO llama al modelo crítico.
    10. Si una corrección del modelo sigue fallando, la vuelve a validar y puede
        usar una corrección determinista de respaldo.
    """

    def __init__(self, adapter: Any = None) -> None:
        self.adapter = adapter

    @staticmethod
    def _latest_user(messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return ""

    @staticmethod
    def _render_context(messages: List[Dict[str, str]]) -> str:
        rendered: List[str] = []

        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if not content:
                continue

            if role in ("user", "assistant"):
                rendered.append(f"{role.upper()}: {content}")
                continue

            if role == "system" and content.startswith(
                ("[MEMORY]", "[RAG]", "[PLAN]", "[COGNITIVE_CONTROL]")
            ):
                rendered.append(content)

        return "\n\n".join(rendered[-8:])

    @staticmethod
    def _normalize_math(text: str) -> str:
        return (
            text.replace(r"\times", "×")
            .replace(r"\cdot", "×")
            .replace(r"\div", "÷")
        )

    @classmethod
    def _check_arithmetic(cls, answer: str) -> List[str]:
        issues: List[str] = []
        text = cls._normalize_math(answer)

        for match in _EQUATION_RE.finditer(text):
            left_raw, op, right_raw, stated_raw = match.groups()

            try:
                left = Decimal(left_raw)
                right = Decimal(right_raw)
                stated = Decimal(stated_raw)

                if op == "+":
                    expected = left + right
                elif op == "-":
                    expected = left - right
                elif op in ("*", "×", "x", "X"):
                    expected = left * right
                elif op in ("/", "÷"):
                    if right == 0:
                        issues.append(
                            "División por cero detectada: "
                            f"{left_raw} {op} {right_raw}"
                        )
                        continue
                    expected = left / right
                else:
                    continue

                tolerance = Decimal("0.000001")

                if abs(expected - stated) > tolerance:
                    issues.append(
                        "Contradicción aritmética: "
                        f"{left_raw} {op} {right_raw} = {stated_raw}, "
                        f"pero debería ser {expected}."
                    )

            except Exception:
                continue

        return issues

    @staticmethod
    def _looks_like_population_projection(
        user_text: str,
    ) -> bool:
        text = str(user_text or "").strip().lower()

        has_population = bool(
            re.search(
                r"\b(?:poblaci[oó]n|habitantes|personas)\b",
                text,
            )
        )
        has_horizon = bool(
            re.search(
                r"\b(?:a[nñ]o|a[nñ]os|futuro|dentro|proyect|"
                r"habr[aá]|tendr[aá]|tendr[ií]a)\b",
                text,
            )
        )
        asks_quantity = bool(
            re.search(
                r"\b(?:cu[aá]nt[oa]s?|calcula|calcular|estima|"
                r"estimar|proyecta|proyectar|predice|predecir)\b",
                text,
            )
        )

        return bool(
            has_population
            and has_horizon
            and asks_quantity
        )

    @staticmethod
    def _structured_facts_from_messages(
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Fallback: extrae hechos simples renderizados dentro de
        [COGNITIVE_CONTROL] cuando el caller no los pasa explícitamente.
        """
        facts: Dict[str, Any] = {}

        for message in messages:
            if message.get("role") != "system":
                continue

            content = str(
                message.get("content")
                or ""
            )

            if not content.startswith(
                "[COGNITIVE_CONTROL]"
            ):
                continue

            for raw_line in content.splitlines():
                match = re.match(
                    r"\s*-\s*([A-Za-z0-9_.-]+)\s*:\s*(.+?)\s*$",
                    raw_line,
                )
                if not match:
                    continue

                key, raw_value = match.groups()
                value_text = raw_value.strip()

                try:
                    if re.fullmatch(
                        r"-?\d+",
                        value_text,
                    ):
                        value: Any = int(
                            value_text
                        )
                    elif re.fullmatch(
                        r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
                        value_text,
                    ):
                        value = float(
                            value_text
                        )
                    else:
                        value = value_text
                except Exception:
                    value = value_text

                facts[key] = value

        return facts

    @staticmethod
    def _population_projection_solution(
        structured_facts: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        required = (
            "colony.population.initial",
            "colony.population.growth_rate_annual",
            "projection.horizon_years",
        )

        if not isinstance(
            structured_facts,
            dict,
        ):
            return None

        if any(
            key not in structured_facts
            for key in required
        ):
            return None

        try:
            initial = float(
                structured_facts[
                    "colony.population.initial"
                ]
            )
            rate = float(
                structured_facts[
                    "colony.population.growth_rate_annual"
                ]
            )
            years = float(
                structured_facts[
                    "projection.horizon_years"
                ]
            )
        except (TypeError, ValueError):
            return None

        if (
            initial < 0.0
            or years < 0.0
            or rate <= -1.0
        ):
            return None

        try:
            result = initial * (
                (1.0 + rate) ** years
            )
        except Exception:
            return None

        if not all(
            math.isfinite(value)
            for value in (
                initial,
                rate,
                years,
                result,
            )
        ):
            return None

        return {
            "initial": initial,
            "rate": rate,
            "years": years,
            "result": result,
        }

    @staticmethod
    def _numeric_candidates(
        answer: str,
    ) -> List[float]:
        values: List[float] = []

        for raw in re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
            str(answer or ""),
        ):
            try:
                values.append(
                    float(raw)
                )
            except Exception:
                continue

        return values

    @classmethod
    def _population_projection_answer_matches(
        cls,
        answer: str,
        expected: float,
    ) -> bool:
        tolerance = max(
            0.5,
            abs(float(expected)) * 1e-6,
        )

        return any(
            abs(value - expected)
            <= tolerance
            for value in cls._numeric_candidates(
                answer
            )
        )

    @staticmethod
    def _build_population_projection_answer(
        solution: Dict[str, float],
    ) -> str:
        initial = float(solution["initial"])
        rate = float(solution["rate"])
        years = float(solution["years"])
        result = float(solution["result"])

        initial_display = (
            str(int(initial))
            if initial.is_integer()
            else f"{initial:.6g}"
        )
        years_display = (
            str(int(years))
            if years.is_integer()
            else f"{years:.6g}"
        )
        rate_percent = rate * 100.0
        rate_display = (
            str(int(rate_percent))
            if rate_percent.is_integer()
            else f"{rate_percent:.6g}"
        )

        return (
            "Suponiendo crecimiento compuesto anual constante del "
            f"{rate_display}%, P = {initial_display} × "
            f"(1 + {rate:.12g})^{years_display} ≈ {result:.4f}. "
            f"Eso equivale a aproximadamente {int(round(result))} habitantes."
        )

    @staticmethod
    def _canonical_name(name: str) -> str:
        # Conservamos la forma visible, pero normalizamos espacios.
        return re.sub(r"\s+", " ", name.strip())

    @staticmethod
    def _dominant_relation_type(
        relations: Set[Relation],
    ) -> Optional[str]:
        rel_types = {rel_type for _, rel_type, _ in relations}

        if len(rel_types) == 1:
            return next(iter(rel_types))

        return None

    @staticmethod
    def _coerce_relation_type(
        relation: Relation,
        target_type: Optional[str],
    ) -> Relation:
        """
        Reinterpreta un símbolo genérico > / < como el comparativo dominante.

        Ejemplo:
            premisas: "Leo es más rápido que Nora" -> speed_gt
            respuesta: "Leo > Nora"                -> speed_gt
        """
        left, rel_type, right = relation

        if (
            target_type
            and target_type != "gt"
            and rel_type == "gt"
        ):
            return (left, target_type, right)

        return relation

    @classmethod
    def _coerce_relations(
        cls,
        relations: Set[Relation],
        target_type: Optional[str],
    ) -> Set[Relation]:
        return {
            cls._coerce_relation_type(relation, target_type)
            for relation in relations
        }

    @classmethod
    def _extract_relations(cls, text: str) -> Set[Relation]:
        relations: Set[Relation] = set()

        for pattern, rel_type, invert in _RELATION_PATTERNS:
            for match in pattern.finditer(text):
                left = cls._canonical_name(match.group(1))
                right = cls._canonical_name(match.group(2))

                if invert:
                    left, right = right, left

                if left.lower() == right.lower():
                    continue

                relations.add((left, rel_type, right))

        return relations

    @staticmethod
    def _transitive_closure(relations: Set[Relation]) -> Set[Relation]:
        closure = set(relations)
        changed = True

        while changed:
            changed = False
            current = list(closure)

            for a, rel1, b in current:
                for c, rel2, d in current:
                    if rel1 != rel2:
                        continue
                    if b.lower() != c.lower():
                        continue

                    candidate = (a, rel1, d)

                    if a.lower() == d.lower():
                        continue

                    if candidate not in closure:
                        closure.add(candidate)
                        changed = True

        return closure

    @staticmethod
    def _section(
        text: str,
        start_patterns: Tuple[str, ...],
        end_patterns: Tuple[str, ...] = (),
    ) -> str:
        lowered = text.lower()
        starts = []

        for pattern in start_patterns:
            idx = lowered.find(pattern.lower())
            if idx >= 0:
                starts.append((idx, pattern))

        if not starts:
            return ""

        start_idx, start_pattern = min(starts, key=lambda item: item[0])
        content_start = start_idx + len(start_pattern)

        end_idx = len(text)

        for pattern in end_patterns:
            idx = lowered.find(pattern.lower(), content_start)
            if idx >= 0:
                end_idx = min(end_idx, idx)

        return text[content_start:end_idx]

    @classmethod
    def _extract_section_claims(cls, text: str) -> Set[Relation]:
        """
        Extrae solo la afirmación principal de cada línea de una sección.

        Evita confundir las premisas citadas dentro de una justificación
        entre paréntesis con la relación que realmente fue clasificada
        como directa o deducida.
        """
        claims: Set[Relation] = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            # Quita numeración tipo "1.", "2)".
            line = re.sub(r"^\s*\d+[\.)]\s*", "", line)

            # Conserva solo la afirmación principal antes de explicaciones.
            primary = line.split("(", 1)[0].strip()

            for marker in (" porque ", " ya que ", " como parte ", " pues "):
                idx = primary.lower().find(marker)
                if idx >= 0:
                    primary = primary[:idx].strip()

            claims.update(cls._extract_relations(primary))

        return claims

    @staticmethod
    def _relation_label(relation: Relation) -> str:
        left, rel_type, right = relation

        labels = {
            "gt": f"{left} > {right}",
            "before": f"{left} antes de {right}",
            "speed_gt": f"{left} es más rápido que {right}",
            "height_gt": f"{left} es más alto que {right}",
            "weight_gt": f"{left} es más pesado que {right}",
            "size_gt": f"{left} es más grande que {right}",
            "length_gt": f"{left} es más largo que {right}",
            "price_gt": f"{left} es más caro que {right}",
            "age_gt": f"{left} es más viejo que {right}",
        }

        return labels.get(rel_type, f"{left} > {right}")

    @staticmethod
    def _inverse_relation(relation: Relation) -> Relation:
        left, rel_type, right = relation
        return (right, rel_type, left)

    @staticmethod
    def _line_is_negated(line: str) -> bool:
        """
        Detecta negaciones/correcciones por palabras completas.

        Importante: no usa búsquedas de subcadenas como "no ",
        porque nombres como "Bruno " contienen accidentalmente "no ".
        """
        lowered = line.lower()

        return bool(
            re.search(
                r"\b(?:"
                r"no|not|falso|false|incorrecto|incorrecta|incorrect|"
                r"wrong|error|inv[aá]lid\w*|invalid\w*|contradic\w*"
                r")\b",
                lowered,
            )
        )

    @classmethod
    def _extract_positive_answer_relations(
        cls,
        answer: str,
    ) -> Set[Relation]:
        """
        Extrae relaciones afirmadas positivamente en la respuesta.

        Ignora líneas que parecen negar/corregir una relación para evitar
        falsos positivos como "No es cierto que Bruno > Diego".
        """
        relations: Set[Relation] = set()

        for line in answer.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if cls._line_is_negated(stripped):
                continue

            relations.update(cls._extract_relations(stripped))

        return relations

    @classmethod
    def _extract_order_chains(
        cls,
        answer: str,
    ) -> List[Tuple[List[str], str]]:
        """
        Extrae cadenas como:
            Ana > Diego > Bruno > Carla
        o:
            Carla < Bruno < Diego < Ana

        Devuelve nombres normalizados en orden mayor->menor /
        antes->después según el tipo de relación.
        """
        chains: List[Tuple[List[str], str]] = []

        for match in _CHAIN_RE.finditer(answer):
            raw = match.group(1)
            tokens = re.split(r"\s*([<>])\s*", raw)

            names = tokens[0::2]
            ops = tokens[1::2]

            if not names or not ops:
                continue

            # Si mezcla > y < dejamos el caso al auditor semántico.
            if len(set(ops)) != 1:
                continue

            op = ops[0]

            if op == ">":
                ordered = [cls._canonical_name(n) for n in names]
            else:
                # A < B < C equivale a C > B > A.
                ordered = [
                    cls._canonical_name(n)
                    for n in reversed(names)
                ]

            if len(ordered) >= 2:
                chains.append((ordered, raw))

        return chains

    @staticmethod
    def _user_requests_order(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            marker in lowered
            for marker in (
                "orden exacto",
                "orden final",
                "ordena",
                "orden de mayor a menor",
                "mayor a menor",
                "menor a mayor",
                "más rápido a más lento",
                "mas rapido a mas lento",
                "más lento a más rápido",
                "mas lento a mas rapido",
                "más alto a más bajo",
                "mas alto a mas bajo",
                "más pesado a más ligero",
                "mas pesado a mas ligero",
                "más grande a más pequeño",
                "mas grande a mas pequeno",
                "más largo a más corto",
                "mas largo a mas corto",
                "más caro a más barato",
                "mas caro a mas barato",
                "más viejo a más joven",
                "mas viejo a mas joven",
                "determina el orden",
                "determinar el orden",
            )
        )

    @classmethod
    def _relational_profile(
        cls,
        user_text: str,
    ) -> Dict[str, Any]:
        """
        Analiza si las premisas relacionales determinan un orden total único.

        Un problema es "total" cuando:
        - usa un solo tipo de relación transitiva;
        - no contiene ciclos;
        - cada par de elementos es comparable en exactamente una dirección.
        """
        explicit = cls._extract_relations(user_text)

        if not explicit:
            return {
                "applicable": False,
                "explicit": set(),
                "closure": set(),
                "nodes": [],
                "rel_type": None,
                "total_order": False,
                "expected_order": [],
            }

        rel_types = {rel_type for _, rel_type, _ in explicit}
        nodes_by_lower: Dict[str, str] = {}

        for left, _, right in explicit:
            nodes_by_lower.setdefault(left.lower(), left)
            nodes_by_lower.setdefault(right.lower(), right)

        nodes = list(nodes_by_lower.values())
        closure = cls._transitive_closure(explicit)

        # Para autoridad determinista exigimos un solo tipo relacional.
        if len(rel_types) != 1:
            return {
                "applicable": True,
                "explicit": explicit,
                "closure": closure,
                "nodes": nodes,
                "rel_type": None,
                "total_order": False,
                "expected_order": [],
            }

        rel_type = next(iter(rel_types))

        # Detecta ciclos: A>B y B>A.
        cyclic = False
        for i, left in enumerate(nodes):
            for right in nodes[i + 1:]:
                if (
                    (left, rel_type, right) in closure
                    and (right, rel_type, left) in closure
                ):
                    cyclic = True
                    break
            if cyclic:
                break

        if cyclic:
            return {
                "applicable": True,
                "explicit": explicit,
                "closure": closure,
                "nodes": nodes,
                "rel_type": rel_type,
                "total_order": False,
                "expected_order": [],
            }

        # Todos los pares deben ser comparables en exactamente una dirección.
        total = True
        for i, left in enumerate(nodes):
            for right in nodes[i + 1:]:
                lr = (left, rel_type, right) in closure
                rl = (right, rel_type, left) in closure

                if lr == rl:  # ambos False o ambos True
                    total = False
                    break
            if not total:
                break

        expected_order: List[str] = []

        if total:
            # En un orden total, el primero domina/precede a n-1,
            # el siguiente a n-2, etc.
            def out_degree(name: str) -> int:
                return sum(
                    1
                    for other in nodes
                    if other.lower() != name.lower()
                    and (name, rel_type, other) in closure
                )

            expected_order = sorted(
                nodes,
                key=out_degree,
                reverse=True,
            )

        return {
            "applicable": True,
            "explicit": explicit,
            "closure": closure,
            "nodes": nodes,
            "rel_type": rel_type,
            "total_order": total,
            "expected_order": expected_order,
        }

    @classmethod
    def _extract_final_orders(
        cls,
        answer: str,
        known_nodes: List[str],
    ) -> List[Tuple[List[str], str]]:
        """
        Extrae órdenes finales tanto con símbolos:
            Elena > Hugo > Irene
        como con una frase:
            El orden exacto es Elena, Hugo, Irene, Gabriela y Fabio.
        """
        orders = list(cls._extract_order_chains(answer))

        if not known_nodes:
            return orders

        known = {name.lower(): name for name in known_nodes}

        for raw_line in answer.splitlines():
            line = raw_line.strip()
            lowered = line.lower()

            if not line:
                continue

            if not any(
                marker in lowered
                for marker in (
                    "orden",
                    "mayor a menor",
                    "menor a mayor",
                )
            ):
                continue

            positions: List[Tuple[int, str]] = []

            for low_name, canonical in known.items():
                match = re.search(
                    rf"\b{re.escape(low_name)}\b",
                    lowered,
                )
                if match:
                    positions.append((match.start(), canonical))

            positions.sort(key=lambda item: item[0])
            names = [name for _, name in positions]

            # Elimina repetidos preservando orden.
            deduped: List[str] = []
            seen: Set[str] = set()

            for name in names:
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(name)

            if len(deduped) >= 3:
                if not any(
                    [n.lower() for n in existing] ==
                    [n.lower() for n in deduped]
                    for existing, _ in orders
                ):
                    orders.append((deduped, line))

        return orders

    @classmethod
    def _build_deterministic_relational_answer(
        cls,
        user_text: str,
    ) -> Optional[str]:
        """
        Respuesta de respaldo 100% determinista para un orden total único.

        Solo se usa si el modelo crítico no logra producir una corrección
        que pase nuevamente las comprobaciones.
        """
        profile = cls._relational_profile(user_text)

        if not profile.get("total_order"):
            return None

        explicit: Set[Relation] = profile["explicit"]
        closure: Set[Relation] = profile["closure"]
        expected: List[str] = profile["expected_order"]
        rel_type = profile["rel_type"]

        if not expected or rel_type is None:
            return None

        order_line = " > ".join(expected)

        headings = {
            "gt": f"Orden exacto de mayor a menor: {order_line}",
            "speed_gt": f"Orden exacto de más rápido a más lento: {order_line}",
            "height_gt": f"Orden exacto de más alto a más bajo: {order_line}",
            "weight_gt": f"Orden exacto de más pesado a más ligero: {order_line}",
            "size_gt": f"Orden exacto de más grande a más pequeño: {order_line}",
            "length_gt": f"Orden exacto de más largo a más corto: {order_line}",
            "price_gt": f"Orden exacto de más caro a más barato: {order_line}",
            "age_gt": f"Orden exacto de más viejo a más joven: {order_line}",
            "before": f"Orden exacto: {' → '.join(expected)}",
        }

        heading = headings.get(
            rel_type,
            f"Orden exacto: {order_line}",
        )

        # Ordena relaciones según la posición de sus elementos.
        index = {name.lower(): i for i, name in enumerate(expected)}

        def rel_key(rel: Relation):
            left, _, right = rel
            return (
                index.get(left.lower(), 999),
                index.get(right.lower(), 999),
                left.lower(),
                right.lower(),
            )

        direct = sorted(explicit, key=rel_key)
        deduced = sorted(
            closure - explicit,
            key=rel_key,
        )

        lines = [heading, "", "Relaciones directas:"]

        for relation in direct:
            lines.append(f"- {cls._relation_label(relation)}")

        if deduced:
            lines.extend(["", "Relaciones deducidas por transitividad:"])
            for relation in deduced:
                lines.append(f"- {cls._relation_label(relation)}")

        lines.extend([
            "",
            "El orden final es compatible con todas las premisas.",
        ])

        return "\n".join(lines)

    @classmethod
    def _check_final_order(
        cls,
        user_text: str,
        answer: str,
    ) -> List[str]:
        """
        Comprueba el orden final contra el cierre transitivo.

        Si las premisas determinan un orden total único y el usuario pide
        el orden, también detecta respuestas que se niegan a darlo.
        """
        issues: List[str] = []
        profile = cls._relational_profile(user_text)

        if not profile.get("applicable"):
            return issues

        explicit: Set[Relation] = profile["explicit"]
        closure: Set[Relation] = profile["closure"]
        known_nodes: List[str] = profile["nodes"]
        rel_type = profile["rel_type"]

        orders = cls._extract_final_orders(
            answer,
            known_nodes,
        )

        expected: List[str] = profile.get("expected_order", [])
        total_order = bool(profile.get("total_order"))

        if (
            total_order
            and cls._user_requests_order(user_text)
            and not orders
        ):
            expected_text = " > ".join(expected)
            issues.append(
                "Orden final omitido o rechazo incorrecto: "
                "las premisas sí determinan un orden total único: "
                f"{expected_text}."
            )
            return issues

        if rel_type is None:
            return issues

        expected_lower = [name.lower() for name in expected]

        for names, raw_order in orders:
            if len(names) < 3:
                continue

            names_lower = [name.lower() for name in names]

            if total_order:
                known_lower = {name.lower() for name in known_nodes}

                if set(names_lower) != known_lower:
                    missing = [
                        name
                        for name in expected
                        if name.lower() not in set(names_lower)
                    ]
                    if missing:
                        issues.append(
                            "Orden final incompleto: "
                            f"'{raw_order}' omite: {', '.join(missing)}."
                        )

                elif names_lower != expected_lower:
                    issues.append(
                        "Orden final incorrecto: "
                        f"la respuesta propone '{raw_order}', "
                        "pero las premisas determinan el orden único "
                        f"'{' > '.join(expected)}'."
                    )

            # Comprueba además cada par afirmado por la cadena.
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    left = names[i]
                    right = names[j]
                    claimed = (left, rel_type, right)
                    inverse = (right, rel_type, left)

                    if inverse in closure:
                        issues.append(
                            "Orden final contradictorio: "
                            f"la cadena '{raw_order}' coloca "
                            f"{cls._relation_label(claimed)}, "
                            "pero las premisas implican lo contrario: "
                            f"{cls._relation_label(inverse)}."
                        )
                    elif claimed not in closure:
                        issues.append(
                            "Orden final no respaldado: "
                            f"la cadena '{raw_order}' presupone "
                            f"{cls._relation_label(claimed)}, "
                            "pero esa relación no se deriva de las premisas."
                        )

        return issues

    @classmethod
    def _check_relational_reasoning(
        cls,
        user_text: str,
        answer: str,
    ) -> List[str]:
        """
        Revisión determinista ligera para tareas de relaciones transitivas.

        Comprueba especialmente:
        - una premisa explícita no debe marcarse como "deducida";
        - una relación "deducida" debe pertenecer al cierre transitivo;
        - si el usuario pide distinguir directas/deducidas, las premisas
          explícitas deberían aparecer en la sección de relaciones directas.
        """
        issues: List[str] = []

        explicit = cls._extract_relations(user_text)

        if not explicit:
            return issues

        closure = cls._transitive_closure(explicit)

        direct_section = cls._section(
            answer,
            (
                "relaciones directas:",
                "relaciones directas",
                "premisas directas:",
                "premisas directas",
            ),
            (
                "relaciones deducidas",
                "relaciones inferidas",
                "por transitividad",
                "inferencias",
            ),
        )

        inferred_section = cls._section(
            answer,
            (
                "relaciones deducidas por transitividad:",
                "relaciones deducidas:",
                "relaciones deducidas",
                "relaciones inferidas:",
                "relaciones inferidas",
                "inferencias:",
                "inferencias",
            ),
        )

        dominant_type = cls._dominant_relation_type(explicit)

        direct_relations = cls._coerce_relations(
            cls._extract_section_claims(direct_section),
            dominant_type,
        )
        inferred_relations = cls._coerce_relations(
            cls._extract_section_claims(inferred_section),
            dominant_type,
        )

        wants_classification = any(
            token in user_text.lower()
            for token in (
                "relaciones directas",
                "directas y",
                "deducen por transitividad",
                "deducidas por transitividad",
                "cuáles se deducen",
                "cuales se deducen",
            )
        )

        if wants_classification and direct_section:
            for relation in sorted(explicit):
                if relation not in direct_relations:
                    issues.append(
                        "Premisa explícita omitida o mal clasificada: "
                        f"{cls._relation_label(relation)} aparece "
                        "directamente en el enunciado y debería figurar "
                        "como relación directa."
                    )

        for relation in sorted(inferred_relations):
            if relation in explicit:
                issues.append(
                    "Relación mal clasificada: "
                    f"{cls._relation_label(relation)} fue marcada como "
                    "deducida, pero es una premisa explícita del enunciado."
                )
                continue

            if relation not in closure:
                issues.append(
                    "Inferencia transitiva no justificada: "
                    f"{cls._relation_label(relation)} no se deriva de las "
                    "premisas explícitas mediante transitividad."
                )

        # Detecta relaciones afirmadas en sentido contrario a lo que
        # implican las premisas. Ejemplo: premisa Diego > Bruno,
        # respuesta Bruno > Diego.
        positive_claims = cls._coerce_relations(
            cls._extract_positive_answer_relations(answer),
            dominant_type,
        )

        for relation in sorted(positive_claims):
            inverse = cls._inverse_relation(relation)

            if inverse in closure and relation not in closure:
                issues.append(
                    "Relación invertida o contradictoria: "
                    f"la respuesta afirma {cls._relation_label(relation)}, "
                    "pero las premisas implican "
                    f"{cls._relation_label(inverse)}."
                )

        # Comprueba el orden final completo.
        issues.extend(cls._check_final_order(user_text, answer))

        # Elimina duplicados conservando el orden.
        deduped: List[str] = []
        seen: Set[str] = set()

        for issue in issues:
            if issue in seen:
                continue
            seen.add(issue)
            deduped.append(issue)

        return deduped

    @staticmethod
    def _looks_like_reasoning(answer: str) -> bool:
        lowered = answer.lower()

        markers = (
            "paso ",
            "pasos:",
            "procedimiento",
            "método",
            "metodo",
            "explicación",
            "explicacion",
            "cálculo",
            "calculo",
            "porque",
            "por lo tanto",
            "por tanto",
            "resultado",
            "comprobación",
            "comprobacion",
            "regla ",
            "relaciones directas",
            "relaciones deducidas",
            "transitividad",
        )

        if any(marker in lowered for marker in markers):
            return True

        if re.search(r"(?m)^\s*\d+\.\s+", answer):
            return True

        return False

    @classmethod
    def _needs_semantic_audit(
        cls,
        user_text: str,
        deterministic_issues: List[str],
        answer: str = "",
    ) -> bool:
        if deterministic_issues:
            return True

        lowered = user_text.lower()

        if any(hint in lowered for hint in _REASONING_HINTS):
            return True

        return cls._looks_like_reasoning(answer)

    @staticmethod
    def _strip_accents(text: str) -> str:
        table = str.maketrans(
            {
                "á": "a", "é": "e", "í": "i",
                "ó": "o", "ú": "u", "ü": "u",
                "Á": "A", "É": "E", "Í": "I",
                "Ó": "O", "Ú": "U", "Ü": "U",
            }
        )
        return (text or "").translate(table)

    @classmethod
    def _normalize_fact_text(cls, text: str) -> str:
        normalized = cls._strip_accents(text).lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _parse_episodic_evidence(
        evidence: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not evidence or not isinstance(evidence, str):
            return []

        try:
            payload = json.loads(evidence)
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []

        raw_items = payload.get("episodic_memories")
        if not isinstance(raw_items, list):
            return []

        episodes: List[Dict[str, Any]] = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            fields = item.get("fields")
            if not isinstance(fields, dict) or not fields:
                continue

            clean_fields = {
                str(k): str(v)
                for k, v in fields.items()
                if str(k).strip()
                and str(v).strip()
                and not re.match(
                    r"^(?:responde|respondeme|respuesta|contesta|contestame|reply|answer)",
                    str(k).strip(),
                    flags=re.IGNORECASE,
                )
            }

            if not clean_fields:
                continue

            episodes.append(
                {
                    "episode_key": str(
                        item.get("episode_key") or "episodio"
                    ),
                    "fields": clean_fields,
                }
            )

        return episodes

    @staticmethod
    def _parse_advanced_memory_evidence(
        evidence: Optional[str],
    ) -> List[Dict[str, Any]]:
        if (
            not evidence
            or not isinstance(
                evidence,
                str,
            )
        ):
            return []

        try:
            payload = json.loads(
                evidence
            )
        except Exception:
            return []

        if not isinstance(
            payload,
            dict,
        ):
            return []

        raw_items = payload.get(
            "advanced_memories"
        )

        if not isinstance(
            raw_items,
            list,
        ):
            return []

        out: List[
            Dict[str, Any]
        ] = []

        for item in raw_items:
            if not isinstance(
                item,
                dict,
            ):
                continue

            lines = [
                str(line).strip()
                for line in (
                    item.get(
                        "display_lines"
                    )
                    or []
                )
                if str(line).strip()
            ]

            if not lines:
                continue

            out.append(
                {
                    "kind":
                        str(
                            item.get(
                                "kind"
                            )
                            or "advanced"
                        ),
                    "display_lines":
                        lines,
                }
            )

        return out

    @classmethod
    def _advanced_memory_authority_for_query(
        cls,
        user_text: str,
        memories: List[Dict[str, Any]],
    ) -> bool:
        if not memories:
            return False

        text = cls._normalize_fact_text(
            user_text
        )

        markers = (
            "procedimiento",
            "pasos para",
            "rutina",
            "workflow",
            "objetivo",
            "objetivos",
            "pendiente",
            "siguiente paso",
            "esquema",
            "estructura de conocimiento",
            "resumen estructurado",
            "conflicto",
            "contradiccion",
            "incertidumbre",
            "procedencia",
            "fuentes",
            "confianza",
            "recuerdos prioritarios",
            "memorias prioritarias",
            "saliencia",
            "archivar",
            "olvido adaptativo",
            "metamemoria",
            "estado de tu memoria",
            "estado de la memoria",
            "que recuerdas sobre",
            "que tan seguro",
            "de donde sabes",
            "que sabes que sabes",
            "que te falta saber",
        )

        return any(
            cls._normalize_fact_text(
                marker
            )
            in text
            for marker
            in markers
        )

    @staticmethod
    def _build_advanced_memory_answer(
        memories: List[Dict[str, Any]],
    ) -> str:
        blocks: List[str] = []

        for memory in memories:
            lines = [
                str(line).strip()
                for line in (
                    memory.get(
                        "display_lines"
                    )
                    or []
                )
                if str(line).strip()
            ]

            if lines:
                blocks.append(
                    "\n".join(
                        lines
                    )
                )

        return "\n\n".join(
            blocks
        )

    @staticmethod
    def _parse_associative_evidence(
        evidence: Optional[str],
    ) -> List[Dict[str, Any]]:
        if (
            not evidence
            or not isinstance(
                evidence,
                str,
            )
        ):
            return []

        try:
            payload = json.loads(
                evidence
            )
        except Exception:
            return []

        if not isinstance(
            payload,
            dict,
        ):
            return []

        raw_items = payload.get(
            "associative_memories"
        )

        if not isinstance(
            raw_items,
            list,
        ):
            return []

        memories: List[
            Dict[str, Any]
        ] = []

        for item in raw_items:
            if not isinstance(
                item,
                dict,
            ):
                continue

            profiles = []

            for profile in (
                item.get(
                    "profiles"
                )
                or []
            ):
                if not isinstance(
                    profile,
                    dict,
                ):
                    continue

                subject = str(
                    profile.get(
                        "subject"
                    )
                    or ""
                ).strip()
                fields = profile.get(
                    "fields"
                )

                if (
                    not subject
                    or not isinstance(
                        fields,
                        dict,
                    )
                    or not fields
                ):
                    continue

                clean_fields = {
                    str(key):
                        str(value)
                    for key, value
                    in fields.items()
                    if str(key).strip()
                    and str(value).strip()
                    and not re.match(
                        r"^(?:responde|respondeme|respuesta|contesta|contestame|reply|answer)",
                        str(key).strip(),
                        flags=re.IGNORECASE,
                    )
                }

                if clean_fields:
                    profiles.append(
                        {
                            "subject":
                                subject,
                            "fields":
                                clean_fields,
                        }
                    )

            if profiles:
                memories.append(
                    {
                        "anchor_subject":
                            str(
                                item.get(
                                    "anchor_subject"
                                )
                                or ""
                            ),
                        "profiles":
                            profiles,
                    }
                )

        return memories

    @classmethod
    def _associative_authority_for_query(
        cls,
        user_text: str,
        memories: List[Dict[str, Any]],
    ) -> bool:
        if not memories:
            return False

        text = cls._normalize_fact_text(
            user_text
        )

        markers = (
            "conocimientos asociados",
            "conocimiento asociado",
            "conocimientos relacionados",
            "temas asociados",
            "temas relacionados",
            "conceptos asociados",
            "conceptos relacionados",
            "ideas asociadas",
            "ideas relacionadas",
            "similares a",
            "similar a",
            "se parecen a",
            "comparten caracteristicas",
            "asociado con",
            "asociados con",
        )

        return any(
            cls._normalize_fact_text(
                marker
            )
            in text
            for marker
            in markers
        )

    @classmethod
    def _build_associative_answer(
        cls,
        memories: List[Dict[str, Any]],
    ) -> str:
        if not memories:
            return ""

        blocks: List[str] = []
        seen_subjects: Set[str] = set()

        for memory in memories:
            for profile in (
                memory.get(
                    "profiles"
                )
                or []
            ):
                subject = str(
                    profile.get(
                        "subject"
                    )
                    or ""
                ).strip()
                fields = dict(
                    profile.get(
                        "fields"
                    )
                    or {}
                )

                if (
                    not subject
                    or not fields
                ):
                    continue

                subject_key = (
                    subject.lower()
                )

                if (
                    subject_key
                    in seen_subjects
                ):
                    continue

                seen_subjects.add(
                    subject_key
                )

                lines = []

                for key, value in sorted(
                    fields.items()
                ):
                    display_key = (
                        cls._episodic_display_key(
                            cls._normalize_fact_text(
                                key
                            )
                        )
                    )
                    lines.append(
                        f"{display_key} = {value}"
                    )

                if lines:
                    blocks.append(
                        subject
                        + ":\n"
                        + "\n".join(
                            lines
                        )
                    )

        return "\n\n".join(
            blocks
        )

    @staticmethod
    def _parse_relational_evidence(
        evidence: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not evidence or not isinstance(evidence, str):
            return []

        try:
            payload = json.loads(evidence)
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []

        raw_items = payload.get("relational_memories")
        if not isinstance(raw_items, list):
            return []

        out: List[Dict[str, Any]] = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            lines = [
                str(line).strip()
                for line in (item.get("display_lines") or [])
                if str(line).strip()
            ]

            if not lines:
                continue

            out.append(
                {
                    "relation_type": str(
                        item.get("relation_type") or "relation"
                    ),
                    "entities": list(item.get("entities") or []),
                    "display_lines": lines,
                }
            )

        return out

    @classmethod
    def _build_relational_answer(
        cls,
        memories: List[Dict[str, Any]],
    ) -> str:
        blocks: List[str] = []

        for memory in memories:
            lines = [
                str(line).strip()
                for line in (memory.get("display_lines") or [])
                if str(line).strip()
            ]
            if lines:
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    @staticmethod
    def _parse_semantic_evidence(
        evidence: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not evidence or not isinstance(evidence, str):
            return []

        try:
            payload = json.loads(evidence)
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []

        raw_items = payload.get("semantic_memories")
        if not isinstance(raw_items, list):
            return []

        memories: List[Dict[str, Any]] = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            fields = item.get("fields")
            if not isinstance(fields, dict) or not fields:
                continue

            clean_fields = {
                str(k): str(v)
                for k, v in fields.items()
                if str(k).strip()
                and str(v).strip()
                and not re.match(
                    r"^(?:responde|respondeme|respuesta|contesta|contestame|reply|answer)",
                    str(k).strip(),
                    flags=re.IGNORECASE,
                )
            }

            if not clean_fields:
                continue

            memories.append(
                {
                    "subject": str(
                        item.get("subject") or "sujeto"
                    ),
                    "fields": clean_fields,
                }
            )

        return memories

    @classmethod
    def _semantic_authority_for_query(
        cls,
        user_text: str,
        memories: List[Dict[str, Any]],
    ) -> bool:
        if not memories:
            return False

        text = cls._normalize_fact_text(user_text)

        # Órdenes de almacenamiento no son consultas.
        if re.search(
            r"\\b(?:guarda|guardar|memoriza|memorizar|conserva|conservar)\\b",
            text,
        ):
            return False

        for item in memories:
            subject = cls._normalize_fact_text(
                str(item.get("subject") or "")
            )
            if subject and subject in text:
                return True

        return False

    @classmethod
    def _requested_semantic_fields(
        cls,
        user_text: str,
        subject: str,
        fields: Dict[str, str],
    ) -> List[str]:
        text = cls._normalize_fact_text(user_text)

        all_markers = (
            "que sabes",
            "qué sabes",
            "todos los datos",
            "datos factuales",
            "datos actuales",
            "informacion actual",
            "información actual",
            "todo lo que sabes",
        )

        if any(marker in text for marker in all_markers):
            return list(fields.keys())

        aliases = {
            "proyecto": (
                "nombre del proyecto",
                "cual proyecto",
                "qué proyecto",
            ),
            "codigo": (
                "codigo",
                "código",
                "identificador",
            ),
            "color": (
                "color",
                "tono",
                "tonalidad",
            ),
            "numero": (
                "numero",
                "número",
                "cifra",
            ),
            "lugar": (
                "lugar",
                "sitio",
            ),
            "dia": (
                "dia",
                "día",
            ),
            "objeto": (
                "objeto",
                "instrumento",
            ),
            "clave": (
                "clave",
            ),
            "estacion": (
                "estacion",
                "estación",
                "parada",
            ),
            "llave": (
                "llave",
            ),
            "animal": (
                "animal",
            ),
            "cantidad": (
                "cantidad",
            ),
            "lenguaje": (
                "lenguaje",
                "idioma de programacion",
                "idioma de programación",
            ),
            "estado": (
                "estado",
                "fase",
            ),
            "especialidad": (
                "especialidad",
                "area",
                "área",
            ),
            "area": (
                "area",
                "área",
            ),
            "ciudad": (
                "ciudad",
            ),
        }

        subject_norm = cls._normalize_fact_text(subject)
        requested: List[str] = []

        for key, value in fields.items():
            key_norm = cls._normalize_fact_text(key)
            value_norm = cls._normalize_fact_text(value)

            # Campo que sólo identifica al sujeto (p.ej. proyecto=Aster):
            # no se añade sólo porque "proyecto Aster" aparezca en la pregunta.
            is_identity = (
                value_norm
                and value_norm == subject_norm
            )

            candidates = aliases.get(
                key_norm,
                (key_norm,),
            )

            explicit = any(
                cls._normalize_fact_text(candidate) in text
                for candidate in candidates
                if candidate
            )

            if explicit and not (
                is_identity
                and key_norm in ("proyecto", "lugar", "estacion", "nombre")
                and not any(
                    marker in text
                    for marker in (
                        "nombre del proyecto",
                        "cual proyecto",
                        "qué proyecto",
                    )
                )
            ):
                requested.append(key)

        return requested or list(fields.keys())

    @classmethod
    def _user_requests_current_value(
        cls,
        user_text: str,
    ) -> bool:
        """
        Verdadero cuando el usuario pide explícitamente el valor vigente.

        Esta señal da autoridad determinista a Semantic Memory sobre
        valores históricos recuperados desde episodios.
        """
        text = cls._normalize_fact_text(
            user_text
        )

        if re.search(
            r"\b(?:actual|actualmente|vigente|ahora|hoy)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        return any(
            marker in text
            for marker in (
                "mas reciente",
                "más reciente",
                "ultimo valor",
                "último valor",
                "valor vigente",
                "dato vigente",
            )
        )

    @classmethod
    def _semantic_response_issues(
        cls,
        user_text: str,
        answer: str,
        memories: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []

        lowered = cls._normalize_fact_text(answer)

        leak_patterns = (
            r"\\bsemantic_memory\\b",
            r"\\bsemantic_subject\\b",
            r"\\bconfidence\\s*=",
            r"\\bsupport_count\\s*=",
            r"\\bsources\\s*=",
            r"\\bscore\\s*=",
        )

        if any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in leak_patterns
        ):
            issues.append(
                "La respuesta expone metadatos internos de memoria semántica."
            )

        for memory in memories:
            subject = str(memory.get("subject") or "sujeto")
            fields = dict(memory.get("fields") or {})
            requested = cls._requested_semantic_fields(
                user_text,
                subject,
                fields,
            )

            for key in requested:
                if key not in fields:
                    continue

                value_norm = cls._normalize_fact_text(
                    str(fields[key])
                )

                if value_norm not in lowered:
                    issues.append(
                        f"Falta el dato semántico actual {key} = {fields[key]}."
                    )

        return issues

    @classmethod
    def _build_semantic_answer(
        cls,
        user_text: str,
        memories: List[Dict[str, Any]],
    ) -> str:
        if not memories:
            return ""

        blocks: List[str] = []

        for memory in memories:
            subject = str(memory.get("subject") or "sujeto")
            fields = dict(memory.get("fields") or {})
            requested = cls._requested_semantic_fields(
                user_text,
                subject,
                fields,
            )

            lines: List[str] = []

            for key in requested:
                if key not in fields:
                    continue

                display_key = cls._episodic_display_key(
                    cls._normalize_fact_text(key)
                )
                lines.append(
                    f"{display_key} = {fields[key]}"
                )

            if not lines:
                continue

            if len(memories) > 1:
                blocks.append(
                    f"{subject}:\n" + "\n".join(lines)
                )
            else:
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    @classmethod
    def _looks_like_episodic_recall(
        cls,
        user_text: str,
    ) -> bool:
        text = cls._normalize_fact_text(user_text)

        # Una instrucción de almacenamiento con varios pares clave=valor
        # no debe confundirse con una consulta de recuperación.
        field_lines = re.findall(
            r"(?m)^\s*[\wáéíóúñü -]{1,40}\s*(?:=|:)\s*.+$",
            user_text or "",
            flags=re.IGNORECASE,
        )

        store_markers = (
            "guarda",
            "guardar",
            "conserva",
            "conservar",
            "memoriza",
            "memorizar",
            "recuerda exactamente estos datos",
            "recuerda estos datos",
        )

        if len(field_lines) >= 2 and any(
            marker in text
            for marker in store_markers
        ):
            return False

        recall_markers = (
            "recupera",
            "recuperar",
            "busca el recuerdo",
            "busca la memoria",
            "recuerdas",
            "que datos",
            "dime los datos",
            "episodio",
            "memoria relacionada",
            "recuerdo relacionado",
            "memorias nuevas",
            "memorias anteriores",
            "recuerdos anteriores",
            "sin que vuelva a darte",
            "sin que vuelva a proporcionarte",
        )

        return any(marker in text for marker in recall_markers)

    @staticmethod
    def _episodic_display_key(key: str) -> str:
        mapping = {
            "codigo": "código",
            "numero": "número",
            "dia": "día",
            "estacion": "estación",
        }
        return mapping.get(key, key)

    @classmethod
    def _requested_episode_fields(
        cls,
        user_text: str,
        fields: Dict[str, str],
    ) -> List[str]:
        """
        Decide si el usuario pidió todos los campos o un subconjunto.

        Si pide "los datos", "todos", "cuatro datos", etc., devuelve todos.
        Si menciona campos concretos (clave, objeto, animal...), devuelve solo
        esos campos.
        """
        text = cls._normalize_fact_text(user_text)

        all_markers = (
            "todos los datos",
            "los cuatro datos",
            "los tres datos",
            "datos de ese episodio",
            "datos pertenecientes a ese episodio",
            "unicamente los cuatro datos",
            "solo los datos",
            "recupera las tres memorias",
            "recupera la memoria",
            "reconstruye",
        )

        if any(marker in text for marker in all_markers):
            return list(fields.keys())

        aliases = {
            "proyecto": (
                "proyecto", "iniciativa",
                "nombre de la iniciativa",
            ),
            "codigo": (
                "codigo", "identificador alfanumerico",
                "alfanumerico",
            ),
            "color": (
                "color", "tono", "tonalidad",
            ),
            "numero": (
                "numero", "cifra",
            ),
            "lugar": (
                "lugar", "sitio",
            ),
            "dia": (
                "dia",
            ),
            "objeto": (
                "objeto", "instrumento",
            ),
            "clave": (
                "clave",
            ),
            "estacion": (
                "estacion", "parada",
            ),
            "llave": (
                "llave",
            ),
            "animal": (
                "animal",
            ),
            "cantidad": (
                "cantidad",
            ),
        }

        requested: List[str] = []

        for key in fields:
            candidates = aliases.get(
                cls._normalize_fact_text(key),
                (cls._normalize_fact_text(key),),
            )

            if any(candidate in text for candidate in candidates):
                requested.append(key)

        return requested or list(fields.keys())

    @classmethod
    def _ordered_episodes_for_query(
        cls,
        user_text: str,
        episodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(episodes) <= 1:
            return episodes

        normalized_user = cls._normalize_fact_text(user_text)

        def position(item: Dict[str, Any]) -> Tuple[int, int]:
            key = cls._normalize_fact_text(
                str(item.get("episode_key") or "")
            )
            idx = (
                normalized_user.find(key)
                if key
                else -1
            )
            return (
                idx if idx >= 0 else 10**9,
                episodes.index(item),
            )

        return sorted(episodes, key=position)

    @classmethod
    def _episodic_response_issues(
        cls,
        user_text: str,
        answer: str,
        episodes: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []

        if not episodes:
            return issues

        lowered = cls._normalize_fact_text(answer)

        leak_patterns = (
            r"\[\s*episodic_memory",
            r"\bscore\s*=",
            r"\btema\s*=",
            r"\bepisode_key\s*=",
            r"\bsource\s*=",
            r"\brespondeme_unicamente\s*=",
            r"\bresponde_unicamente\s*=",
        )

        if any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in leak_patterns
        ):
            issues.append(
                "La respuesta expone metadatos internos del sistema de memoria."
            )

        ordered = cls._ordered_episodes_for_query(
            user_text,
            episodes,
        )

        for episode in ordered:
            fields = dict(episode.get("fields") or {})
            requested = cls._requested_episode_fields(
                user_text,
                fields,
            )

            for key in requested:
                if key not in fields:
                    continue

                value = str(fields[key])
                key_norm = cls._normalize_fact_text(key)
                value_norm = cls._normalize_fact_text(value)

                # Se exige el valor y, cuando el usuario pide todos los datos,
                # también una referencia reconocible al nombre del campo.
                if value_norm not in lowered:
                    issues.append(
                        f"Falta el dato episódico {key} = {value}."
                    )
                    continue

                key_aliases = {
                    "codigo": ("codigo", "código"),
                    "numero": ("numero", "número"),
                    "dia": ("dia", "día"),
                    "estacion": ("estacion", "estación"),
                }
                aliases = key_aliases.get(
                    key_norm,
                    (key_norm,),
                )

                if not any(
                    cls._normalize_fact_text(alias) in lowered
                    for alias in aliases
                ):
                    issues.append(
                        f"El valor {value} aparece, pero el campo {key} "
                        "no está identificado claramente."
                    )

        return issues

    @classmethod
    def _build_episodic_answer(
        cls,
        user_text: str,
        episodes: List[Dict[str, Any]],
    ) -> str:
        ordered = cls._ordered_episodes_for_query(
            user_text,
            episodes,
        )

        multi = len(ordered) > 1
        blocks: List[str] = []

        for episode in ordered:
            fields = dict(episode.get("fields") or {})
            requested = cls._requested_episode_fields(
                user_text,
                fields,
            )

            lines = []

            if multi:
                lines.append(
                    f"{episode.get('episode_key') or 'Memoria'}:"
                )

            for key in requested:
                if key not in fields:
                    continue

                display_key = cls._episodic_display_key(
                    cls._normalize_fact_text(key)
                )
                lines.append(
                    f"{display_key} = {fields[key]}"
                )

            if lines:
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, dict) else None
            except Exception:
                pass

        return None

    def _model_review(
        self,
        user_text: str,
        answer: str,
        deterministic_issues: List[str],
        context_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        if self.adapter is None:
            return None

        deterministic = (
            "\n".join(f"- {issue}" for issue in deterministic_issues)
            if deterministic_issues
            else "Ninguno detectado."
        )

        context_block = (
            context_text.strip()
            if context_text.strip()
            else "No hay contexto conversacional adicional."
        )

        review_messages = [
            {
                "role": "system",
                "content": (
                    "Eres Verifier 2.9.2, un crítico estricto de respuestas. "
                    "Debes comprobar la respuesta contra la instrucción original "
                    "y contra el contexto disponible. "
                    "Distingue siempre entre premisas explícitas e inferencias. "
                    "Una premisa explícita no debe presentarse como deducción. "
                    "Toda inferencia debe estar respaldada por premisas que realmente "
                    "la impliquen; si usa transitividad, verifica la cadena completa "
                    "y la dirección de cada relación. "
                    "Comprueba cualquier orden final propuesto contra todas las "
                    "relaciones explícitas y transitivas; no puede invertir ni "
                    "contradecir ninguna de ellas. "
                    "Busca además errores de aritmética, contradicciones, reglas "
                    "incumplidas, restricciones ignoradas, elementos inventados, "
                    "errores de lógica y pasos que no justifican la conclusión. "
                    "En procedimientos aritméticos, comprueba acarreo, préstamo, "
                    "signos, desplazamientos y consistencia entre resultados "
                    "intermedios y resultado final. "
                    "Si el usuario pide un método específico, comprueba que realmente "
                    "se haya usado ese método. "
                    "Si pide un formato exacto, comprueba que se respete. "
                    "No cambies una respuesta correcta. "
                    "Si existe un error, produce una versión corregida completa. "
                    "Devuelve SOLO JSON válido con esta estructura: "
                    '{"ok": true, "issues": [], "corrected_answer": null} '
                    "o "
                    '{"ok": false, "issues": ["..."], '
                    '"corrected_answer": "respuesta completa corregida"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    "CONTEXTO DISPONIBLE:\n"
                    f"{context_block}\n\n"
                    "INSTRUCCIÓN ORIGINAL MÁS RECIENTE:\n"
                    f"{user_text}\n\n"
                    "RESPUESTA CANDIDATA:\n"
                    f"{answer}\n\n"
                    "PROBLEMAS DETERMINISTAS DETECTADOS AUTOMÁTICAMENTE:\n"
                    f"{deterministic}"
                ),
            },
        ]

        raw = self.adapter.generate(
            messages=review_messages,
            max_tokens=768,
            temperature=0.0,
            stream=False,
            stop=None,
        )

        return self._extract_json(raw)

    def verify(
        self,
        answer: str,
        messages: List[Dict[str, str]],
        evidence: Optional[str] = None,
        structured_facts: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_text = self._latest_user(messages)
        context_text = self._render_context(messages)

        facts = dict(
            structured_facts
            or self._structured_facts_from_messages(
                messages
            )
            or {}
        )

        if self._looks_like_population_projection(
            user_text
        ):
            projection = (
                self._population_projection_solution(
                    facts
                )
            )

            if projection is not None:
                expected = float(
                    projection["result"]
                )

                if (
                    self._population_projection_answer_matches(
                        answer,
                        expected,
                    )
                ):
                    return answer

                return self._build_population_projection_answer(
                    projection
                )

        # --------------------------------------------------------------
        # Memory 3.0.2: valor actual semántico > historial episódico.
        #
        # Se ejecuta ANTES de la autoridad episódica. Si el usuario pide
        # "actual", "actualmente", "vigente", etc. y existe evidencia
        # semántica del sujeto, reconstruimos directamente la respuesta
        # desde el estado consolidado. Así se elimina cualquier valor
        # superseded aunque aparezca en la respuesta candidata.
        # --------------------------------------------------------------
        semantic_memories = self._parse_semantic_evidence(
            evidence
        )
        semantic_authority = self._semantic_authority_for_query(
            user_text,
            semantic_memories,
        )

        if (
            semantic_authority
            and self._user_requests_current_value(
                user_text
            )
        ):
            current_semantic_answer = (
                self._build_semantic_answer(
                    user_text,
                    semantic_memories,
                )
            )

            if current_semantic_answer:
                return current_semantic_answer

        # --------------------------------------------------------------
        # Verifier 2.7: autoridad determinista para memoria episódica.
        # --------------------------------------------------------------
        episodic_memories = self._parse_episodic_evidence(evidence)
        episodic_authority = bool(
            episodic_memories
            and self._looks_like_episodic_recall(user_text)
        )

        if episodic_authority:
            episodic_issues = self._episodic_response_issues(
                user_text,
                answer,
                episodic_memories,
            )

            # Si faltan campos o se filtraron metadatos internos, no pedimos
            # otra generación al LLM: reconstruimos desde la evidencia.
            if episodic_issues:
                rebuilt = self._build_episodic_answer(
                    user_text,
                    episodic_memories,
                )
                if rebuilt:
                    return rebuilt

            # Si ya contiene todos los datos solicitados y no hay filtraciones,
            # la respuesta queda intacta; evitamos una sobre-corrección del LLM.
            return answer

        # --------------------------------------------------------------
        # Verifier 2.9.1: autoridad determinista para Memory 2.9.
        # --------------------------------------------------------------
        advanced_memories = self._parse_advanced_memory_evidence(
            evidence
        )
        advanced_memory_authority = (
            self._advanced_memory_authority_for_query(
                user_text,
                advanced_memories,
            )
        )

        if advanced_memory_authority:
            advanced_memory_answer = (
                self._build_advanced_memory_answer(
                    advanced_memories
                )
            )

            if advanced_memory_answer:
                return advanced_memory_answer

        # --------------------------------------------------------------
        # Verifier 2.9.1: autoridad determinista para memoria asociativa.
        # --------------------------------------------------------------
        associative_memories = self._parse_associative_evidence(
            evidence
        )
        associative_authority = (
            self._associative_authority_for_query(
                user_text,
                associative_memories,
            )
        )

        if associative_authority:
            associative_answer = (
                self._build_associative_answer(
                    associative_memories
                )
            )

            if associative_answer:
                return associative_answer

        # --------------------------------------------------------------
        # Verifier 2.7: autoridad determinista para memoria relacional.
        # --------------------------------------------------------------
        relational_memories = self._parse_relational_evidence(
            evidence
        )

        if relational_memories:
            relational_answer = self._build_relational_answer(
                relational_memories
            )
            if relational_answer:
                return relational_answer

        # --------------------------------------------------------------
        # Verifier 2.7: autoridad determinista para memoria semántica.
        # --------------------------------------------------------------
        # semantic_memories / semantic_authority ya fueron calculados
        # al inicio de verify() para poder resolver primero las consultas
        # que piden explícitamente el valor actual.
        if semantic_authority:
            semantic_issues = self._semantic_response_issues(
                user_text,
                answer,
                semantic_memories,
            )

            if semantic_issues:
                rebuilt = self._build_semantic_answer(
                    user_text,
                    semantic_memories,
                )
                if rebuilt:
                    return rebuilt

            return answer

        arithmetic_issues = self._check_arithmetic(answer)
        relational_issues = self._check_relational_reasoning(
            user_text,
            answer,
        )

        deterministic_issues = (
            arithmetic_issues
            + relational_issues
        )

        profile = self._relational_profile(user_text)
        relational_authority = bool(
            profile.get("total_order")
        )

        # Regla central de Verifier 2.4:
        # si el problema relacional es completamente decidible por reglas
        # deterministas y la respuesta pasa esas reglas, NO permitimos que
        # un segundo LLM la "sobre-corrija".
        if relational_authority and not deterministic_issues:
            return answer

        if not self._needs_semantic_audit(
            user_text,
            deterministic_issues,
            answer,
        ):
            return answer

        review = self._model_review(
            user_text=user_text,
            answer=answer,
            deterministic_issues=deterministic_issues,
            context_text=context_text,
        )

        if not review:
            if relational_authority and deterministic_issues:
                fallback = self._build_deterministic_relational_answer(
                    user_text
                )
                if fallback:
                    return fallback
            return answer

        review_ok = bool(review.get("ok", False))

        must_correct = (
            bool(deterministic_issues)
            or not review_ok
        )

        if not must_correct:
            return answer

        corrected = review.get("corrected_answer")

        if isinstance(corrected, str) and corrected.strip():
            candidate = corrected.strip()

            # Si tenemos autoridad determinista relacional,
            # NO aceptamos la corrección hasta volver a verificarla.
            if relational_authority:
                candidate_issues = self._check_relational_reasoning(
                    user_text,
                    candidate,
                )

                if not candidate_issues:
                    return candidate

                # Segundo y último intento con los errores concretos
                # encontrados en la propia corrección.
                second_review = self._model_review(
                    user_text=user_text,
                    answer=candidate,
                    deterministic_issues=candidate_issues,
                    context_text=context_text,
                )

                if second_review:
                    second = second_review.get("corrected_answer")

                    if isinstance(second, str) and second.strip():
                        second_candidate = second.strip()
                        second_issues = self._check_relational_reasoning(
                            user_text,
                            second_candidate,
                        )

                        if not second_issues:
                            return second_candidate

                # Si el LLM crítico sigue equivocándose, usamos una
                # respuesta construida desde las relaciones deterministas.
                fallback = self._build_deterministic_relational_answer(
                    user_text
                )

                if fallback:
                    return fallback

                # Último recurso: no sustituimos una salida por otra
                # corrección que sabemos que sigue siendo inválida.
                return answer

            return candidate

        if relational_authority and deterministic_issues:
            fallback = self._build_deterministic_relational_answer(
                user_text
            )
            if fallback:
                return fallback

        return answer
