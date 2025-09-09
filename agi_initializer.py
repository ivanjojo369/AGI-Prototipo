# agi_initializer.py
# Inicializador robusto y optimizado para configurar diferentes backends de un sistema AGI.
# Lee configuración desde entorno, argumentos o archivos, valida parámetros y prepara el backend listo para uso.

from __future__ import annotations

import os
import json
import logging
import sys
from typing import Optional, Any, Dict
try:
    import yaml
except ImportError:
    yaml = None

# Importar posibles clases de backend y agentes (si existen en el proyecto)
try:
    from agents.agent import AgentBackend, AgentConfig, Agent
except ImportError:
    AgentBackend = None
    AgentConfig = None
    Agent = None
try:
    from agents.multi_agent_manager import MultiAgentManager
except ImportError:
    MultiAgentManager = None
try:
    from agents.meta_agent import MetaAgent
except ImportError:
    MetaAgent = None
try:
    from agents.reflection_engine import ReflectionEngine
except ImportError:
    ReflectionEngine = None

# Pequeño contenedor de configuración estilo "API" para inicializar un Agent/AgentConfig fácilmente
class _ApiCfg:
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 512,
        timeout: float = 30.0,
        system_prompt: Optional[str] = None,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt

def init_from_env() -> Dict[str, Any]:
    """
    Lee configuración desde variables de entorno.
    Variables usadas:
      DEMO_URL, DEMO_KEY, DEMO_T, DEMO_TOP_P, DEMO_MAX_TOKENS, DEMO_TIMEOUT,
      DEMO_MODE, DEMO_STRICT, DEMO_MAX_STEPS, DEMO_RETRY_COUNT, DEMO_REFLECT_BUDGET,
      DEMO_MULTI_COUNT, DEMO_BACKEND.
    Retorna un dict con los valores (usando None si no están definidos).
    """
    def _to_int(val: Optional[str]) -> Optional[int]:
        try:
            return int(val) if val is not None else None
        except:
            return None

    def _to_float(val: Optional[str]) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except:
            return None

    def _to_bool(val: Optional[str]) -> Optional[bool]:
        if val is None:
            return None
        s = val.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "si"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        return None

    return {
        "api_url": os.getenv("DEMO_URL", "http://127.0.0.1:8010/chat"),
        "api_key": os.getenv("DEMO_KEY", None),
        "temperature": _to_float(os.getenv("DEMO_T", None)),
        "top_p": _to_float(os.getenv("DEMO_TOP_P", None)),
        "max_new_tokens": _to_int(os.getenv("DEMO_MAX_TOKENS", None)) or None,
        "timeout": _to_float(os.getenv("DEMO_TIMEOUT", None)) or None,
        "mode": os.getenv("DEMO_MODE", None),
        "strict": _to_bool(os.getenv("DEMO_STRICT", None)),
        "max_steps": _to_int(os.getenv("DEMO_MAX_STEPS", None)),
        "retry_count": _to_int(os.getenv("DEMO_RETRY_COUNT", None)),
        "reflect_budget": _to_int(os.getenv("DEMO_REFLECT_BUDGET", None)),
        "multi_count": _to_int(os.getenv("DEMO_MULTI_COUNT", None)),
        "backend": os.getenv("DEMO_BACKEND", None),
    }

def load_config_file(path: str) -> Dict[str, Any]:
    """
    Carga configuración desde un archivo JSON o YAML.
    Retorna un diccionario con la configuración. Lanza excepción si falla.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        if path.endswith(".yaml") or path.endswith(".yml"):
            if yaml is None:
                raise ImportError("PyYAML is not installed, cannot load YAML file.")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported config file format (use .json or .yaml).")
    except Exception as e:
        logging.error("Error loading config file %s: %s", path, e)
        raise
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON/YAML object (dict) at root.")
    return data

# Protocolo de backend para indicar interfaz esperada (método run)
class BackendProtocol:
    def run(self, goal: str, **kwargs) -> Any: ...

def get_backend(
    backend: Optional[str] = None,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    system_prompt: Optional[str] = None,
    mode: Optional[str] = None,
    strict: Optional[bool] = None,
    max_steps: Optional[int] = None,
    retry_count: Optional[int] = None,
    reflect_budget: Optional[int] = None,
    config_file: Optional[str] = None,
) -> BackendProtocol:
    """
    Inicializa y retorna un backend de AGI listo para usar, con interfaz .run(goal, **kwargs).
    Parámetros:
      - backend: Tipo de backend a inicializar. Puede ser "agent", "http", "multi_agent", "planner" o "mock".
                 Si no se especifica, se tomará de la configuración (archivo/env) o por defecto "agent".
      - api_url: URL del servicio de lenguaje (ej: endpoint de API).
      - api_key: Clave de API para el servicio (si aplica).
      - temperature: Temperatura de muestreo (float entre 0.0 y 1.0). None para usar valor por defecto del modelo.
      - top_p: Porcentaje de nucleó (top-p) para muestreo (float entre 0.0 y 1.0). None para usar valor por defecto.
      - max_new_tokens: Máximo número de tokens de respuesta a generar.
      - timeout: Timeout en segundos para llamadas al modelo/servicio.
      - system_prompt: Prompt de sistema inicial (contexto) para el agente, si aplica.
      - mode: Modo de operación del agente (por ejemplo "auto", "manual", etc., según implementación).
      - strict: Modo estricto (bool) que puede ajustar restricciones en formato o validación.
      - max_steps: Máximo número de pasos que puede ejecutar el agente (en bucles de pensamiento/acción).
      - retry_count: Número de reintentos permitidos en caso de falla (por ejemplo, para volver a llamar al modelo si la respuesta es inválida).
      - reflect_budget: Presupuesto de reflexiones (iteraciones de auto-reflexión) que el agente puede utilizar.
      - config_file: Ruta a un archivo .json o .yaml con la configuración. 
                    Si se proporciona, sus valores se combinan con los anteriores.
    Prioridad de la configuración:
      1. Valores explícitos pasados como argumentos a esta función (no None).
      2. Valores obtenidos del archivo de configuración (si se especificó config_file).
      3. Valores de variables de entorno (según `init_from_env()`).
      4. Defaults razonables en caso de no definirse un parámetro crítico.
    """
    env_cfg = init_from_env()
    file_cfg: Dict[str, Any] = {}
    if config_file:
        file_cfg = load_config_file(config_file)
        logging.info("Configuración cargada desde archivo: %s", config_file)
    # Extraer lista de agentes si existe (para multi_agent o planner)
    agent_list_cfg = None
    if "agents" in file_cfg:
        agent_list_cfg = file_cfg.pop("agents")
    elif "multi_agents" in file_cfg:
        agent_list_cfg = file_cfg.pop("multi_agents")
    if agent_list_cfg is not None and not isinstance(agent_list_cfg, list):
        logging.error("El archivo de config debe definir 'agents' como lista.")
        agent_list_cfg = None

    # Determinar tipo de backend (prioridad: argumento > archivo > entorno > default)
    if backend is None or backend == "":
        backend = None
    if backend is None:
        backend = file_cfg.get("backend") or env_cfg.get("backend") or "agent"
    if backend is None:
        backend = "agent"
    backend = str(backend).lower()

    # Combinar configuración de parámetros con prioridad adecuada (param > file > env)
    api_url_val = api_url if api_url is not None else (file_cfg.get("api_url") if file_cfg.get("api_url") is not None else env_cfg.get("api_url"))
    api_key_val = api_key if api_key is not None else (file_cfg.get("api_key") if file_cfg.get("api_key") is not None else env_cfg.get("api_key"))
    temp_val = temperature if temperature is not None else (file_cfg.get("temperature") if file_cfg.get("temperature") is not None else env_cfg.get("temperature"))
    top_p_val = top_p if top_p is not None else (file_cfg.get("top_p") if file_cfg.get("top_p") is not None else env_cfg.get("top_p"))
    tokens_val = max_new_tokens if max_new_tokens is not None else (file_cfg.get("max_new_tokens") if file_cfg.get("max_new_tokens") is not None else env_cfg.get("max_new_tokens"))
    timeout_val = timeout if timeout is not None else (file_cfg.get("timeout") if file_cfg.get("timeout") is not None else env_cfg.get("timeout"))
    system_prompt_val = system_prompt if system_prompt is not None else (file_cfg.get("system_prompt") if file_cfg.get("system_prompt") is not None else env_cfg.get("system_prompt"))
    mode_val = mode if mode is not None else (file_cfg.get("mode") if file_cfg.get("mode") is not None else env_cfg.get("mode"))
    strict_val = strict if strict is not None else (file_cfg.get("strict") if file_cfg.get("strict") is not None else env_cfg.get("strict"))
    max_steps_val = max_steps if max_steps is not None else (file_cfg.get("max_steps") if file_cfg.get("max_steps") is not None else env_cfg.get("max_steps"))
    retry_count_val = retry_count if retry_count is not None else (file_cfg.get("retry_count") if file_cfg.get("retry_count") is not None else env_cfg.get("retry_count"))
    reflect_budget_val = reflect_budget if reflect_budget is not None else (file_cfg.get("reflect_budget") if file_cfg.get("reflect_budget") is not None else env_cfg.get("reflect_budget"))
    multi_count_val = None
    if backend == "multi_agent":
        multi_count_val = file_cfg.get("multi_count") if file_cfg.get("multi_count") is not None else env_cfg.get("multi_count")
    if backend == "multi_agent" and agent_list_cfg is None:
        multi_count_val = multi_count_val or 2

    # Validación y fallback de parámetros críticos
    if not api_url_val:
        raise ValueError("api_url no definido. Establezca DEMO_URL, config, o pase api_url como parámetro.")
    if temp_val is not None:
        try:
            temp_val = float(temp_val)
        except Exception:
            logging.warning("Valor de temperature inválido (%r), usando default 1.0.", temp_val)
            temp_val = 1.0
        if temp_val < 0.0 or temp_val > 1.0:
            logging.warning("Temperature %.3f fuera de rango [0.0, 1.0]. Se ajustará al rango.", temp_val)
            temp_val = max(0.0, min(1.0, temp_val))
    else:
        temp_val = 1.0
    if top_p_val is not None:
        try:
            top_p_val = float(top_p_val)
        except Exception:
            logging.warning("Valor de top_p inválido (%r), usando default 1.0.", top_p_val)
            top_p_val = 1.0
        if top_p_val <= 0.0 or top_p_val > 1.0:
            logging.warning("Top_p %.3f fuera de rango (0.0 < top_p <= 1.0). Se ajustará al rango.", top_p_val)
            top_p_val = 1.0 if top_p_val > 1.0 else 0.01
    else:
        top_p_val = 1.0
    if tokens_val is not None:
        try:
            tokens_val = int(tokens_val)
        except Exception:
            logging.warning("Valor de max_new_tokens inválido (%r), usando default 256.", tokens_val)
            tokens_val = 256
        if tokens_val < 1:
            logging.warning("max_new_tokens=%d es menor que 1, ajustando a 1.", tokens_val)
            tokens_val = 1
        elif tokens_val > 10000:
            logging.warning("max_new_tokens=%d es muy alto, se recomienda un valor menor (<=1000).", tokens_val)
    else:
        tokens_val = 256
    if timeout_val is not None:
        try:
            timeout_val = float(timeout_val)
        except Exception:
            logging.warning("Valor de timeout inválido (%r), usando default 30.0.", timeout_val)
            timeout_val = 30.0
        if timeout_val <= 0.0:
            logging.warning("Timeout %.2f no válido, usando mínimo 1.0s.", timeout_val)
            timeout_val = 1.0
    else:
        timeout_val = 30.0
    if strict_val is not None and not isinstance(strict_val, bool):
        strict_converted = None
        if isinstance(strict_val, str):
            s = strict_val.strip().lower()
            if s in ("1", "true", "t", "yes", "y", "si"):
                strict_converted = True
            elif s in ("0", "false", "f", "no", "n"):
                strict_converted = False
        elif isinstance(strict_val, (int, float)):
            strict_converted = bool(strict_val)
        if strict_converted is None:
            logging.warning("Valor de strict no reconocido (%r), se asumirá False.", strict_val)
            strict_val = False
        else:
            strict_val = strict_converted
    if strict_val is None:
        strict_val = False
    if max_steps_val is not None:
        try:
            max_steps_val = int(max_steps_val)
        except Exception:
            logging.warning("Valor de max_steps inválido (%r), usando default 100.", max_steps_val)
            max_steps_val = 100
        if max_steps_val <= 0:
            logging.warning("max_steps=%d no válido, usando default 100.", max_steps_val)
            max_steps_val = 100
        elif max_steps_val > 1000:
            logging.info("Advertencia: max_steps=%d es muy alto, el agente podría entrar en un bucle prolongado.", max_steps_val)
    else:
        max_steps_val = 100
    if retry_count_val is not None:
        try:
            retry_count_val = int(retry_count_val)
        except Exception:
            logging.warning("Valor de retry_count inválido (%r), usando default 0.", retry_count_val)
            retry_count_val = 0
        if retry_count_val < 0:
            logging.warning("retry_count negativo (%d), se usará 0.", retry_count_val)
            retry_count_val = 0
        if retry_count_val > 10:
            logging.info("retry_count=%d es alto, puede prolongar los tiempos de respuesta.", retry_count_val)
    else:
        retry_count_val = 0
    if reflect_budget_val is not None:
        try:
            reflect_budget_val = int(reflect_budget_val)
        except Exception:
            logging.warning("Valor de reflect_budget inválido (%r), usando default 0.", reflect_budget_val)
            reflect_budget_val = 0
        if reflect_budget_val < 0:
            logging.warning("reflect_budget negativo (%d), se usará 0.", reflect_budget_val)
            reflect_budget_val = 0
        if reflect_budget_val > 0 and ReflectionEngine is None:
            logging.info("reflect_budget=%d pero ReflectionEngine no está disponible; ignorado.", reflect_budget_val)
    else:
        reflect_budget_val = 0

    # Loggear configuración final (ocultar detalles sensibles)
    logging.info(
        "Inicializando backend '%s' con configuración: api_url=%s, api_key=%s, temperature=%.3f, top_p=%.3f, max_new_tokens=%d, timeout=%.1f, system_prompt=%s, mode=%s, strict=%s, max_steps=%d, retry_count=%d, reflect_budget=%d",
        backend, api_url_val, ("[SET]" if api_key_val else None),
        temp_val, top_p_val, tokens_val, timeout_val,
        (system_prompt_val[:40] + '...' if system_prompt_val and len(system_prompt_val) > 40 else system_prompt_val),
        mode_val or "(default)", strict_val, max_steps_val, retry_count_val, reflect_budget_val
    )

    def _apply_config_overrides(obj: Any):
        """Aplica mode, strict, max_steps, retry_count, reflect_budget a objeto con .cfg o sus sub-agentes."""
        for field, value in [("mode", mode_val), ("strict", strict_val), ("max_steps", max_steps_val),
                              ("retry_count", retry_count_val), ("reflect_budget", reflect_budget_val)]:
            if value is None:
                continue
            if hasattr(obj, "cfg") and hasattr(obj.cfg, field):
                setattr(obj.cfg, field, value)
            elif hasattr(obj, field):
                try:
                    setattr(obj, field, value)
                except Exception:
                    pass

    backend_instance: Any = None
    if backend == "agent":
        api_cfg = _ApiCfg(
            api_url=api_url_val,
            api_key=api_key_val,
            temperature=temp_val,
            top_p=top_p_val,
            max_new_tokens=tokens_val,
            timeout=timeout_val,
            system_prompt=system_prompt_val,
        )
        if AgentBackend:
            backend_instance = AgentBackend(api_cfg)
        else:
            backend_instance = Agent(AgentConfig(
                api_url=api_cfg.api_url,
                api_key=api_cfg.api_key,
                temperature=api_cfg.temperature,
                top_p=api_cfg.top_p,
                max_new_tokens=api_cfg.max_new_tokens,
                timeout=api_cfg.timeout,
                system_prompt=api_cfg.system_prompt
            ))
        _apply_config_overrides(backend_instance)
        if hasattr(backend_instance, "agent"):
            _apply_config_overrides(backend_instance.agent)
    elif backend == "http":
        class HTTPBackend:
            def __init__(self, api: _ApiCfg):
                self.agent = Agent(AgentConfig(
                    api_url=api.api_url,
                    api_key=api.api_key,
                    temperature=api.temperature,
                    top_p=api.top_p,
                    max_new_tokens=api.max_new_tokens,
                    timeout=api.timeout,
                    system_prompt=api.system_prompt,
                ))
                _apply_config_overrides(self.agent)
            def run(self, goal: str, **kwargs) -> Any:
                if "mode" in kwargs:
                    self.agent.cfg.mode = kwargs["mode"]
                if "strict" in kwargs:
                    self.agent.cfg.strict = kwargs["strict"]
                if "max_steps" in kwargs:
                    self.agent.cfg.max_steps = kwargs["max_steps"]
                if "retry_count" in kwargs:
                    self.agent.cfg.retry_count = kwargs["retry_count"]
                if "reflect_budget" in kwargs:
                    self.agent.cfg.reflect_budget = kwargs["reflect_budget"]
                return self.agent.run(goal)
        backend_instance = HTTPBackend(_ApiCfg(api_url_val, api_key_val, temp_val, top_p_val, tokens_val, timeout_val, system_prompt_val))
    elif backend == "multi_agent":
        class MultiAgentBackend:
            def __init__(self, agent_configs: list[_ApiCfg]):
                self.agents = []
                for cfg in agent_configs:
                    ag = Agent(AgentConfig(
                        api_url=cfg.api_url,
                        api_key=cfg.api_key,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_new_tokens=cfg.max_new_tokens,
                        timeout=cfg.timeout,
                        system_prompt=cfg.system_prompt,
                    ))
                    _apply_config_overrides(ag)
                    self.agents.append(ag)
            def run(self, goal: str, **kwargs) -> Any:
                results = []
                for i, ag in enumerate(self.agents, start=1):
                    logging.info("Ejecutando sub-agente %d con objetivo: %s", i, goal)
                    if "mode" in kwargs: ag.cfg.mode = kwargs["mode"]
                    if "strict" in kwargs: ag.cfg.strict = kwargs["strict"]
                    if "max_steps" in kwargs: ag.cfg.max_steps = kwargs["max_steps"]
                    if "retry_count" in kwargs: ag.cfg.retry_count = kwargs["retry_count"]
                    if "reflect_budget" in kwargs: ag.cfg.reflect_budget = kwargs["reflect_budget"]
                    res = ag.run(goal)
                    results.append(res)
                return results
        agent_cfgs: list[_ApiCfg] = []
        if agent_list_cfg:
            base_cfg = {
                "api_url": api_url_val, "api_key": api_key_val, 
                "temperature": temp_val, "top_p": top_p_val,
                "max_new_tokens": tokens_val, "timeout": timeout_val,
                "system_prompt": system_prompt_val
            }
            for idx, agent_dict in enumerate(agent_list_cfg):
                if not isinstance(agent_dict, dict):
                    logging.warning("Elemento de 'agents' en config no es dict, se ignorará: %r", agent_dict)
                    continue
                merged = base_cfg.copy()
                for k, v in agent_dict.items():
                    merged[k] = v
                cfg_obj = _ApiCfg(
                    api_url=str(merged.get("api_url", api_url_val)),
                    api_key=merged.get("api_key", api_key_val),
                    temperature=float(merged.get("temperature", temp_val)) if merged.get("temperature", temp_val) is not None else None,
                    top_p=float(merged.get("top_p", top_p_val)) if merged.get("top_p", top_p_val) is not None else None,
                    max_new_tokens=int(merged.get("max_new_tokens", tokens_val)) if merged.get("max_new_tokens", tokens_val) is not None else tokens_val,
                    timeout=float(merged.get("timeout", timeout_val)) if merged.get("timeout", timeout_val) is not None else timeout_val,
                    system_prompt=merged.get("system_prompt", system_prompt_val)
                )
                agent_cfgs.append(cfg_obj)
        else:
            count = multi_count_val or 2
            for _ in range(count):
                agent_cfgs.append(_ApiCfg(
                    api_url=api_url_val,
                    api_key=api_key_val,
                    temperature=temp_val,
                    top_p=top_p_val,
                    max_new_tokens=tokens_val,
                    timeout=timeout_val,
                    system_prompt=system_prompt_val,
                ))
        backend_instance = MultiAgentBackend(agent_cfgs)
    elif backend == "planner":
        class PlannerBackend:
            def __init__(self, planner_agent: Any, exec_agent: Any):
                self.planner = planner_agent
                self.executor = exec_agent
            def run(self, goal: str, **kwargs) -> Any:
                logging.info("PlannerBackend: generando plan para el objetivo '%s'", goal)
                plan_result = self.planner.run(f"Desglosa la meta en pasos: {goal}")
                logging.info("Plan generado: %s", plan_result)
                steps = []
                if isinstance(plan_result, str):
                    parts = [p.strip() for p in plan_result.splitlines() if p.strip()]
                    if parts:
                        steps = parts
                if not steps:
                    steps = [goal]
                results = []
                for step in steps:
                    logging.info("Ejecutando paso del plan: %s", step)
                    res = self.executor.run(step, **kwargs)
                    results.append(res)
                return results
        plan_api_cfg = _ApiCfg(
            api_url=api_url_val,
            api_key=api_key_val,
            temperature=temp_val,
            top_p=top_p_val,
            max_new_tokens=tokens_val,
            timeout=timeout_val,
            system_prompt=system_prompt_val
        )
        exec_api_cfg = _ApiCfg(
            api_url=api_url_val,
            api_key=api_key_val,
            temperature=temp_val,
            top_p=top_p_val,
            max_new_tokens=tokens_val,
            timeout=timeout_val,
            system_prompt=system_prompt_val
        )
        if agent_list_cfg:
            if len(agent_list_cfg) >= 2:
                planner_over = agent_list_cfg[0]; exec_over = agent_list_cfg[1]
                if isinstance(planner_over, dict):
                    for k, v in planner_over.items():
                        if hasattr(plan_api_cfg, k):
                            setattr(plan_api_cfg, k, v)
                if isinstance(exec_over, dict):
                    for k, v in exec_over.items():
                        if hasattr(exec_api_cfg, k):
                            setattr(exec_api_cfg, k, v)
                logging.info("Usando configuraciones específicas de archivo para planner y executor.")
            elif len(agent_list_cfg) == 1:
                only_cfg = agent_list_cfg[0]
                if isinstance(only_cfg, dict):
                    for k, v in only_cfg.items():
                        if hasattr(plan_api_cfg, k):
                            setattr(plan_api_cfg, k, v)
                            setattr(exec_api_cfg, k, v)
                logging.info("Usando única configuración proporcionada para ambos planner y executor.")
        planner_agent = Agent(AgentConfig(
            api_url=plan_api_cfg.api_url,
            api_key=plan_api_cfg.api_key,
            temperature=plan_api_cfg.temperature,
            top_p=plan_api_cfg.top_p,
            max_new_tokens=plan_api_cfg.max_new_tokens,
            timeout=plan_api_cfg.timeout,
            system_prompt=plan_api_cfg.system_prompt
        ))
        executor_agent = Agent(AgentConfig(
            api_url=exec_api_cfg.api_url,
            api_key=exec_api_cfg.api_key,
            temperature=exec_api_cfg.temperature,
            top_p=exec_api_cfg.top_p,
            max_new_tokens=exec_api_cfg.max_new_tokens,
            timeout=exec_api_cfg.timeout,
            system_prompt=exec_api_cfg.system_prompt
        ))
        _apply_config_overrides(planner_agent)
        _apply_config_overrides(executor_agent)
        backend_instance = PlannerBackend(planner_agent, executor_agent)
    elif backend == "mock":
        class MockBackend:
            def __init__(self):
                self.last_input = None
            def run(self, goal: str, **kwargs) -> Any:
                self.last_input = goal
                logging.info("MockBackend: recibido goal '%s' (kwargs=%s)", goal, kwargs)
                return {"goal": goal, "response": "OK (mock)", "params": kwargs}
        backend_instance = MockBackend()
    else:
        raise ValueError(f"Tipo de backend desconocido: {backend}")

    for module_name in ("tracing", "telemetry", "hooks"):
        try:
            mod = __import__(module_name)
        except ImportError:
            continue
        try:
            if hasattr(mod, "inject"):
                mod.inject(backend_instance)
                logging.info("Módulo '%s' injectado en el backend.", module_name)
            elif hasattr(mod, "initialize"):
                mod.initialize(backend_instance)
                logging.info("Módulo '%s' inicializado con el backend.", module_name)
            elif hasattr(mod, "setup"):
                mod.setup(backend_instance)
                logging.info("Módulo '%s' configurado con el backend.", module_name)
            else:
                logging.info("Módulo opcional '%s' importado (no se encontró función de inyección específica).", module_name)
        except Exception as e:
            logging.warning("Falló la inyección del módulo '%s': %s", module_name, e)

    return backend_instance

def build_backend_from_cli(args: Any) -> BackendProtocol:
    """
    Helper para construir un backend a partir de un objeto de argumentos (ej. de argparse).
    Ejemplo de uso: backend = build_backend_from_cli(args)
    """
    return get_backend(
        backend=getattr(args, "backend", None),
        api_url=getattr(args, "api_url", None),
        api_key=getattr(args, "api_key", None),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        max_new_tokens=getattr(args, "max_new_tokens", None),
        timeout=getattr(args, "timeout", None),
        system_prompt=getattr(args, "system_prompt", None),
        mode=getattr(args, "mode", None),
        strict=getattr(args, "strict", None),
        max_steps=getattr(args, "max_steps", None),
        retry_count=getattr(args, "retry_count", None),
        reflect_budget=getattr(args, "reflect_budget", None),
        config_file=getattr(args, "config", None),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inicializador de AGI - crea el backend especificado con la configuración dada.")
    parser.add_argument("--backend", choices=["agent", "http", "multi_agent", "planner", "mock"], help="Tipo de backend a usar.")
    parser.add_argument("--config", dest="config", type=str, help="Ruta a archivo de configuración (.json o .yaml).")
    parser.add_argument("--api_url", dest="api_url", type=str, help="URL del API de lenguaje.")
    parser.add_argument("--api_key", dest="api_key", type=str, help="Clave de API para el servicio de lenguaje.")
    parser.add_argument("--temperature", dest="temperature", type=float, help="Temperatura de muestreo (0.0 a 1.0).")
    parser.add_argument("--top_p", dest="top_p", type=float, help="Top-p (fracción de nucleó) para muestreo.")
    parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, help="Máx tokens a generar en la respuesta.")
    parser.add_argument("--timeout", dest="timeout", type=float, help="Timeout en segundos para la llamada al modelo.")
    parser.add_argument("--system_prompt", dest="system_prompt", type=str, help="Prompt de sistema inicial para el agente.")
    parser.add_argument("--mode", dest="mode", type=str, help="Modo de operación del agente.")
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true", help="Activa el modo estricto.")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false", help="Desactiva el modo estricto.")
    parser.set_defaults(strict=None)
    parser.add_argument("--max_steps", dest="max_steps", type=int, help="Máximo número de pasos del agente.")
    parser.add_argument("--retry_count", dest="retry_count", type=int, help="Cantidad de reintentos en caso de falla.")
    parser.add_argument("--reflect_budget", dest="reflect_budget", type=int, help="Presupuesto de iteraciones de reflexión.")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        backend_obj = build_backend_from_cli(args)
    except Exception as e:
        logging.error("Error al inicializar el backend: %s", e)
        sys.exit(1)
    logging.info("Backend '%s' inicializado correctamente.", getattr(args, "backend", None) or type(backend_obj).__name__)
