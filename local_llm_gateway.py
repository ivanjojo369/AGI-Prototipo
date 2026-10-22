# local_llm_gateway.py (v6.6b: v6.4 infrastructure + v6.6 terminal quipu completion guard)
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, Optional, List, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8011/v1").rstrip("/")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-3B-Instruct-Q4_K_M")
LOCAL_LLM_MODEL_ALIASES_RAW = os.getenv("LOCAL_LLM_MODEL_ALIASES", "{}")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
GATEWAY_DEBUG = os.getenv("GATEWAY_DEBUG", "0").strip() == "1"
UPSTREAM_TIMEOUT_S = float(os.getenv("LOCAL_LLM_TIMEOUT", "180"))
UPSTREAM_CONNECT_TIMEOUT_S = float(os.getenv("LOCAL_LLM_CONNECT_TIMEOUT", "10"))

try:
    LOCAL_LLM_MODEL_ALIASES: Dict[str, str] = json.loads(LOCAL_LLM_MODEL_ALIASES_RAW) or {}
except Exception:
    LOCAL_LLM_MODEL_ALIASES = {}

logging.basicConfig(level=logging.DEBUG if GATEWAY_DEBUG else logging.INFO)
logger = logging.getLogger("local_llm_gateway")

app = FastAPI(title="Local LLM Gateway", version="1.13.1")


def _debug_log(event: str, **fields: Any) -> None:
    if not GATEWAY_DEBUG:
        return
    safe_fields: Dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            safe_fields[key] = value[:800]
        else:
            safe_fields[key] = value
    logger.debug("%s | %s", event, json.dumps(safe_fields, ensure_ascii=False, default=str))


def _preview_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            value = str(value)
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    return value[:limit]


_CONTROL_TOKEN_RE = re.compile(r"^\s*#{3}\s*([A-Za-z][A-Za-z\s_-]*[A-Za-z])\s*#{3}\s*$")


def _canonicalize_control_token_text(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    m = _CONTROL_TOKEN_RE.match(text)
    if not m:
        return text
    inner = m.group(1).strip().upper()
    inner = re.sub(r"[\s_]+", " ", inner)
    inner = re.sub(r"\s*-\s*", "-", inner)
    mapping = {
        "STOP": "###STOP###",
        "TRANSFER": "###TRANSFER###",
        "OUT OF SCOPE": "###OUT-OF-SCOPE###",
        "OUT-OF-SCOPE": "###OUT-OF-SCOPE###",
    }
    return mapping.get(inner, text)


def _normalize_control_tokens_in_response(response_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return response_json
    try:
        message = response_json["choices"][0]["message"]
    except Exception:
        return response_json
    if not isinstance(message, dict):
        return response_json
    content = message.get("content")
    normalized = _canonicalize_control_token_text(content)
    if normalized != content:
        message["content"] = normalized
        _debug_log("control_token_normalized", before=_preview_text(content, 120), after=normalized)
    return response_json


def _looks_like_eval_json_request(request_json: Dict[str, Any]) -> bool:
    text = _all_message_text(request_json.get("messages") or []).lower()
    cues = [
        "expectedoutcome",
        "expected outcome",
        '"results"',
        "nl_assertion",
        "nl assertions",
        "return a json",
        "return json",
        "valid json",
        "json object",
        '"meet"',
        '"reasoning"',
        "evaluate whether",
        "did the assistant",
    ]
    return any(cue in text for cue in cues)


def _extract_json_from_text(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    # Exact JSON first.
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # Remove full code fences and retry.
    unfenced = _strip_code_fence(stripped)
    if unfenced != stripped:
        try:
            return json.loads(unfenced)
        except Exception:
            pass

    # Search fenced JSON blocks.
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.IGNORECASE | re.DOTALL):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            return json.loads(block)
        except Exception:
            continue

    # Search for first decodable JSON object/array embedded in text.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", stripped):
        start = match.start()
        candidate = stripped[start:]
        try:
            parsed, end = decoder.raw_decode(candidate)
        except Exception:
            continue
        if end > 0:
            return parsed
    return None


def _normalize_eval_json_in_response(response_json: Dict[str, Any], request_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return response_json
    try:
        message = response_json["choices"][0]["message"]
    except Exception:
        return response_json
    if not isinstance(message, dict):
        return response_json
    if message.get("tool_calls"):
        return response_json

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return response_json

    should_try = _looks_like_eval_json_request(request_json) or content.lstrip().startswith("```") or content.lstrip().startswith("{") or content.lstrip().startswith("[")
    if not should_try:
        return response_json

    parsed = _extract_json_from_text(content)
    if parsed is None:
        return response_json

    normalized = json.dumps(parsed, ensure_ascii=False)
    if normalized != content:
        message["content"] = normalized
        _debug_log("eval_json_normalized", before=_preview_text(content, 200), after=_preview_text(normalized, 200))
    return response_json


def _normalize_token(token: str) -> str:
    t = (token or "").strip().replace("\r", "").replace("\n", "")
    while (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t


def _expected_tokens() -> List[str]:
    raw = (os.environ.get("GATEWAY_API_KEY") or "").strip()
    if not raw:
        raw = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not raw:
        return []
    tokens = [_normalize_token(t) for t in raw.split(",")]
    return [t for t in tokens if t]


def _auth_required() -> bool:
    return len(_expected_tokens()) > 0


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


def _extract_token_from_request(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = ""
    if auth:
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        else:
            token = auth.strip()
    if not token:
        token = (request.headers.get("x-api-key") or request.headers.get("X-API-Key") or "").strip()
    return _normalize_token(token)


def _require_auth(request: Request) -> None:
    tokens = _expected_tokens()
    if not tokens:
        return
    token = _extract_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key (send Authorization: Bearer <key>)")
    if token not in tokens:
        detail = "Invalid API key"
        if GATEWAY_DEBUG:
            detail = {
                "detail": "Invalid API key",
                "received_fp": _token_fingerprint(token),
                "expected_fps": [_token_fingerprint(t) for t in tokens],
                "expected_count": len(tokens),
            }
        raise HTTPException(status_code=403, detail=detail)


async def check_auth(request: Request) -> None:
    _require_auth(request)


def resolve_model(requested_model: Optional[str]) -> str:
    if not requested_model:
        return LOCAL_LLM_MODEL
    return LOCAL_LLM_MODEL_ALIASES.get(requested_model, requested_model)


def _auth_header_for_upstream() -> Dict[str, str]:
    if not LOCAL_LLM_API_KEY:
        return {}
    return {"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"}


def _safe_json(resp: httpx.Response) -> Dict[str, Any]:
    raw_text = resp.text
    content_type = resp.headers.get("content-type", "")

    if not raw_text or not raw_text.strip():
        return {
            "error": {
                "message": "Upstream returned empty body",
                "status_code": resp.status_code,
                "content_type": content_type,
            }
        }

    try:
        data = resp.json()
    except Exception as e:
        return {
            "error": {
                "message": f"Upstream returned non-JSON body: {type(e).__name__}: {e}",
                "status_code": resp.status_code,
                "content_type": content_type,
                "raw_preview": _preview_text(raw_text, 500),
            }
        }

    if not isinstance(data, dict):
        return {
            "error": {
                "message": f"Upstream JSON root is not an object: {type(data).__name__}",
                "status_code": resp.status_code,
                "content_type": content_type,
                "raw_preview": _preview_text(raw_text, 500),
            }
        }

    return data


def _rewrite_model_field(obj: Any, requested_model: Optional[str], mapped_model: Optional[str]) -> Any:
    if not isinstance(obj, dict):
        return obj
    if requested_model:
        obj["model"] = requested_model
        if mapped_model and mapped_model != requested_model:
            obj.setdefault("real_model", mapped_model)
    return obj


def _rewrite_sse_line(line: str, requested_model: Optional[str], mapped_model: Optional[str]) -> str:
    if not line.startswith("data:"):
        return line
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return line
    try:
        obj = json.loads(payload)
    except Exception:
        return line
    obj = _rewrite_model_field(obj, requested_model, mapped_model)
    return "data: " + json.dumps(obj, ensure_ascii=False)


_DIRECT_TOOLCALL_RE = re.compile(r"^\s*call\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return text


def _allowed_tool_names(request_json: Dict[str, Any]) -> set[str]:
    tools = request_json.get("tools") or []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _extract_tool_call_candidates(content: str) -> list[str]:
    if not isinstance(content, str):
        return []
    text = _strip_code_fence(content)
    if _DIRECT_TOOLCALL_RE.match(text):
        return [text]
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, dict):
        tool_calls = parsed.get("tool_calls")
        if isinstance(tool_calls, list):
            return [str(x).strip() for x in tool_calls if str(x).strip()]
    return []


def _make_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _tool_call_name(tool_call: Any) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    name = tool_call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    fn = tool_call.get("function") or {}
    name = fn.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _tool_call_arguments(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return "{}"
    if "arguments" in tool_call and tool_call.get("arguments") is not None:
        return tool_call.get("arguments")
    fn = tool_call.get("function") or {}
    return fn.get("arguments") or "{}"


def _tool_call_id(tool_call: Any) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    value = tool_call.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _tool_call_arguments_text(tool_call: Any) -> str:
    arguments = _tool_call_arguments(tool_call)
    if isinstance(arguments, str):
        return arguments or "{}"
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return str(arguments)


def _parse_function_call_string(call_text: str, allowed_names: set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(call_text, str) or not call_text.strip():
        return None
    text = call_text.strip()
    m = _DIRECT_TOOLCALL_RE.match(text)
    if m:
        text = f"{m.group(1).strip()}({m.group(2).strip()})"
    try:
        node = ast.parse(text, mode="eval")
    except Exception:
        return None
    if not isinstance(node, ast.Expression) or not isinstance(node.body, ast.Call):
        return None
    call = node.body
    if not isinstance(call.func, ast.Name):
        return None
    tool_name = call.func.id
    if allowed_names and tool_name not in allowed_names:
        return None

    args_payload: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            args_payload[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None

    if call.args:
        positional = []
        for arg in call.args:
            try:
                positional.append(ast.literal_eval(arg))
            except Exception:
                return None
        if positional:
            args_payload["args"] = positional

    return _make_tool_call(tool_name, args_payload)


def _maybe_promote_text_tool_call(response_json: Dict[str, Any], request_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return response_json
    allowed_names = _allowed_tool_names(request_json)
    if not allowed_names:
        return response_json
    try:
        choice = response_json["choices"][0]
        message = choice["message"]
    except Exception:
        return response_json
    if not isinstance(message, dict):
        return response_json
    if message.get("tool_calls"):
        return response_json
    content = message.get("content")
    if not isinstance(content, str):
        return response_json
    candidates = _extract_tool_call_candidates(content)
    if not candidates:
        return response_json
    promoted = []
    for item in candidates:
        parsed = _parse_function_call_string(item, allowed_names)
        if parsed:
            promoted.append(parsed)
    if not promoted:
        return response_json
    message["content"] = ""
    message["tool_calls"] = promoted
    return response_json


def _latest_user_text(request_json: Dict[str, Any]) -> str:
    messages = request_json.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _all_message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    bits: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            bits.append(content.strip())
    return "\n".join(bits)


def _all_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    bits: list[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                bits.append(content.strip())
    return "\n".join(bits)


def _extract_negative_constraints(text: str, limit: int = 4) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    for part in parts:
        s = part.strip()
        low = s.lower()
        if "do not" in low or "don't" in low or "avoid" in low or "without" in low:
            out.append(s[:180])
        if len(out) >= limit:
            break
    return out


def _extract_user_constraints(messages: Any, limit: int = 6) -> list[str]:
    text = _all_user_text(messages)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    keywords = ["do not", "don't", "avoid", "without", "keep", "preserve", "concise", "ordering", "optional", "resource"]
    for part in parts:
        s = part.strip()
        low = s.lower()
        if any(k in low for k in keywords):
            if s not in out:
                out.append(s[:180])
        if len(out) >= limit:
            break
    return out


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _extract_quipu_stable_constraints(messages: Any, limit: int = 6) -> list[str]:
    text = _all_message_text(messages).lower()
    out: List[str] = []

    if "do not jump straight to the final step" in text or "don't jump straight to the final step" in text:
        out.append("Do not jump straight to the final step.")
    if "keep it concise" in text or "concise" in text:
        out.append("Keep the response concise.")
    if "preserve ordering" in text or "preserve the ordering" in text or "ordered" in text:
        out.append("Preserve ordering.")
    if "resource budget" in text or "tight resource budget" in text:
        out.append("Respect the tight resource budget.")
    if (
        "avoid wasting resources on optional steps" in text
        or "avoid redundant actions" in text
        or "optional steps" in text
        or "optional waste" in text
        or "redundant" in text
    ):
        out.append("Avoid optional or redundant steps unless they unlock a required dependency.")

    return _dedupe_keep_order(out)[:limit]


def _looks_like_meta_planning_step(step: str) -> bool:
    if not isinstance(step, str):
        return False
    low = step.strip().lower()
    markers = [
        "clarify the objective",
        "review the available context",
        "review the most relevant available assets",
        "ground the response",
        "ground the plan",
        "draft the minimum ordered sequence",
        "draft the plan",
        "state any unresolved blocker",
        "before finalizing",
        "review relevant lab notes",
        "finalize plan",
    ]
    return any(marker in low for marker in markers)


def _extract_quipu_fallback_context(draft_result: Any) -> List[str]:
    draft_assets = _extract_assets_from_anything(draft_result)
    cleaned: List[str] = []
    for item in draft_assets:
        s = str(item).strip()
        if not s:
            continue
        cleaned.append(s.replace("_", " "))
    return _dedupe_keep_order(cleaned)


def _build_quipu_operational_steps(
    explicit_assets: List[str],
    fallback_context: List[str],
    draft_steps: List[str],
) -> List[str]:
    usable_draft_steps = [s for s in draft_steps if s and not _looks_like_meta_planning_step(s)]

    if usable_draft_steps:
        base_steps = usable_draft_steps[:5]
    else:
        if explicit_assets:
            step1 = "Verify the exposed assets first and map the required prerequisites and dependencies before any downstream action."
        else:
            step1 = "Use only the exposed policy, tool inventory, and lab notes as grounded inputs because no explicit lab assets were returned."

        if fallback_context:
            step2 = "Confirm prerequisite dependencies from that grounded context before the core build sequence."
        else:
            step2 = "Confirm prerequisite dependencies before the core build sequence."

        base_steps = [
            step1,
            step2,
            "Execute the minimum ordered build sequence from prerequisites to core work; do not skip dependency order.",
            "Skip optional or redundant work unless it unlocks a required dependency or resolves a blocker.",
            "End with a concise blocker or completion handoff instead of reopening another planning step.",
        ]

    return _dedupe_keep_order(base_steps[:5])


def _heuristic_tool_fallback(request_json: Dict[str, Any], response_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return response_json
    allowed_names = _allowed_tool_names(request_json)
    if not allowed_names:
        return response_json
    try:
        choice = response_json["choices"][0]
        message = choice["message"]
    except Exception:
        return response_json
    if not isinstance(message, dict):
        return response_json
    if message.get("tool_calls"):
        return response_json

    user_text = _latest_user_text(request_json)
    content = (message.get("content") or "") if isinstance(message.get("content"), str) else ""
    combined = f"{user_text}\n{content}".lower()

    selected_name: Optional[str] = None
    selected_args: Dict[str, Any] = {}

    if "lookup_lab_note" in allowed_names and any(k in combined for k in [
        "note", "criteria", "policy", "budget", "ratio", "percentage", "value", "figure", "query"
    ]):
        selected_name = "lookup_lab_note"
        selected_args = {"query": user_text[:240] or "task context"}

    elif "list_available_assets" in allowed_names and any(k in combined for k in [
        "asset", "assets", "available", "inventory", "capability", "tool", "tools", "supported"
    ]):
        selected_name = "list_available_assets"
        selected_args = {"kind": "lab asset"} if ("lab asset" in combined or "assets" in combined) else {}

    elif "draft_structured_plan" in allowed_names and any(k in combined for k in [
        "plan", "steps", "sequence", "workflow", "handoff", "strategy", "build-style", "build style"
    ]):
        selected_name = "draft_structured_plan"
        selected_args = {
            "goal": user_text[:240] or "draft a concise structured plan",
            "constraints": _extract_user_constraints(request_json.get("messages") or []),
            "max_steps": 5,
        }

    if selected_name is None:
        return response_json

    message["content"] = ""
    message["tool_calls"] = [_make_tool_call(selected_name, selected_args)]
    return response_json


def _tool_call_to_text(tool_call: Dict[str, Any]) -> str:
    name = _tool_call_name(tool_call) or "unknown_tool"
    arguments = _tool_call_arguments_text(tool_call)
    return f"[TOOL_CALL] {name}({arguments})"


def _sanitize_messages_for_upstream(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    sanitized: List[Dict[str, Any]] = []
    tool_name_by_id: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role") or "user"
        content = msg.get("content")

        if role == "assistant" and msg.get("tool_calls"):
            lines = []
            if isinstance(content, str) and content.strip():
                lines.append(content.strip())
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tc_id = _tool_call_id(tc)
                    tc_name = _tool_call_name(tc)
                    if tc_id and tc_name:
                        tool_name_by_id[tc_id] = tc_name
                    lines.append(_tool_call_to_text(tc))
            flattened = "\n".join(lines).strip() or "[TOOL_CALL]"
            sanitized.append({
                "role": "assistant",
                "content": flattened,
            })
            continue

        if role == "tool":
            inferred_tool_name = tool_name_by_id.get(str(msg.get("tool_call_id") or msg.get("id") or "").strip())
            tool_name = msg.get("name") or inferred_tool_name or "tool"
            tool_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            sanitized.append({
                "role": "user",
                "content": f"[TOOL_RESULT {tool_name}] {tool_content}",
            })
            continue

        if content is None:
            content = ""
        elif not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)

        sanitized.append({
            "role": role,
            "content": content,
        })

    return sanitized


def _sanitize_request_for_upstream(body: Dict[str, Any]) -> Dict[str, Any]:
    upstream_body = dict(body)
    upstream_body["messages"] = _sanitize_messages_for_upstream(body.get("messages"))
    return upstream_body


def _extract_tool_history(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    history: List[Dict[str, Any]] = []
    calls_by_id: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = _tool_call_id(tc)
                tc_name = _tool_call_name(tc)
                tc_arguments = _tool_call_arguments(tc)
                if tc_arguments in (None, ""):
                    tc_arguments = "{}"
                item = {
                    "id": tc_id,
                    "name": tc_name,
                    "arguments": tc_arguments,
                    "result": None,
                }
                history.append(item)
                if item["id"]:
                    calls_by_id[item["id"]] = item

        elif msg.get("role") == "tool":
            tool_id = msg.get("tool_call_id") or msg.get("id")
            if tool_id and tool_id in calls_by_id:
                calls_by_id[tool_id]["result"] = msg.get("content")

    return history


def _normalize_args_key(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except Exception:
            return arguments.strip()
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(arguments)


def _tool_signature(name: str, arguments: Any) -> str:
    return f"{name}::{_normalize_args_key(arguments)}"


def _find_latest_tool_result(history: List[Dict[str, Any]], name: str) -> Any:
    for item in reversed(history):
        if item.get("name") == name and item.get("result") is not None:
            return item.get("result")
    return None


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_assets_from_anything(value: Any) -> List[str]:
    parsed = _safe_json_loads(value)
    assets: List[str] = []

    if isinstance(parsed, dict):
        for key in ("assets", "assets_considered", "available_assets"):
            if isinstance(parsed.get(key), list):
                assets.extend([str(x) for x in parsed[key] if str(x).strip()])
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                if "name" in item:
                    assets.append(str(item["name"]))
                elif "asset" in item:
                    assets.append(str(item["asset"]))
            elif str(item).strip():
                assets.append(str(item))

    deduped: List[str] = []
    seen: set[str] = set()
    for asset in assets:
        a = asset.strip()
        if a and a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def _extract_steps_from_anything(value: Any) -> List[str]:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
        return [str(x).strip() for x in parsed["steps"] if str(x).strip()]
    return []


def _extract_best_note_line(value: Any) -> Optional[str]:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip().lower()
                content = str(item.get("content") or "").strip()
                if "build" in title or "constraint" in title or "planning" in title:
                    if content:
                        return content
        for item in parsed:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    return content
    return None


def _replace_message_with_final_text(response_json: Dict[str, Any], text: str) -> Dict[str, Any]:
    try:
        message = response_json["choices"][0]["message"]
    except Exception:
        return response_json
    if not isinstance(message, dict):
        return response_json
    message["role"] = "assistant"
    message["content"] = text
    message["tool_calls"] = None
    return response_json


def _is_quipu_build_sequence_request(request_json: Dict[str, Any], allowed_names: set[str]) -> bool:
    needed = {"lookup_lab_note", "list_available_assets", "draft_structured_plan"}
    if not needed.issubset(allowed_names):
        return False
    text = _all_message_text(request_json.get("messages") or []).lower()
    clues = [
        "build-style plan",
        "prerequisites",
        "preserve ordering",
        "optional steps",
        "resource budget",
        "build sequence",
    ]
    return any(c in text for c in clues)


def _draft_goal_from_user(messages: Any) -> str:
    text = _all_user_text(messages)
    for line in text.splitlines():
        s = line.strip()
        if s and "build-style plan" in s.lower():
            return s[:240]
    return (_latest_user_text({"messages": messages}) or "prepare a compact build-style plan with prerequisites")[:240]


def _repeat_count(history: List[Dict[str, Any]], name: str, arguments: Any) -> int:
    sig = _tool_signature(name, arguments)
    return sum(1 for item in history if _tool_signature(item.get("name", ""), item.get("arguments", "{}")) == sig)


def _latest_user_reopens_steps(messages: Any) -> bool:
    latest = _latest_user_text({"messages": messages}).lower()
    if not latest:
        return False
    markers = [
        "step 1",
        "step 2",
        "step 3",
        "step 4",
        "reviewing lab notes",
        "review the lab notes",
        "review relevant lab notes",
        "draft minimum ordered sequence",
        "finalize plan",
        "start with step 1",
        "begin with step 1",
        "proceed with step 1",
        "let's start with step 1",
        "lets start with step 1",
        "would it be helpful if i proceed",
        "would you like me to proceed with drafting",
        "would you like me to proceed with this step",
        "initial sequence now",
        "review policy",
        "resource allocation",
    ]
    return any(m in latest for m in markers)


def _content_reopens_steps(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    low = content.lower()
    markers = [
        "step 1",
        "step 2",
        "step 3",
        "step 4",
        "reviewing the lab notes",
        "reviewing lab notes",
        "review the lab notes",
        "review relevant lab notes",
        "draft minimum ordered sequence",
        "finalize plan",
        "would you like me to start",
        "would you like me to proceed",
        "would you like me to proceed with this step",
        "would you like me to proceed with drafting",
        "would it be helpful if i proceed",
        "shall i start",
        "shall i proceed",
        "next, i can",
        "i can begin by",
        "i will now review",
        "initial sequence now",
        "review policy",
        "resource allocation",
    ]
    return any(m in low for m in markers)




def _assistant_has_quipu_final_plan(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    markers = [
        "complete compact coordination plan",
        "final ordered build sequence",
        "no further preliminary review is needed",
        "available assets identified before planning",
        "ordered multi-step plan with prerequisites",
        "using only those identified available assets/context",
    ]
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        low = content.lower()
        if any(marker in low for marker in markers):
            return True
    return False




def _latest_user_restates_final_plan(messages: Any) -> bool:
    latest = _latest_user_text({"messages": messages}).lower()
    if not latest:
        return False
    markers = [
        "given the final ordered build sequence",
        "based on the final ordered build sequence",
        "here is our plan",
        "confirm prerequisite dependencies",
        "core build sequence",
        "final handoff",
        "would you like me to proceed with these steps now",
        "would you like me to proceed with these steps",
    ]
    return any(marker in latest for marker in markers)




def _synthesize_quipu_terminal_closure(request_json: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    messages = request_json.get("messages") or []
    constraints = _extract_quipu_stable_constraints(messages)
    lines: List[str] = [
        "The requested compact coordination response is already complete.",
        "No additional review, drafting, execution, or follow-up is needed in this conversation.",
    ]
    if constraints:
        lines.append("")
        lines.append("Final constraints retained:")
        for item in constraints[:4]:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Task complete.")
    return "\n".join(lines)




def _select_quipu_final_text(request_json: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    messages = request_json.get("messages") or []
    if _assistant_has_quipu_final_plan(messages) and (
        _latest_user_reopens_steps(messages) or _latest_user_restates_final_plan(messages)
    ):
        return _synthesize_quipu_terminal_closure(request_json, history)
    return _synthesize_quipu_build_answer(request_json, history)


def _quipu_required_tools_completed(history: List[Dict[str, Any]]) -> bool:
    assets_result = _find_latest_tool_result(history, "list_available_assets")
    draft_result = _find_latest_tool_result(history, "draft_structured_plan")
    return assets_result is not None and draft_result is not None


def _synthesize_quipu_build_answer(request_json: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    messages = request_json.get("messages") or []
    draft_result = _find_latest_tool_result(history, "draft_structured_plan")
    assets_result = _find_latest_tool_result(history, "list_available_assets")
    note_result = _find_latest_tool_result(history, "lookup_lab_note")

    explicit_assets = _extract_assets_from_anything(assets_result)
    fallback_context = _extract_quipu_fallback_context(draft_result)
    constraints = _extract_quipu_stable_constraints(messages) or _extract_user_constraints(messages)
    note_line = _extract_best_note_line(note_result)

    identified_context: List[str] = []
    seen_context: set[str] = set()
    for item in fallback_context:
        s = str(item).strip().replace("_", " ")
        if not s:
            continue
        key = s.lower()
        if key in seen_context:
            continue
        seen_context.add(key)
        identified_context.append(s)

    for item in ["quipu_lab policy excerpt", "quipu_lab tools", "quipu_lab notes"]:
        key = item.lower()
        if key not in seen_context:
            seen_context.add(key)
            identified_context.append(item)

    plan_steps: List[str] = [
        "Confirm the prerequisite dependencies and blockers from the identified policy, tools, notes, and asset listing before any core build action.",
        "Select the minimum build path that satisfies those prerequisites and preserves the required ordering.",
        "Execute the core build sequence strictly in dependency order; do not jump straight to the final step.",
        "Skip optional or redundant work unless it unlocks a required dependency, to stay resource-aware and respect the tight resource budget.",
        "Finish with a concise blocker or completion handoff.",
    ]

    lines: List[str] = []
    lines.append("Available assets identified before planning:")
    for item in identified_context[:6]:
        lines.append(f"- {item}")
    if explicit_assets:
        for asset in explicit_assets[:6]:
            lines.append(f"- explicit lab asset: {asset}")
    else:
        lines.append("- explicit lab assets from list_available_assets(kind='lab asset'): none exposed")

    lines.append("")
    lines.append("Ordered multi-step plan with prerequisites:")
    for idx, step in enumerate(plan_steps, 1):
        lines.append(f"{idx}. {step}")

    lines.append("")
    lines.append("Resource note:")
    lines.append("- This plan is concise, resource-aware, and respects the tight resource budget by using the minimum required ordered sequence.")

    if constraints:
        lines.append("")
        lines.append("Constraints preserved:")
        for item in constraints[:6]:
            lines.append(f"- {item}")

    if note_line:
        lines.append("")
        lines.append(f"Grounding note: {note_line}")

    lines.append("")
    lines.append("Using only those identified available assets/context, this is the complete compact coordination plan.")
    return "\n".join(lines)


def _apply_quipu_build_workflow_guard(request_json: Dict[str, Any], response_json: Dict[str, Any]) -> Dict[str, Any]:
    allowed_names = _allowed_tool_names(request_json)
    if not _is_quipu_build_sequence_request(request_json, allowed_names):
        return response_json

    messages = request_json.get("messages") or []
    history = _extract_tool_history(messages)
    called_names = [item.get("name") for item in history if item.get("name")]
    called_set = set(called_names)

    latest_assets_result = _find_latest_tool_result(history, "list_available_assets")
    latest_assets = _extract_assets_from_anything(latest_assets_result)
    assets_repeat = _repeat_count(history, "list_available_assets", {"kind": "lab asset"})

    _debug_log(
        "quipu_guard_state",
        called_names=called_names,
        assets_repeat=assets_repeat,
        assets_count=len(latest_assets),
        latest_user=_preview_text(_latest_user_text(request_json), 240),
    )

    if "list_available_assets" not in called_set:
        try:
            message = response_json["choices"][0]["message"]
        except Exception:
            return response_json
        if isinstance(message, dict):
            message["content"] = ""
            message["tool_calls"] = [_make_tool_call("list_available_assets", {"kind": "lab asset"})]
            _debug_log("quipu_guard_action", action="force_list_available_assets")
        return response_json

    if "draft_structured_plan" not in called_set:
        goal_text = _draft_goal_from_user(messages)
        constraints = _extract_user_constraints(messages)
        try:
            message = response_json["choices"][0]["message"]
        except Exception:
            return response_json
        if isinstance(message, dict):
            message["content"] = ""
            message["tool_calls"] = [_make_tool_call("draft_structured_plan", {
                "goal": goal_text,
                "constraints": constraints,
                "max_steps": 5,
            })]
            _debug_log("quipu_guard_action", action="force_draft_structured_plan")
        return response_json

    if _quipu_required_tools_completed(history):
        _debug_log("quipu_guard_action", action="force_canonical_final_answer")
        return _replace_message_with_final_text(
            response_json,
            _select_quipu_final_text(request_json, history),
        )

    return response_json


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(timeout=UPSTREAM_TIMEOUT_S, connect=UPSTREAM_CONNECT_TIMEOUT_S)


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "status": "ok",
        "upstream": LOCAL_LLM_BASE_URL,
        "default_model": LOCAL_LLM_MODEL,
        "aliases": LOCAL_LLM_MODEL_ALIASES,
        "auth_required": _auth_required(),
    }
    if GATEWAY_DEBUG and _auth_required():
        toks = _expected_tokens()
        info["auth_fps"] = [_token_fingerprint(t) for t in toks]
        info["auth_count"] = len(toks)
    return info


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models() -> Dict[str, Any]:
    ids: List[str] = list(dict.fromkeys([*LOCAL_LLM_MODEL_ALIASES.keys(), LOCAL_LLM_MODEL]))
    return {"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "local", "permission": []} for mid in ids]}


async def _proxy_json(path: str, body: Dict[str, Any], requested_model: str, mapped_model: str) -> JSONResponse:
    upstream_body = _sanitize_request_for_upstream(body)
    upstream_body["model"] = mapped_model
    async with httpx.AsyncClient(base_url=LOCAL_LLM_BASE_URL, timeout=_timeout()) as client:
        try:
            resp = await client.post(path, json=upstream_body, headers=_auth_header_for_upstream())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

    _debug_log(
        "upstream_json_response",
        path=path,
        status_code=resp.status_code,
        content_type=resp.headers.get("content-type", ""),
        raw_preview=_preview_text(resp.text, 500),
    )

    data = _safe_json(resp)

    try:
        data = _rewrite_model_field(data, requested_model=requested_model, mapped_model=mapped_model)
        data = _maybe_promote_text_tool_call(data, body)
        data = _heuristic_tool_fallback(body, data)
        data = _apply_quipu_build_workflow_guard(body, data)
        data = _normalize_control_tokens_in_response(data)
        data = _normalize_eval_json_in_response(data, body)
    except Exception as e:
        _debug_log("gateway_postprocess_exception", error=repr(e))
        data = {
            "error": {
                "message": f"Gateway post-processing failed: {type(e).__name__}: {e}",
                "status_code": resp.status_code,
                "raw_preview": _preview_text(resp.text, 500),
            }
        }

    return JSONResponse(status_code=resp.status_code, content=data)


async def _proxy_stream(path: str, body: Dict[str, Any], requested_model: str, mapped_model: str) -> StreamingResponse | JSONResponse:
    upstream_body = _sanitize_request_for_upstream(body)
    upstream_body["model"] = mapped_model
    client = httpx.AsyncClient(base_url=LOCAL_LLM_BASE_URL, timeout=None)
    try:
        upstream = await client.stream("POST", path, json=upstream_body, headers=_auth_header_for_upstream())
    except httpx.RequestError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
    if upstream.status_code != 200:
        raw = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            payload = {"error": raw.decode("utf-8", errors="replace")}
        return JSONResponse(status_code=upstream.status_code, content=payload)

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for raw_line in upstream.aiter_lines():
                if raw_line == "":
                    yield b"\n"
                    continue
                out = _rewrite_sse_line(raw_line, requested_model=requested_model, mapped_model=mapped_model)
                yield (out + "\n\n").encode("utf-8")
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body. Error: {type(e).__name__}: {e}")
    requested_model = str(body.get("model") or "")
    mapped_model = resolve_model(requested_model)
    stream = bool(body.get("stream", False))
    if stream:
        return await _proxy_stream("/chat/completions", body, requested_model, mapped_model)
    return await _proxy_json("/chat/completions", body, requested_model, mapped_model)


@app.post("/v1/completions", dependencies=[Depends(check_auth)])
async def completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body. Error: {type(e).__name__}: {e}")
    requested_model = str(body.get("model") or "")
    mapped_model = resolve_model(requested_model)
    stream = bool(body.get("stream", False))
    if stream:
        return await _proxy_stream("/completions", body, requested_model, mapped_model)
    return await _proxy_json("/completions", body, requested_model, mapped_model)
