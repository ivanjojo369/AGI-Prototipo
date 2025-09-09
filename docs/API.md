# API
## Endpoints
- `GET /` → health
- `POST /chat` → inferencia
- `POST /memory/semantic/upsert`
- `POST /memory/semantic/search`

## /chat (ejemplo)
Body:
{
  "messages": [
    {"role":"system","content":"Asistente conciso."},
    {"role":"user","content":"hola"}
  ],
  "params":{
    "max_new_tokens":100,"temperature":0.6,"top_p":0.9,"top_k":40,
    "min_p":0.05,"repeat_penalty":1.08,
    "stop":["<|end_of_turn|>","</s>"],"stream":false
  }
}

## /memory/semantic/upsert (ejemplo)
{"facts":[{"text":"Prefiere respuestas concisas.","tags":["pref","style"]}]}

## /memory/semantic/search (ejemplo)
{"q":"concisas","k":3}
