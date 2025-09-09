# Arquitectura
## Componentes
- FastAPI (`/chat`, `/memory/semantic/*`)
- llama.cpp (OpenChat-3.5-1210)
- Demo web (presentation.html)

## Diagrama (ASCII)
[Browser]──HTTP(8009)──> [presentation.html]
    │                           │ fetch /chat, /memory
    └────────────────────HTTP(8010)──> [FastAPI] ──> [llama.cpp/OpenChat]
                                     └── RAM store (memoria “lite”)

## Flujo /chat
1) UI junta system + (memoria opcional) + user.
2) FastAPI → prompt ChatML → llama.cpp.
3) Stops: `<|end_of_turn|>`, `</s>` → respuesta recortada.
