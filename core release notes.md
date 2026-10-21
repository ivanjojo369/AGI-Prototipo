# Quipu Core 4.0.0

Esta versión consolida el núcleo activo de Quipu sin reescribir Memory, Cognitive Control ni FactualVerifier.

## Ruta activa

`QuipuLoop -> Cognitive Control -> Scientific Engine / RAG / Orchestrator -> LLM cuando corresponde -> Verifier/FactualVerifier -> SelfCorrection -> OutputGuard -> Memory`

## Cambios de cierre

- Scientific Engine 1.1 mantiene autoridad determinista sobre Schwarzschild, álgebra simbólica, derivadas y conversiones de unidades.
- RAG vuelve a tener un contrato real `RagPipeline`; `QuipuLoop` ya no cae en el stub por ausencia de la clase.
- `root/settings.py` define explícitamente `RAG_STORE_JSON` separado de `SEMANTIC_STORE_JSON`.
- El indexador RAG en modo append actualiza por id y evita duplicar chunks.
- Planner tiene una sola autoridad (`planner.htn_simple`) y ya no llama implícitamente al endpoint legacy `:8010`.
- El planner sólo se invoca cuando Cognitive Control detecta una tarea de planificación.
- Las respuestas científicas deterministas no construyen innecesariamente contexto probabilístico de RAG/planner/memoria.
- El gateway HTTP usa RAG, planner, verifier y Cognitive Control igual que el núcleo CLI.
- Cada `conversation_id` HTTP obtiene estado conversacional efímero separado y contexto reciente limitado al chat seleccionado; la memoria consolidada sigue siendo global.
- Los estados de mundo por tema se aíslan también por conversación.
- UnifiedMemory deja de duplicar logs por propagación y corrige el fallback de `semantic_base_dir`.
- Embeddings intenta primero la caché local para evitar consultas repetidas a Hugging Face cuando el modelo ya está descargado.

## Compatibilidad / legacy

Permanecen en el repositorio módulos y tests históricos para no borrar trabajo sin necesidad. No forman parte de la ruta de producción de Core 4.0.0. Entre ellos pueden existir APIs antiguas de `root.quipu_loop`, `root.verifier`, `planner.task_planner`, `rag_service`, `tools.registry`, antiguos backends vectoriales y tests Phase 1. Por eso el gate oficial es `FINAL_RELEASE_CHECK.ps1`, no un `pytest tests` indiscriminado.

## Release gate

La reconstrucción usada para esta entrega superó 128 pruebas activas, además de compilación de los módulos modificados. La validación final debe repetirse en el entorno virtual real del proyecto con `FINAL_RELEASE_CHECK.ps1`.
