from llama_cpp import Llama

MODEL_PATH = "models/openchat-3.5-1210.Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

def generate_response(memoria):
    history = ""
    for entrada in memoria:
        rol = entrada["rol"]
        contenido = entrada["contenido"]
        if rol == "usuario":
            history += f"Usuario: {contenido}\n"
        else:
            history += f"AGI: {contenido}\n"
    prompt = f"{history}AGI:"

    output = llm(prompt, max_tokens=150, stop=["Usuario:", "AGI:"])
    texto = output["choices"][0]["text"].strip()
    return texto
