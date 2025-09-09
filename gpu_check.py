# gpu_check.py

import torch
import subprocess
import sys

def check_torch_cuda():
    print("🔎 Verificando PyTorch y CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible | GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("❌ CUDA no disponible en PyTorch.")

def check_llama_cpp():
    print("\n🔎 Verificando soporte de CUDA/cuBLAS en llama-cpp-python...")
    try:
        result = subprocess.run(
            ["pip", "show", "llama-cpp-python"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "Location" in result.stdout:
            print("✅ llama-cpp-python instalado.")
        else:
            print("❌ llama-cpp-python no instalado.")
    except Exception as e:
        print(f"⚠️ Error detectando llama-cpp-python: {e}")

if __name__ == "__main__":
    print("=== GPU Check ===")
    check_torch_cuda()
    check_llama_cpp()
