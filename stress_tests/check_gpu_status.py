import torch
import json

def check_gpu_status():
    print("🔎 Verificando solo soporte CUDA (FAISS usará CPU)...\n")

    report = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    print("✅ Estado de CUDA/PyTorch:")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    check_gpu_status()
