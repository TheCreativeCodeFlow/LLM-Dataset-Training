# DSA Tutor Deployment Guide

This guide covers deployment parameters, hardware specifications, and starting commands for local execution.

---

## 1. Hardware Specifications

### CPU Mode (Development / Verification)
* **RAM**: 8 GB minimum (16 GB recommended).
* **Disk Space**: 5 GB available space.
* **Compatibility**: Utilizes `eager` PyTorch attention implementations.

### GPU Mode (Production Execution)
* **VRAM**: 24 GB minimum (NVIDIA RTX 3090, A10G, or equivalent).
* **CUDA Version**: 11.8+ or 12.1+ recommended.
* **Optimization**: Uses `flash_attention_2` to reduce memory footprints for long conversation bounds.

---

## 2. Local Startup Instructions

### Step 1: Initialize Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 2: Set Python Path & Run Server
Run the FastAPI web application on port 8000:
```powershell
$env:PYTHONPATH="C:\Users\Web-wizrd\Desktop\Github\LLM-Dataset-Training"
python scripts/serve.py
```

### Step 3: Test Health Endpoint
```powershell
curl http://127.0.0.1:8000/health
```

---

## 3. Production Deployment Checklist
1. **Model Weight Verification**: Ensure base model path `configs/train_config.yaml` is pointing to the unquantized or 4-bit config of choice.
2. **Adapter Archive check**: Verify that `models/adapters/dsa_tutor_v1` directory exists and contains `adapter_config.json` and `adapter_model.safetensors`.
3. **API Port Binding**: Bind uvicorn to `0.0.0.0` or local port configurations behind an Nginx reverse proxy.
