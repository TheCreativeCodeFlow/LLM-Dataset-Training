# Known Issues & Cleanup Recommendations

This document outlines detected bugs, inconsistencies, duplicate code, dependency gaps, architectural risks, and recommended actions for cleanup.

## 1. Dead Code and Legacy Scripts

- **`scripts/clean.py`**: Legacy, duplicate script of `scripts/clean_dataset.py`. It hardcodes dataset paths and lacks modular cleaning validations.
- **`scripts/transform.py`**: Legacy, duplicate script of `scripts/transform_dataset.py`. It lacks key compatibility mapping fields.
- **`scripts/download.py`**: Older, legacy dataset downloader superseded by the production-ready `scripts/fetch_dataset.py`.
- **`scripts/transform.py`**: Duplicate functionality.

**Recommendation:** Remove `scripts/clean.py`, `scripts/transform.py`, and `scripts/download.py` to keep the codebase clean.

---

## 2. Dependency Inconsistencies

- **`requirements.txt` gaps:** The codebase imports `matplotlib` (for failure charts), `pyyaml` (for configs parsing), and `numpy` (for validation metrics), but none of these are declared in `requirements.txt`.

**Recommendation:** Append the following packages to `requirements.txt`:
```text
pyyaml
matplotlib
numpy
```

---

## 3. Architectural Risks

- **Memory/VRAM constraints:** Phi-3-mini fine-tuning requires ~7.15 GB of VRAM. Smaller hardware configurations will hit CUDA Out-Of-Memory limits if gradient checkpointing or 4-bit quantization config is modified or disabled.
- **Dynamic Adapter Loading in serving layer (`serve.py`):** If multiple adapters are run concurrently without adequate GPU allocation, server request handling latencies will spike.

---

## 4. Recommended Cleanup Actions

1. Delete unused/duplicate files.
2. Run standard formatting tools (`black`, `isort`) on the scripts directory.
3. Update `requirements.txt` to include missing dependencies.
