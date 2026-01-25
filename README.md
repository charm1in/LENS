# GLEM: Global-Local Evidence Mining for Training-Free Image Detection

Official PyTorch implementation of the paper **"Global-Local Evidence Mining for Training-Free Latent-Diffusion Image Detection"**.

GLEM is a training-free framework for detecting AI-generated images. It leverages the spectral inconsistencies of the VAE used in Latent Diffusion Models (LDMs) via a two-stage pipeline: a global spectral gate for efficiency and a local evidence mining module for hard-sample detection.

## 🛠️ Installation

The code has been tested on Python 3.10, PyTorch 2.1.2, and CUDA 11.8/12.1.
We explicitly pin dependencies in `requirements.txt` to avoid compatibility issues with NumPy 2.0+.

```bash
pip install -r requirements.txt
```

## 📂 Data Preparation

The script `main.py` automatically scans the root directory for datasets. It identifies real/fake images based on directory names:



**Expected Directory Structure:**

```text
/path/to/benchmark_root/
├── GenImage/
│   ├── Midjourney/
│   │   ├── ai/          # <--- Contains fake images
│   │   └── nature/      # <--- Contains real images
│   └── ...
└── SynthBuster/
    ├── midjourney-v5/
    │   ├── 0_real/
    │   └── 1_fake/
    └── ...
```

## 🚀 Usage



Here is an example with custom arguments:
```bash
python main.py \
  --data_root /path/to/SynthBuster \
  --output_dir ./results_repro \
  --repo_id "CompVis/stable-diffusion-v1-4" \
  --gate_threshold 0.801 \
  --hf_cutoff 0.3 \
  --scales 0.6 0.5 0.4 \
  --top_k 3 \
  --stride 16 \
  --softmin_temp 0.05
```