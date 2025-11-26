# CMPE-258-Image-Caption-Generator

## Docker Usage

The project now ships with a CUDA-enabled Docker image (Python 3.11 + cu121 PyTorch wheels) so you get GPU acceleration without changing your host environment. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you plan to run with `--gpus all`.

1. **Build the image**
   ```bash
   docker build -t image-caption .
   ```

2. **Run commands inside the container**
   Mount any folders you want to persist (e.g., `raw_data`, `data`, `checkpoints`) and start an interactive shell:
   ```bash
   docker run --rm -it --gpus all \
     -v "$(pwd)/raw_data:/app/raw_data" \
     -v "$(pwd)/data:/app/data" \
     -v "$(pwd)/checkpoints:/app/checkpoints" \
     image-caption
   ```

3. **Execute the usual workflow inside the container**
   ```bash
   python preprocessing.py
   python -m model_baseline_only.train
   python evaluate.py
   ```

The container already includes required NLTK downloads; if you need additional packages run `python -m nltk.downloader <package>` inside the container.

## Baseline-Only Training

To train just the BaselineCaptionModel (without modifying `models/`), run:

```bash
python -m model_baseline_only.train
```

This script uses the dataset/training utilities that live alongside the simplified model inside `model_baseline_only/`.
