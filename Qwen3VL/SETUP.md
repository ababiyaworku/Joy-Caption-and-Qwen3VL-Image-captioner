# Quick Setup Guide

## Prerequisites

1. **Python 3.10-3.12** (NOT 3.13 - compatibility issues)
2. **NVIDIA GPU** with CUDA support
3. **6GB+ VRAM** (4GB minimum with 4-bit quantization)
4. **~3GB disk space** for model download

## Installation Steps

### 1. Clone or Download Repository

```bash
git clone <your-repo-url>
cd qwen3vl-batch-captioner
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have CUDA issues, install PyTorch separately first:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements.txt
```

### 4. Verify CUDA

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

### 5. Create Input Directory

```bash
mkdir input
# Copy your images to the input folder
```

### 6. Run the Script

```bash
python batch_qwen3vl.py --img_dir ./input
```

## First Run

The first time you run the script:
- Model will download (~2.4GB for 4-bit version)
- First batch takes longer (model loading)
- Subsequent batches are faster

**Model downloads to:**
- Windows: `C:\Users\<YourName>\.cache\huggingface\`
- Linux/Mac: `~/.cache/huggingface/`

## Configuration

Before running, you may want to adjust these settings in `batch_qwen3vl.py`:

```python
BATCH_PROCESSING_COUNT = 8    # Lower if out of memory
CAPTION_TYPE = "Descriptive"  # Change caption style
CAPTION_LENGTH = "long"       # short, medium, long, any
TEMPERATURE = 0.5             # 0.3 = consistent, 0.7 = creative
```

## Testing

Try with a single test image first:
```bash
# Put one image in ./input/test.jpg
python batch_qwen3vl.py --img_dir ./input --no-recursive
```

Check `./input/test.txt` for the generated caption.

## Common Issues

### "CUDA out of memory"
```python
BATCH_PROCESSING_COUNT = 4  # Or even 2
```

### "No module named 'PIL'"
```bash
pip install pillow pillow-heif
```

### Model download fails
- Check internet connection
- Check disk space (~3GB needed)
- Try with HuggingFace token (for gated models)

### Slow generation
- Normal for first batch (model loading)
- Check GPU usage: `nvidia-smi`
- Close other GPU applications

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Check your CUDA installation: `nvidia-smi`
3. Verify Python version: `python --version`
4. Check package versions: `pip list | grep -E "torch|transformers"`

## Next Steps

- Adjust caption style in config
- Try different temperature settings
- Process your image dataset
- Experiment with batch sizes

Happy captioning! 🎉
