# Troubleshooting Guide for Qwen3-VL Batch Captioner

## The AttributeError: 'dict' object has no attribute 'to_dict' Error

This error occurs when using Unsloth's quantized models with certain versions of transformers. 

### Quick Fix

**Version 2 of the script** (the updated one) fixes this by:
1. Using `AutoModelForVision2Seq` instead of `Qwen2VLForConditionalGeneration`
2. Adding `trust_remote_code=True` parameter
3. Removing the dependency on `qwen-vl-utils`

### If You Still Get Errors

Try these steps:

1. **Update transformers:**
```bash
pip install --upgrade transformers>=4.37.0
```

2. **Update accelerate:**
```bash
pip install --upgrade accelerate>=0.20.0
```

3. **If using Python 3.13, downgrade to 3.11:**
Python 3.13 is very new and some packages may not be fully compatible yet.
```bash
# Use Python 3.11 instead
python3.11 -m pip install -r requirements_qwen3vl.txt
```

4. **Clear HuggingFace cache (if model downloaded incorrectly):**
```bash
# Windows
rmdir /s %USERPROFILE%\.cache\huggingface\hub\models--unsloth--Qwen3-VL-2B-Instruct-unsloth-bnb-4bit

# Linux/Mac
rm -rf ~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
```

Then re-run the script to re-download.

## Common Issues

### "You are using a model of type qwen3_vl to instantiate a model of type qwen2_vl"

This is just a warning, not an error. It happens because Qwen3-VL is based on Qwen2-VL architecture. You can ignore it.

### "CUDA out of memory"

Solutions:
- Reduce `BATCH_PROCESSING_COUNT` from 8 to 4 or 2
- Close other GPU applications
- Make sure `LOW_VRAM_MODE = True`

### Images not being processed

Check:
- Images are in supported formats (.jpg, .jpeg, .png, .webp, .heic)
- Input directory path is correct
- If using `SKIP_EXISTING = True`, check if .txt files already exist

### Slow generation

This is normal for the first batch as the model loads into VRAM. Subsequent batches should be faster.

### Windows Symlink Warning

This warning is harmless. To fix it (optional):
1. Enable Windows Developer Mode
2. OR run as administrator
3. OR set environment variable: `set HF_HUB_DISABLE_SYMLINKS_WARNING=1`

## Model Alternatives

If the Unsloth 4-bit model continues to have issues, try:

```python
# In the script, change:
LOW_VRAM_MODE = False
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
```

This uses the official non-quantized model (requires ~8-12GB VRAM).

## Still Having Issues?

Check:
1. GPU is available: `torch.cuda.is_available()` should return `True`
2. CUDA version matches PyTorch version
3. Sufficient disk space (~3GB for model download)
4. HuggingFace token is valid (if using gated models)

## Performance Tips

- **Batch size**: Start with 4, increase if you have VRAM to spare
- **Temperature**: Lower (0.3-0.5) for more consistent captions, higher (0.7-0.9) for more creative ones
- **Max tokens**: 300 is good for long descriptions, use 100-150 for shorter captions
- **Recursive mode**: Turn off if you only want to process the main folder

## Getting Help

If you're still stuck:
1. Check the error message carefully
2. Try the suggestions above
3. Check if your GPU has enough VRAM (run `nvidia-smi`)
4. Verify transformers version: `pip show transformers`
