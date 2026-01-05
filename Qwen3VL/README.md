# Qwen3-VL Batch Image Captioner

A streamlined Python script for batch captioning images using Qwen3-VL, a state-of-the-art vision-language model with native image understanding capabilities.

## Features

- 🚀 **Native Vision Processing** - No separate CLIP model needed
- 💾 **Low VRAM Mode** - 4-bit quantization for ~4-6GB VRAM usage
- 📦 **Batch Processing** - Process multiple images efficiently
- 🎨 **Multiple Caption Styles** - Descriptive, artistic, social media, and more
- 📁 **Recursive Directory Support** - Process entire folder structures
- ⚡ **Smart Skipping** - Automatically skip already-captioned images

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10-3.12 (3.13 may have compatibility issues)
- CUDA-capable GPU (NVIDIA)
- 6GB+ VRAM (4GB minimum with low VRAM mode)

## Quick Start

```bash
# Basic usage
python Qwen3VL_BatchCaptioner.py --img_dir ./your_images

# Process subdirectories
python Qwen3VL_BatchCaptioner.py --img_dir ./your_images --recursive

# Single directory only
python Qwen3VL_BatchCaptioner.py --img_dir ./your_images --no-recursive
```

## Configuration

Edit settings at the top of `Qwen3VL_BatchCaptioner.py`:

```python
LOW_VRAM_MODE = True              # Use 4-bit quantization
BATCH_PROCESSING_COUNT = 8        # Images per batch
CAPTION_TYPE = "Descriptive"      # Caption style
CAPTION_LENGTH = "long"           # any, short, medium, long
TEMPERATURE = 0.5                 # Creativity (0.1-1.0)
SKIP_EXISTING = True              # Skip existing captions
```

### Caption Types

| Type | Description |
|------|-------------|
| `Descriptive` | Formal, detailed descriptions |
| `Descriptive (Informal)` | Casual, conversational tone |
| `Training Prompt` | Stable Diffusion style prompts |
| `MidJourney` | MidJourney-optimized prompts |
| `Booru tag list` | Tag-based descriptions |
| `Art Critic` | Artistic analysis with composition details |
| `Product Listing` | E-commerce product descriptions |
| `Social Media Post` | Social media-ready captions |

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- HEIC (.heic)

## Output

Captions are saved as `.txt` files alongside images:
```
input/
├── photo1.jpg
├── photo1.txt      ← Generated caption
├── sunset.png
└── sunset.txt      ← Generated caption
```

## Model Information

**Default (Low VRAM):**
- Model: `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit`
- VRAM: ~4-6GB
- Quantization: 4-bit

**Full Precision (Optional):**
- Model: `Qwen/Qwen2-VL-2B-Instruct`
- VRAM: ~8-12GB
- Set `LOW_VRAM_MODE = False`

## Performance

| Batch Size | 4-bit VRAM | Full Precision VRAM |
|------------|------------|---------------------|
| 1          | ~3GB       | ~6GB                |
| 4          | ~4GB       | ~8GB                |
| 8          | ~5GB       | ~10GB               |
| 16         | ~6-7GB     | ~14GB               |

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `BATCH_PROCESSING_COUNT`
- Ensure `LOW_VRAM_MODE = True`
- Close other GPU applications

**Import Errors:**
```bash
pip install --upgrade transformers accelerate
```

**For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

## Example

**Input:** `sunset.jpg`

**Output (`sunset.txt`):**
```
A breathtaking sunset over a calm ocean with vibrant orange and pink hues 
reflecting on the water's surface. The sky transitions from deep purple at 
the top to warm golden tones near the horizon, while silhouetted clouds add 
depth and drama to the scene.
```

## License

This project uses the Qwen3-VL model, which has its own license terms. Please review the [Qwen model license](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) before commercial use.

## Credits

- **Qwen3-VL**: Alibaba Cloud - [Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **Quantization**: Unsloth - [Optimized Models](https://huggingface.co/unsloth)
- **Inspiration**: JoyCaption project for batch processing approach

## Citation

If you use this tool in your research, please cite the Qwen3-VL model:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

---

**Made with ❤️ for the AI community**
