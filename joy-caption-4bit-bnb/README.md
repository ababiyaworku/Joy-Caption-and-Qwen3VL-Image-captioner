# Batch Image Captioning with JoyCaption

A Python script for batch captioning images using the JoyCaption model with Meta-Llama-3.1-8B and SigLIP vision encoder. Supports recursive folder processing and multiple caption styles.

## Features

- **Batch Processing**: Process multiple images efficiently with configurable batch sizes
- **Recursive Folder Support**: Automatically caption images in subdirectories
- **Low VRAM Mode**: 4-bit quantization support for systems with limited GPU memory
- **Multiple Caption Styles**: Descriptive, Training Prompt, MidJourney, Booru tags, and more
- **Skip Existing**: Automatically skip images that already have captions
- **Flexible Configuration**: Customize caption length, style, and processing parameters

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- Hugging Face account (optional but recommended for model downloads)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set your Hugging Face token:
```bash
export HF_TOKEN=your_token_here
```
4, Download the adapter
[https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6](joy-caption)
## Usage

### Basic Usage

```bash
python batch.py --img_dir ./input
```

### Command Line Arguments

- `--img_dir`: Directory containing images to caption (default: `./input`)
- `--recursive`: Process images in subdirectories recursively (enabled by default)
- `--no-recursive`: Only process images in the main directory

### Examples

```bash
# Process all images in input folder and subfolders
python batch.py --img_dir ./my_images

# Process only top-level folder (no subfolders)
python batch.py --img_dir ./my_images --no-recursive
```

## Configuration

Edit the configuration section in `batch.py` to customize:

- `LOW_VRAM_MODE`: Enable 4-bit quantization (default: `True`)
- `BATCH_PROCESSING_COUNT`: Number of images per batch (default: `16`)
- `CAPTION_TYPE`: Style of captions (default: `"Descriptive"`)
- `CAPTION_LENGTH`: Length of captions - `any`, `short`, `medium`, or `long` (default: `"long"`)
- `SKIP_EXISTING`: Skip images with existing captions (default: `True`)
- `RECURSIVE`: Process subdirectories (default: `True`)

### Available Caption Types

- Descriptive
- Descriptive (Informal)
- Training Prompt
- MidJourney
- Booru tag list
- Booru-like tag list
- Art Critic
- Product Listing
- Social Media Post

## Supported Image Formats

- JPG/JPEG
- PNG
- WebP
- BMP
- GIF
- HEIC/HEIF

## Output

Captions are saved as `.txt` files in the same directory as their corresponding images, with the same filename.

Example:
- Image: `input/subfolder/photo.jpg`
- Caption: `input/subfolder/photo.txt`

## Performance

- **Low VRAM Mode (4-bit)**: ~6-8GB VRAM recommended
- **Full Precision**: ~16GB+ VRAM recommended
- Processing speed varies based on GPU, batch size, and image resolution

## License

This project uses the JoyCaption model and associated components. Please refer to the original model licenses for usage terms.

## Acknowledgments

- [JoyCaption Pre-Alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)
- [Meta Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Unsloth](https://huggingface.co/unsloth) for 4-bit quantized models

## Troubleshooting

**Out of Memory Errors**: Reduce `BATCH_PROCESSING_COUNT` or enable `LOW_VRAM_MODE`

**Missing Dependencies**: Ensure all packages in `requirements.txt` are installed

**Model Download Issues**: Set your `HF_TOKEN` environment variable or check your internet connection
