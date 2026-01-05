# Project Structure

```
qwen3vl-batch-captioner/
│
├── batch_qwen3vl.py          # Main script - run this!
├── requirements.txt           # All Python dependencies
│
├── README.md                  # Main documentation (start here)
├── SETUP.md                   # Installation guide
├── TROUBLESHOOTING.md         # Common issues & solutions
├── LICENSE                    # MIT License
├── .gitignore                # Git ignore patterns
│
├── input/                     # Create this - put your images here
│   ├── image1.jpg
│   ├── image1.txt            # Auto-generated captions
│   ├── image2.png
│   └── image2.txt
│
└── [.cache/]                 # Auto-created by HuggingFace
    └── huggingface/
        └── hub/
            └── models--unsloth--Qwen3-VL-2B-Instruct-unsloth-bnb-4bit/
                └── [model files ~2.4GB]
```

## File Descriptions

### Core Files

- **`batch_qwen3vl.py`** - Main Python script that processes images
  - Configure settings at the top of this file
  - Run with: `python batch_qwen3vl.py --img_dir ./input`

- **`requirements.txt`** - Python package dependencies
  - Install with: `pip install -r requirements.txt`

### Documentation

- **`README.md`** - Comprehensive documentation
  - Features overview
  - Quick start guide
  - Configuration options
  - Model information

- **`SETUP.md`** - Step-by-step installation guide
  - Prerequisites
  - Installation steps
  - First run instructions
  - Testing procedures

- **`TROUBLESHOOTING.md`** - Solutions to common problems
  - CUDA/memory issues
  - Import errors
  - Performance tips
  - Model alternatives

### Other Files

- **`LICENSE`** - MIT License for this project
- **`.gitignore`** - Files to exclude from git

## Usage Flow

1. **Setup**: Follow `SETUP.md`
2. **Configure**: Edit settings in `batch_qwen3vl.py`
3. **Run**: `python batch_qwen3vl.py --img_dir ./input`
4. **Troubleshoot**: Check `TROUBLESHOOTING.md` if needed

## Key Directories

### `input/` (Create this)
Put your images here. The script will:
- Read all supported image files
- Generate `.txt` files with captions
- Skip images that already have captions (if `SKIP_EXISTING = True`)

### `.cache/huggingface/` (Auto-created)
Model downloads here automatically on first run:
- Windows: `C:\Users\<YourName>\.cache\huggingface\`
- Linux/Mac: `~/.cache/huggingface/`
- Size: ~2.4GB for 4-bit model

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- WebP (`.webp`)
- HEIC (`.heic`)

## Output Format

For each image file, a corresponding `.txt` file is created:

```
image.jpg  →  image.txt
photo.png  →  photo.txt
```

The caption text file contains the generated description.

## Configuration

All configuration is done at the top of `batch_qwen3vl.py`:

```python
LOW_VRAM_MODE = True              # 4-bit quantization
BATCH_PROCESSING_COUNT = 8        # Images per batch
CAPTION_TYPE = "Descriptive"      # Caption style
CAPTION_LENGTH = "long"           # Caption length
TEMPERATURE = 0.5                 # Creativity level
SKIP_EXISTING = True              # Skip existing captions
RECURSIVE = True                  # Process subdirectories
```

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Basic run
python batch_qwen3vl.py --img_dir ./input

# With subdirectories
python batch_qwen3vl.py --img_dir ./input --recursive

# Single directory only
python batch_qwen3vl.py --img_dir ./input --no-recursive
```

## Git Workflow

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit"

# Add remote and push
git remote add origin <your-repo-url>
git push -u origin main
```

The `.gitignore` file ensures that:
- Input images and captions are not committed
- Model cache is not committed
- Virtual environment is not committed
- IDE files are not committed

## Next Steps

1. Read `README.md` for full documentation
2. Follow `SETUP.md` for installation
3. Configure settings in `batch_qwen3vl.py`
4. Create `input/` folder and add images
5. Run the script!
