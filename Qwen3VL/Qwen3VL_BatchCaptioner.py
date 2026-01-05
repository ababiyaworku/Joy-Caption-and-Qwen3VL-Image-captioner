import os
import torch
from pathlib import Path
from PIL import Image
import pillow_heif
from tqdm import tqdm
import argparse
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

print("Captioning Batch Images with Qwen3-VL Initializing...")

# ============================
# CONFIGURATION SETTINGS
# ============================
LOW_VRAM_MODE = True  # Set to True to use 4-bit quantized model
BATCH_PROCESSING_COUNT = 8  # Qwen3-VL processes images natively, adjust based on your VRAM
CAPTION_TYPE = "Descriptive"  # Can be: Descriptive, Descriptive (Informal), Training Prompt, MidJourney, Booru tag list, Booru-like tag list, Art Critic, Product Listing, Social Media Post
CAPTION_LENGTH = "long"  # Can be: any, short, medium, long
TEMPERATURE = 0.5
TOP_K = 10
MAX_NEW_TOKENS = 300
PREPEND_STRING = ""
APPEND_STRING = ""
SKIP_EXISTING = True
CAPTION_EXT = ".txt"
RECURSIVE = True

# Hugging Face token (optional, but recommended)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ============================
# MODEL CONFIGURATION
# ============================
if LOW_VRAM_MODE:
    MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    print("LOW_VRAM_MODE enabled - using 4-bit quantized Qwen3-VL model")
else:
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    print("Using full precision Qwen3-VL model")

# ============================
# CAPTION TYPE PROMPTS
# ============================
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

# ============================
# UTILITY FUNCTIONS
# ============================
def get_caption_prompt():
    length = None if CAPTION_LENGTH == "any" else CAPTION_LENGTH
    
    if isinstance(length, str):
        length_map = {
            "any": None,
            "very short": "a very short",
            "short": "a short",
            "medium-length": "a medium-length",
            "long": "a long",
            "very long": "a very long",
        }
        length = length_map.get(length, length)
    
    if CAPTION_TYPE not in CAPTION_TYPE_MAP:
        raise ValueError(f"Invalid caption type: {CAPTION_TYPE}")
    
    # Select appropriate prompt template based on length
    if length is None:
        prompt_str = CAPTION_TYPE_MAP[CAPTION_TYPE][0]
    elif isinstance(length, int):
        prompt_str = CAPTION_TYPE_MAP[CAPTION_TYPE][1].format(word_count=length)
    else:
        prompt_str = CAPTION_TYPE_MAP[CAPTION_TYPE][2].format(length=length)
    
    return prompt_str

def get_image_files(directory: Path, recursive: bool = False):
    """Get all image files from directory"""
    image_extensions = {'.jpg', '.jpeg', '.jfif', '.png', '.webp', '.bmp', '.gif', '.heic', '.heif'}
    
    if recursive:
        files = []
        for ext in image_extensions:
            files.extend(directory.rglob(f'*{ext}'))
            files.extend(directory.rglob(f'*{ext.upper()}'))
    else:
        files = []
        for ext in image_extensions:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)

def load_image(image_path: Path) -> Image.Image:
    """Load image and convert HEIC if necessary"""
    if image_path.suffix.lower() == '.heic':
        heif_file = pillow_heif.read_heif(str(image_path))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
    else:
        image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def process_images(input_dir: Path, processor, model):
    """Process all images in the directory"""
    image_files = get_image_files(input_dir, recursive=RECURSIVE)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Filter out images that already have captions if SKIP_EXISTING is True
    files_to_process = []
    for img_path in image_files:
        caption_path = img_path.parent / (img_path.stem + CAPTION_EXT)
        if SKIP_EXISTING and caption_path.exists():
            print(f"Skipping {img_path.name} (caption already exists)")
        else:
            files_to_process.append(img_path)
    
    if not files_to_process:
        print("No new images to process!")
        return
    
    print(f"Processing {len(files_to_process)} images...")
    
    # Get the caption prompt
    prompt_text = get_caption_prompt()
    print(f"\nUsing prompt: {prompt_text}\n")
    
    # Process in batches
    for i in tqdm(range(0, len(files_to_process), BATCH_PROCESSING_COUNT), desc="Processing batches"):
        batch_files = files_to_process[i:i + BATCH_PROCESSING_COUNT]
        
        try:
            # Prepare messages for batch processing
            batch_messages = []
            batch_images = []
            valid_files = []
            
            for img_path in batch_files:
                try:
                    # Load image
                    image = load_image(img_path)
                    batch_images.append(image)
                    
                    # Create message with image placeholder
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    
                    batch_messages.append(messages)
                    valid_files.append(img_path)
                    
                except Exception as e:
                    print(f"\nError loading {img_path.name}: {e}")
                    continue
            
            if not batch_messages:
                continue
            
            # Process batch
            # Prepare for inference
            texts = []
            for messages in batch_messages:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
            
            # Prepare inputs - pass images directly
            inputs = processor(
                text=texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Generate captions
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                )
            
            # Trim off input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode outputs
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            
            # Save captions
            for img_path, caption in zip(valid_files, output_texts):
                try:
                    caption = caption.strip()
                    caption = f"{PREPEND_STRING}{caption}{APPEND_STRING}"
                    
                    # Save caption in the same directory as the image
                    caption_path = img_path.parent / (img_path.stem + CAPTION_EXT)
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    
                    # Show relative path for better readability
                    try:
                        rel_path = img_path.relative_to(input_dir)
                        print(f"\n✓ Saved: {rel_path}")
                    except ValueError:
                        print(f"\n✓ Saved: {img_path.name}")
                    
                    print(f"  Caption: {caption[:150]}..." if len(caption) > 150 else f"  Caption: {caption}")
                    
                except Exception as e:
                    print(f"\n✗ Error saving caption for {img_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear memory
            del inputs, generated_ids, generated_ids_trimmed, output_texts, batch_images
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue

# ============================
# MAIN EXECUTION
# ============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch caption images using Qwen3-VL')
    parser.add_argument('--img_dir', type=str, default='./input', help='Directory containing images to caption')
    parser.add_argument('--recursive', action='store_true', help='Process images in subdirectories recursively')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help='Only process images in the main directory')
    parser.set_defaults(recursive=RECURSIVE)
    args = parser.parse_args()
    
    input_directory = Path(args.img_dir)
    RECURSIVE = args.recursive
    
    if not input_directory.exists():
        print(f"Error: Directory {input_directory} does not exist!")
        exit(1)
    
    print(f"Processing images from: {input_directory}")
    print(f"Batch size: {BATCH_PROCESSING_COUNT}")
    print(f"Caption type: {CAPTION_TYPE}")
    print(f"Caption length: {CAPTION_LENGTH}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Recursive mode: {RECURSIVE}")
    
    # Load Qwen3-VL model and processor
    print(f"\nLoading Qwen3-VL model: {MODEL_NAME}")
    
    # Use AutoModelForVision2Seq which handles both regular and quantized models
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    
    # The minimum required version is transformers 4.37.0
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
    )
    
    print("\nModel loaded successfully!")
    print("="*50)
    
    # Process images
    process_images(input_directory, processor, model)
    
    print("\n" + "="*50)
    print("Processing complete!")
