import os
import torch
from pathlib import Path
from PIL import Image
import pillow_heif
from tqdm import tqdm
import argparse
import requests
from transformers import (
    AutoProcessor, 
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig
)

print("Captioning Batch Images Initializing...")

# ============================
# CONFIGURATION SETTINGS
# ============================
LOW_VRAM_MODE = True  # Set to True to use 4-bit quantized model
BATCH_PROCESSING_COUNT = 16  # With 4-bit model, you can use higher batch sizes
CHECKPOINT_PATH = Path("wpkklhc6")
CAPTION_TYPE = "Descriptive"  # Can be: Descriptive, Descriptive (Informal), Training Prompt, MidJourney, Booru tag list, Booru-like tag list, Art Critic, Product Listing, Social Media Post
CAPTION_LENGTH = "long"  # Can be: any, short, medium, long
VLM_PROMPT = "A descriptive caption for this image:\n"
TEMPERATURE = 0.5
TOP_K = 10
MAX_NEW_TOKENS = 300
PREPEND_STRING = ""
APPEND_STRING = ""
SKIP_EXISTING = True
CAPTION_EXT = ".txt"
RECURSIVE = True  # Changed to True to enable sub-folder detection by default

# Hugging Face token (optional, but recommended)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ============================
# MODEL CONFIGURATION
# ============================
if LOW_VRAM_MODE:
    # Use Unsloth's 4-bit quantized model
    MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    CLIP_MODEL = "google/siglip-so400m-patch14-384"
    print("LOW_VRAM_MODE enabled - using 4-bit quantized model")
else:
    # Use full precision model
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
    CLIP_MODEL = "google/siglip-so400m-patch14-384"
    print("Using full precision model")

# ============================
# IMAGE ADAPTER DOWNLOAD
# ============================
def download_image_adapter(force_download=False):
    file_path = CHECKPOINT_PATH / "image_adapter.pt"
    if force_download or not file_path.exists():
        print(f"Downloading {file_path.name} from Hugging Face Space...")
        url = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"
        response = requests.get(url)
        if response.status_code == 200:
            CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_path.name} successfully.")
        else:
            print(f"Failed to download {file_path.name}. Status code: {response.status_code}")
            raise Exception("Failed to download image adapter")
    else:
        print(f"{file_path.name} already exists.")
    return file_path

# ============================
# IMAGE ADAPTER CLASS
# ============================
class ImageAdapter(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract
        
        if self.deep_extract:
            input_features = input_features * 5
        
        self.linear1 = torch.nn.Linear(input_features, output_features)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(output_features, output_features)
        self.ln1 = torch.nn.LayerNorm(input_features) if ln1 else torch.nn.Identity()
        self.pos_emb = torch.nn.Parameter(torch.zeros(num_image_tokens, input_features)) if pos_emb else None
    
    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.cat([
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ], dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]
        
        x = self.ln1(x)
        
        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        return x

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
        map_idx = 2
        length_str = length
    elif isinstance(length, int):
        map_idx = 1
        length_str = None
    else:
        map_idx = 0
        length_str = None
    
    prompt_str = CAPTION_TYPE_MAP.get(CAPTION_TYPE, CAPTION_TYPE_MAP["Descriptive"])[map_idx]
    
    if isinstance(length, int):
        prompt_str = prompt_str.format(word_count=length)
    elif isinstance(length, str):
        prompt_str = prompt_str.format(length=length_str)
    
    return prompt_str

def load_image(image_path):
    """Load and convert image to RGB"""
    try:
        # Register HEIF opener
        pillow_heif.register_heif_opener()
        
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_image_files(directory, recursive=False):
    """Get all image files from directory, optionally searching recursively"""
    supported_extensions = {'.jpg', '.jpeg', '.jfif', '.png', '.webp', '.bmp', '.gif', '.heic', '.heif'}
    image_files = []
    
    if recursive:
        # Search recursively through all subdirectories
        print(f"Searching recursively for images in {directory} and subdirectories...")
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_extensions:
                    image_files.append(file_path)
        print(f"Found {len(image_files)} images across all subdirectories")
    else:
        # Only search in the specified directory (not subdirectories)
        print(f"Searching for images in {directory} (non-recursive)...")
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() in supported_extensions:
                image_files.append(file)
        print(f"Found {len(image_files)} images in directory")
    
    return sorted(image_files)

# ============================
# IMAGE PROCESSING
# ============================
def process_images(input_dir, clip_processor, clip_model, text_model, image_adapter, tokenizer):
    """Process all images in the input directory"""
    # Get all image files based on RECURSIVE setting
    image_files = get_image_files(input_dir, recursive=RECURSIVE)
    
    if not image_files:
        print("No images found to process!")
        return
    
    # Filter out images that already have captions if SKIP_EXISTING is True
    if SKIP_EXISTING:
        files_to_process = []
        for img_path in image_files:
            caption_path = img_path.parent / (img_path.stem + CAPTION_EXT)
            if not caption_path.exists():
                files_to_process.append(img_path)
            else:
                print(f"Skipping {img_path.name} (caption exists)")
        image_files = files_to_process
    
    if not image_files:
        print("All images already have captions!")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # Get the caption prompt
    caption_prompt = get_caption_prompt()
    print(f"Using caption prompt: {caption_prompt}")
    
    # Process images in batches
    for i in range(0, len(image_files), BATCH_PROCESSING_COUNT):
        batch_files = image_files[i:i + BATCH_PROCESSING_COUNT]
        print(f"\n{'='*50}")
        print(f"Processing batch {i//BATCH_PROCESSING_COUNT + 1}/{(len(image_files)-1)//BATCH_PROCESSING_COUNT + 1}")
        print(f"Images {i+1}-{min(i+BATCH_PROCESSING_COUNT, len(image_files))} of {len(image_files)}")
        print(f"{'='*50}")
        
        # Load images for this batch
        batch_images = []
        valid_files = []
        
        for img_path in batch_files:
            # Show relative path from input directory for better readability
            try:
                rel_path = img_path.relative_to(input_dir)
                print(f"Loading: {rel_path}")
            except ValueError:
                print(f"Loading: {img_path}")
            
            image = load_image(img_path)
            if image is not None:
                batch_images.append(image)
                valid_files.append(img_path)
        
        if not batch_images:
            print("No valid images in this batch, skipping...")
            continue
        
        try:
            # Process images through CLIP
            print(f"Processing {len(batch_images)} images through CLIP...")
            image_inputs = clip_processor(images=batch_images, return_tensors='pt', padding=True).to('cuda')
            
            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                vision_outputs = clip_model(**image_inputs, output_hidden_states=True)
                image_features = vision_outputs.hidden_states
                embedded_images = image_adapter(image_features)
            
            # Create embeddings for each image in the batch
            print("Creating input embeddings...")
            input_embeds_list = []
            
            for embedded_image in embedded_images:
                # Create the conversation
                convo = [
                    {
                        "role": "system",
                        "content": "You are a helpful image captioner.",
                    },
                    {
                        "role": "user",
                        "content": caption_prompt,
                    },
                ]
                
                # Apply chat template
                convo_string = tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Check if we need to add BOS token
                if not convo_string.startswith(tokenizer.bos_token):
                    convo_string = tokenizer.bos_token + convo_string
                
                # Split the conversation at the image insertion point (end of user message)
                prompt_split = convo_string.rsplit("<|eot_id|>", 1)
                if len(prompt_split) == 2:
                    preamble = prompt_split[0] + "<|eot_id|>"
                    postamble = prompt_split[1]
                else:
                    preamble = convo_string
                    postamble = ""
                
                # Tokenize preamble and postamble
                preamble_ids = tokenizer.encode(preamble, add_special_tokens=False, return_tensors='pt').to('cuda')
                postamble_ids = tokenizer.encode(postamble, add_special_tokens=False, return_tensors='pt').to('cuda')
                
                # Get embeddings
                preamble_embeds = text_model.get_input_embeddings()(preamble_ids)
                postamble_embeds = text_model.get_input_embeddings()(postamble_ids)
                
                # Ensure embedded_image has the right shape (add batch dimension if needed)
                if embedded_image.dim() == 2:
                    embedded_image = embedded_image.unsqueeze(0)
                
                # Concatenate everything
                input_embeds = torch.cat([
                    preamble_embeds,
                    embedded_image,
                    postamble_embeds,
                ], dim=1)
                
                input_embeds_list.append(input_embeds)
            
            # Stack all inputs
            input_embeds_batch = torch.cat(input_embeds_list, dim=0)
            
            # Create attention mask (all ones since we have actual embeddings)
            attention_mask = torch.ones(
                input_embeds_batch.shape[0], 
                input_embeds_batch.shape[1], 
                dtype=torch.long, 
                device='cuda'
            )
            
            # Store the prompt length for later
            prompt_length = input_embeds_batch.shape[1]
            
            # Generate - only pass inputs_embeds
            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                generate_ids = text_model.generate(
                    inputs_embeds=input_embeds_batch,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    use_cache=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode and save captions
            for idx, (img_path, gen_ids) in enumerate(zip(valid_files, generate_ids)):
                try:
                    # Show relative path for better readability
                    try:
                        rel_path = img_path.relative_to(input_dir)
                        print(f"\n--- Processing {rel_path} ---")
                    except ValueError:
                        print(f"\n--- Processing {img_path.name} ---")
                    
                    print(f"Generated sequence length: {len(gen_ids)}")
                    print(f"Prompt length: {prompt_length}")
                    print(f"First 10 tokens: {gen_ids[:10].tolist()}")
                    print(f"Last 10 tokens: {gen_ids[-10:].tolist()}")
                    
                    # Decode the FULL sequence first to see what we have
                    full_decode = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    print(f"Full decode (first 200 chars): {full_decode[:200]}")
                    
                    # The generated IDs include the prompt, so we need to remove it
                    # Since we used inputs_embeds, the output is just the generated tokens
                    caption = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    caption = caption.strip()
                    
                    # If caption is still empty, try without skipping special tokens
                    if not caption:
                        caption = tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        # Remove special tokens manually
                        for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "system", "user", "assistant"]:
                            caption = caption.replace(token, "")
                        caption = caption.strip()
                    
                    caption = f"{PREPEND_STRING}{caption}{APPEND_STRING}"
                    
                    # Save caption in the same directory as the image
                    caption_path = img_path.parent / (img_path.stem + CAPTION_EXT)
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    
                    print(f"✓ Saved caption to: {caption_path.relative_to(input_dir) if caption_path.is_relative_to(input_dir) else caption_path}")
                    print(f"  Caption: {caption[:150]}..." if len(caption) > 150 else f"  Caption: {caption}")
                except Exception as e:
                    print(f"\n✗ Error saving caption for {img_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear batch from memory
            del batch_images, image_inputs, vision_outputs, image_features, embedded_images
            del input_embeds_batch, generate_ids
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
    parser = argparse.ArgumentParser(description='Batch caption images using JoyCaption')
    parser.add_argument('--img_dir', type=str, default='./input', help='Directory containing images to caption')
    parser.add_argument('--recursive', action='store_true', help='Process images in subdirectories recursively')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help='Only process images in the main directory')
    parser.set_defaults(recursive=RECURSIVE)  # Use the RECURSIVE constant as default
    args = parser.parse_args()
    
    input_directory = Path(args.img_dir)
    
    # Override RECURSIVE setting with command line argument if provided
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
    
    # Download image adapter
    adapter_path = download_image_adapter()
    
    # Load CLIP model
    print(f"\nLoading CLIP model: {CLIP_MODEL}")
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL, token=HF_TOKEN)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL, token=HF_TOKEN).vision_model.eval().requires_grad_(False).to('cuda')
    
    # Load image adapter
    print("Loading image adapter...")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size,  # 1152 for SigLIP
        4096,  # Output features to match LLaMA embedding dimension
        False,  # ln1
        False,  # pos_emb
        38,     # num_image_tokens
        False   # deep_extract - must be False to match checkpoint
    )
    image_adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
    image_adapter.eval().to('cuda')
    
    # Load text model with 4-bit quantization
    print(f"\nLoading text model: {MODEL_NAME}")
    
    if LOW_VRAM_MODE:
        # Unsloth models already have quantization built-in, just load directly
        text_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=True,
        )
    else:
        text_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN,
        )
    
    text_model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    # Ensure tokenizer has required attributes
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set Llama 3.1 chat template if not present
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    
    print("\nModels loaded successfully!")
    print("="*50)
    
    # Process images
    process_images(input_directory, clip_processor, clip_model, text_model, image_adapter, tokenizer)
    
    print("\n" + "="*50)
    print("Processing complete!")