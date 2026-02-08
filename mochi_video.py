import os
# Set cache to D drive to avoid filling C drive
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache/transformers"

# ⚡⚡⚡ CONSUME ENTIRE INTERNET SPEED - MAXIMUM AGGRESSIVE SETTINGS ⚡⚡⚡
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use fastest hf_transfer (Rust-based)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "900"  # Extra long timeout for large files
os.environ["HTTPX_TIMEOUT"] = "900"  # HTTP timeout
os.environ["REQUESTS_TIMEOUT"] = "900"  # Requests timeout
os.environ["HF_TRANSFER_CONCURRENCY"] = "32"  # Max concurrent chunk downloads per file
os.environ["HF_TRANSFER_CHUNK_SIZE"] = "10485760"  # 10MB chunks for better speed

import torch
import gradio as gr
from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
from diffusers.utils.export_utils import export_to_video
from huggingface_hub import snapshot_download
from pathlib import Path

# Check GPU availability and setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Set memory allocator for better GPU performance
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# Check existing cache to avoid re-downloading
cache_path = Path("D:/huggingface_cache/hub/models--genmo--mochi-1-preview")

if cache_path.exists():
    cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024**3)
    print(f"📦 Found existing cache: {cache_size:.2f} GB")
    
    # Check if cache is complete (should be ~133 GB for full model)
    if cache_size < 100:
        print(f"⚠️  Incomplete cache ({cache_size:.2f} GB / ~133 GB needed)")
        print("📥 Resuming download to get missing files...")
    else:
        print("✅ Cache appears complete!")
else:
    print("📥 Starting fresh download")
    cache_size = 0

# ⚡⚡⚡ MAXIMUM SPEED DOWNLOAD - 32 PARALLEL CONNECTIONS + 32 CHUNKS PER FILE ⚡⚡⚡
# Always run snapshot_download with smart resume - it will skip complete files
print("⚡⚡⚡ CONSUMING MAXIMUM INTERNET SPEED (32 parallel files + 32 chunks each = 1024 total connections!)...")
print("💨 Smart Resume: Only downloading new/incomplete files!")
print("🔥 Using hf_transfer with 10MB chunks for maximum throughput!")
snapshot_download(
    "genmo/mochi-1-preview",
    cache_dir="D:/huggingface_cache",
    max_workers=32,  # 32 parallel file downloads
    resume_download=True,
    force_download=False,
    local_files_only=False,
)
print("✅ Download complete! All files verified.")

# Load pipeline - use device_map="auto" for automatic memory management
print("📦 Loading pipeline (using sequential CPU offload for low VRAM)...")
pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.bfloat16,
    variant="bf16",
    cache_dir="D:/huggingface_cache",
    low_cpu_mem_usage=True,
    device_map="balanced",  # Automatically balance across devices
)

# Enable memory optimizations for 8GB VRAM
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing(1)  # Process attention in chunks
print("✅ Pipeline loaded with CPU offload (optimized for 8GB VRAM)")

def gen_video(p):
    with torch.inference_mode():  # Optimize inference
        out = pipe(p, num_inference_steps=28, guidance_scale=3.5).frames[0]
    return export_to_video(out, fps=30)

# Uncomment below to run standalone generation:
# prompt = "smooth realistic horse galloping in Tamil village, natural physics, 30fps cinematic"
# out = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).frames[0]
# export_to_video(out, "tamil_horse.mp4", fps=30)
# print("Video ready: tamil_horse.mp4")

# Launch Gradio UI
gr.Interface(gen_video, "text", "video").launch()
