# 🎬 Mochi Video Generator

Generate AI videos using Genmo's Mochi model via Hugging Face.

## Requirements

- **Local**: NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090, A5000, etc.)
- **Cloud**: Google Colab with A100 GPU (recommended)

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DYNAMOatGHUB/Video-generator-using-Mochi-via-huggingface/blob/main/mochi_colab.ipynb)

1. Click the badge above or upload `mochi_colab.ipynb` to Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → A100`
3. Run all cells

## Files

| File | Description |
|------|-------------|
| `mochi_video.py` | Local script with Gradio UI |
| `mochi_colab.ipynb` | Google Colab notebook |
| `cleanup_cache.py` | Utility to clean incomplete downloads |

## Model Info

- **Model**: [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- **Size**: ~46GB (bf16 variant)
- **VRAM**: 24GB minimum, 40GB+ recommended

## Usage

```python
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

frames = pipe("A horse galloping through a village", num_inference_steps=28).frames[0]
export_to_video(frames, "output.mp4", fps=30)
```

## License

MIT
