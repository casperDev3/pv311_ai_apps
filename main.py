import torch
from diffusers import StableDiffusionPipeline
import os

MODEL_CACHE_DIR = "./models/stable-diffusion"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

def download_model():
    print("=" * 60)
    print("Downloading model...")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
        )

        print("Model downloaded successfully!")
        print(f"Save to {MODEL_ID}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False



if __name__ == "__main__":
    download_model()