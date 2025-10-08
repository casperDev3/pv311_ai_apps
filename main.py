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
        # Download without loading to GPU immediately
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
            low_cpu_mem_usage=True,  # Use this instead of offload_state_dict
            variant="fp16",  # Explicitly request fp16 variant
            # offload_state_dict=
        )

        # Save the pipeline to the cache directory
        save_path = os.path.join(MODEL_CACHE_DIR, "saved_model")
        pipe.save_pretrained(save_path)

        print("Model downloaded successfully!")
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")

    if os.path.exists(MODEL_CACHE_DIR):
        print("Loading model from cache...")
        local = True
    else:
        print("Model not found in cache")
        local = False

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR if local else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        local_files_only=local,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe

def generate_image(pipe, prompt, out_paths="gen_images/gen_img.png", steps=30, guidance=7.5):
    print("Generation....")

    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
    image.save(out_paths)

    print("Image generated successfully!")
    return image

def main():
    prompt = input("Enter prompt: ")
    try:
        pipe = load_model()
        generate_image(pipe, prompt)
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    # download_model()
    main()