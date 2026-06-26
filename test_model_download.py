import time
from huggingface_hub import hf_hub_download

print("Starting download test...")
try:
    start_time = time.time()
    path = hf_hub_download(
        repo_id="unsloth/Llama-3.2-1B-Instruct",
        filename="model.safetensors",
        local_files_only=False
    )
    print(f"Download completed successfully in {time.time() - start_time:.2f} seconds!")
    print(f"Path: {path}")
except Exception as e:
    print(f"Error occurred: {e}")
