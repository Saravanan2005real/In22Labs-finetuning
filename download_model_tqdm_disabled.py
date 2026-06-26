import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download

print("Starting model download with progress bars disabled...")
try:
    path = snapshot_download(repo_id="unsloth/Llama-3.2-1B-Instruct")
    print(f"Model downloaded successfully to: {path}")
except Exception as e:
    print(f"Error downloading model: {e}")
