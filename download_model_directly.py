import os
import requests
import time

FILES = [
    ".gitattributes",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

REPO_URL = "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/"
TARGET_DIR = "llama_base_model"

os.makedirs(TARGET_DIR, exist_ok=True)

print("Starting direct model download bypass pipeline...", flush=True)

for filename in FILES:
    target_path = os.path.join(TARGET_DIR, filename)
    url = REPO_URL + filename
    
    print(f"\nDownloading {filename}...", flush=True)
    
    start_time = time.time()
    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code != 200:
            print(f"Error downloading {filename}: Status code {r.status_code}", flush=True)
            continue
            
        total_size = int(r.headers.get('content-length', 0))
        
        # Download files with size > 1MB with progress updates
        if total_size > 1024*1024:
            print(f"File size: {total_size / (1024*1024):.2f} MB", flush=True)
            downloaded = 0
            last_print = time.time()
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print >= 2.0 or downloaded == total_size:
                            print(f"[{time.strftime('%H:%M:%S')}] Downloaded: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({(downloaded/total_size)*100:.1f}%)", flush=True)
                            last_print = now
        else:
            # Download small files instantly
            with open(target_path, "wb") as f:
                f.write(r.content)
            print(f"Saved {filename} ({os.path.getsize(target_path)} bytes)", flush=True)
            
        print(f"Finished {filename} in {time.time() - start_time:.2f}s", flush=True)
        
    except Exception as e:
        print(f"Failed to download {filename}: {e}", flush=True)

print("\nDirect download pipeline finished!", flush=True)
