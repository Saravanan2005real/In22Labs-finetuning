import requests
import time
import sys

url = "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/model.safetensors"
print("Sending GET request to Hugging Face...", flush=True)
start = time.time()
try:
    r = requests.get(url, stream=True, timeout=10)
    print(f"Response status: {r.status_code}", flush=True)
    print(f"Content Length: {r.headers.get('content-length')} bytes", flush=True)
    print("Starting chunk download...", flush=True)
    downloaded = 0
    last_print = time.time()
    for chunk in r.iter_content(chunk_size=1024*1024):
        if chunk:
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 1.0:
                print(f"[{time.strftime('%H:%M:%S')}] Downloaded: {downloaded / (1024*1024):.2f} MB (Speed: {len(chunk)/(now-last_print)/1024:.2f} KB/s)", flush=True)
                last_print = now
    print(f"Download finished in {time.time() - start:.2f}s", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
