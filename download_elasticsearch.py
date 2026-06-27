import os
import urllib.request
import zipfile
import sys

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading Elasticsearch: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading Elasticsearch: {read_so_far / (1024*1024):.1f} MB")
    sys.stdout.flush()

def main():
    url = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.17.0-windows-x86_64.zip"
    zip_path = "elasticsearch-8.17.0-windows-x86_64.zip"
    extract_path = "."

    print(f"Target URL: {url}")
    
    if not os.path.exists("elasticsearch-8.17.0"):
        print("Starting download...")
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\nDownload complete. Extracting zip file...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print("Extraction complete. Cleaning up zip file...")
        os.remove(zip_path)
        print("Elasticsearch folder 'elasticsearch-8.17.0' is ready!")
    else:
        print("Folder 'elasticsearch-8.17.0' already exists.")

if __name__ == "__main__":
    main()
