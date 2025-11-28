import os 
import sys
from huggingface_hub import snapshot_download

BASE_MODEL = "Canonik/Autotorino-Llama-3.1-8B-instruct_v2" 
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "autotorino")

def download():
    print(f"\n\n\nDownloading {BASE_MODEL} in {MODEL_DIR}.\n\n\n")
    print("__________________________________________________________________________________")
    print(f"\n\n\nThis operation may take several minutes, keep computer on and plugged in.\n\n\n")
    print("__________________________________________________________________________________")

    try:
        snapshot_download(
            repo_id = BASE_MODEL,
            local_dir= MODEL_DIR,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.git*", "*.md*"]
            )
        
        print(f"Download complete!\n Model and tokenizer saved in: {MODEL_DIR}\n")
    
    except Exception as error:
        print(f"Error during download of model: {error}\n")
        print(f"Try checking your connection and download again\n")
        print("__________________________________________________________________________________")
        sys.exit(1)

if __name__ == "__main__":
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
    except Exception as error:
        print(f"Unable to create directory at {MODEL_DIR} filepath due to: {error}")
    download()
