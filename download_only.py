import os
from huggingface_hub import snapshot_download

# Mirror Strategy
mirror_name = "NousResearch/Meta-Llama-3-8B-Instruct"

print(f"Downloading {mirror_name} to cache...")
snapshot_download(
    repo_id=mirror_name,
    resume_download=True
)
print("Download complete.")
