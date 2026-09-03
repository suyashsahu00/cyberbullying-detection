import os
import shutil
import tempfile
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "suyashsahu00/GuardText-Cyberbullying-Detection"

SPACE_README = """---
title: GuardText AI - Cyberbullying Detection Tool
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# GuardText AI &mdash; Cyberbullying Detection Tool

Web application for real-time multilingual cyberbullying detection, target categorization, and trigger-word explainability powered by **Google MuRIL Transformer**.

- **Model Hub Weights:** [`suyashsahu00/muril-cyberbullying-detection`](https://huggingface.co/suyashsahu00/muril-cyberbullying-detection)
- **Categories:** Age, Ethnicity, Gender, Religion, Other Harassment, Safe / Not Bullying
- **Supported Languages:** English, Hindi, and Hinglish
"""

def main():
    api = HfApi()
    
    print(f"1. Creating Hugging Face Space: {SPACE_REPO_ID} (Docker SDK)...")
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )
    print(" Space repository created or already exists!")

    with tempfile.TemporaryDirectory() as staging_dir:
        print(f"\n2. Staging files into {staging_dir}...")
        
        # Copy Dockerfile
        shutil.copy2("Dockerfile", os.path.join(staging_dir, "Dockerfile"))
        
        # Write Space README.md
        with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(SPACE_README)
            
        # Copy backend folder
        shutil.copytree("backend", os.path.join(staging_dir, "backend"))
        
        # Copy frontend folder
        shutil.copytree("frontend", os.path.join(staging_dir, "frontend"))
        
        # Copy src folder
        shutil.copytree("src", os.path.join(staging_dir, "src"))
        
        # Copy models/baseline_model.pkl (excluding the large 950MB MuRIL folder since it fetches from HF Hub)
        os.makedirs(os.path.join(staging_dir, "models"), exist_ok=True)
        if os.path.exists("models/baseline_model.pkl"):
            shutil.copy2("models/baseline_model.pkl", os.path.join(staging_dir, "models", "baseline_model.pkl"))

        print("\n3. Uploading Space files to Hugging Face...")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=SPACE_REPO_ID,
            repo_type="space"
        )
        print(f" Space deployed successfully!")
        print(f" Webpage URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")

if __name__ == "__main__":
    main()
