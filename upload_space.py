import os
import tempfile
import shutil
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "suyashsahu00/GuardText-Cyberbullying-Detection"

SPACE_README = """---
title: GuardText AI - Cyberbullying Detection Tool
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

# GuardText AI &mdash; Cyberbullying Detection Tool

Web application for real-time multilingual cyberbullying detection, target categorization, and trigger-word explainability powered by Google MuRIL Transformer.

- **Model Hub Weights:** [`suyashsahu00/muril-cyberbullying-detection`](https://huggingface.co/suyashsahu00/muril-cyberbullying-detection)
- **Categories:** Age, Ethnicity, Gender, Religion, Other Harassment, Safe / Not Bullying
- **Supported Languages:** English, Hindi, and Hinglish
"""

def main():
    api = HfApi()
    
    print(f"1. Checking Hugging Face Space: {SPACE_REPO_ID} (Static SDK)...")
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="static",
        exist_ok=True
    )
    print(" Space repository confirmed!")

    with tempfile.TemporaryDirectory() as td:
        # 1. README
        with open(os.path.join(td, "README.md"), "w", encoding="utf-8") as f:
            f.write(SPACE_README)
            
        # 2. index.html (with relative asset links for static serving)
        with open("frontend/templates/index.html", "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("{{ url_for('static', filename='css/style.css') }}", "css/style.css")
        html = html.replace("{{ url_for('static', filename='js/app.js') }}", "js/app.js")
        
        with open(os.path.join(td, "index.html"), "w", encoding="utf-8") as f:
            f.write(html)
            
        # 3. CSS
        os.makedirs(os.path.join(td, "css"), exist_ok=True)
        shutil.copy2("frontend/static/css/style.css", os.path.join(td, "css", "style.css"))
        
        # 4. JS
        os.makedirs(os.path.join(td, "js"), exist_ok=True)
        shutil.copy2("frontend/static/js/app.js", os.path.join(td, "js", "app.js"))

        print("\n2. Uploading static files to Hugging Face...")
        api.upload_folder(
            folder_path=td,
            repo_id=SPACE_REPO_ID,
            repo_type="space"
        )
        print(f" Space deployed successfully!")
        print(f" Live Webpage URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")

if __name__ == "__main__":
    main()
