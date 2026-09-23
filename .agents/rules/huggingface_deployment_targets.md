# Hugging Face Deployment & Repository Locations

## 1. Official Cloud Endpoints (Locked Targets)
- **Web App Space (Docker SDK):**  
  👉 [`suyashsahu00/GuardText-Cyberbullying-Detection`](https://huggingface.co/spaces/suyashsahu00/GuardText-Cyberbullying-Detection)  
  *Commits & History:* `https://huggingface.co/spaces/suyashsahu00/GuardText-Cyberbullying-Detection/commits/main`
- **Model Hub Weights (Transformers / SafeTensors):**  
  👉 [`suyashsahu00/muril-cyberbullying-detection`](https://huggingface.co/suyashsahu00/muril-cyberbullying-detection)  
  *Commits & History:* `https://huggingface.co/suyashsahu00/muril-cyberbullying-detection/commits/main`
- **GitHub Source Repository:**  
  👉 [`suyashsahu00/cyberbullying-detection`](https://github.com/suyashsahu00/cyberbullying-detection)

## 2. Deployment Workflows
- **Deploying Space Frontend & API:** Always run `python deploy_space_to_hf.py` (bundles `Dockerfile`, `backend/`, `frontend/`, and `src/`).
- **Pushing Model Weights:** Always run `python upload_to_hf.py` (uploads `models/muril_cyberbullying_v2/` safetensors and configs to `suyashsahu00/muril-cyberbullying-detection`).
- **Render vs Space Priority:** When Render monthly free limits are exhausted, Hugging Face Space is the primary, permanent production deployment (16 GB RAM, Docker container).
