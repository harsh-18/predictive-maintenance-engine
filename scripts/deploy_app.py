
import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "harshverma27"
SPACE_REPO = f"{HF_USERNAME}/predictive-maintenance-app"

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

for f in ["app.py", "requirements.txt", "Dockerfile", "README.md"]:
    api.upload_file(
        path_or_fileobj=os.path.join("deployment", f),
        path_in_repo=f,
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print(f"Uploaded {f} -> {SPACE_REPO}")
print(f"Live at: https://huggingface.co/spaces/{SPACE_REPO}")
