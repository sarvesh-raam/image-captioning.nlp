import os
from getpass import getpass
from huggingface_hub import HfApi, login

# Files the Docker image actually needs (see Dockerfile)
FILES_TO_UPLOAD = ["Dockerfile", ".dockerignore", "requirements.txt", "api.py", "model.py", "inference.py", "data_loader.py"]
STALE_SPACE_FILES = ["checkpoints/best_model.pth", "app.py", "vocabulary.pkl"]


def deploy_to_spaces():
    print("[INIT] Deploying to Hugging Face Spaces...")

    # 1. Authentication (token read from HF_TOKEN or prompted without echo)
    print("\n[Step 1] Authentication")
    token = os.environ.get("HF_TOKEN") or getpass("Enter your Hugging Face WRITE token (input hidden): ").strip()
    login(token=token)

    # 2. Space details
    print("\n[Step 2] Space Details")
    username = input("Hugging Face username [sarveshraam]: ").strip() or "sarveshraam"
    space_name = input("Space name [image-captioner]: ").strip() or "image-captioner"
    repo_id = f"{username}/{space_name}"
    model_repo_id = f"{username}/{space_name}-model"

    api = HfApi()

    # 3. Upload the model checkpoint to a MODEL repository (bypasses the Space storage limit).
    # This must happen before the Space rebuilds, because the Docker build downloads it.
    # The API loads model.pth (made by export_model.py) in preference to best_model.pth.
    for local, remote in [("checkpoints/model.pth", "model.pth"), ("checkpoints/best_model.pth", "best_model.pth")]:
        if not os.path.exists(local):
            continue
        if input(f"\nUpload {local} to {model_repo_id} as {remote}? [y/N]: ").strip().lower() == "y":
            api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=model_repo_id, repo_type="model")
            print(f"[SUCCESS] {remote} uploaded.")
        break

    # 4. Remove files that would shadow the Model Hub checkpoint or are no longer used
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="space"))
    for path in STALE_SPACE_FILES:
        if path in existing:
            print(f"Removing stale {path} from the Space...")
            api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type="space")

    # 5. Upload the application code in a single commit (one rebuild instead of one per file)
    print("\n[Step 3] Uploading application code...")
    missing = [f for f in FILES_TO_UPLOAD if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"[ERROR] Missing files: {missing}")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=FILES_TO_UPLOAD,
        commit_message="Deploy API",
    )

    print(f"\n[DONE] Deployment triggered! Check your space at: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    deploy_to_spaces()
