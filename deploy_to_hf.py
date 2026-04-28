import os
from huggingface_hub import HfApi, login

def deploy_to_spaces():
    print("🚀 Starting Professional Deployment to Hugging Face Spaces...")
    
    # 1. Ask for Hugging Face Token
    print("\n[Step 1] Authentication")
    print("Get your token from: https://huggingface.co/settings/tokens (Make sure it has 'WRITE' permission)")
    token = input("Enter your Hugging Face Access Token: ").strip()
    login(token=token)
    
    # 2. Get Space Details
    print("\n[Step 2] Space Details")
    username = input("Enter your Hugging Face username: ").strip()
    space_name = input("Enter the name of your Space (e.g., hybrid-image-captioner): ").strip()
    repo_id = f"{username}/{space_name}"
    
    api = HfApi()
    
    # 3. Define what files to upload to production
    print("\n[Step 3] Uploading Application & Model...")
    
    # Essential files for the app to run
    files_to_upload = [
        "app.py",
        "api.py",
        "model.py",
        "inference.py",
        "data_loader.py",
        "requirements.txt",
        "vocabulary.pkl"
    ]
    
    # Uploading code files
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="space"
            )
        else:
            print(f"⚠️ Warning: {file} not found locally.")

    # Uploading the large model checkpoint securely
    if os.path.exists("checkpoints/best_model.pth"):
        print("Uploading best_model.pth (This might take a while for large 2GB-3GB models...)")
        api.upload_file(
            path_or_fileobj="checkpoints/best_model.pth",
            path_in_repo="checkpoints/best_model.pth",
            repo_id=repo_id,
            repo_type="space"
        )
        print("✅ Model uploaded successfully!")
    else:
        print("⚠️ Warning: checkpoints/best_model.pth not found! Make sure to train first.")

    print(f"\n🎉 Deployment Triggered! Check your space at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy_to_spaces()
