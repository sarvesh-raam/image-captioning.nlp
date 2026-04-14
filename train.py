import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from data_loader import get_loader, get_transforms, Vocabulary
from model import ImageCaptioningModel
import json
import os
import time
from tqdm import tqdm

def train():
    # --- Config ---
    # Balanced Mode: Optimization for Laptop Safety
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if DEVICE.type == "cuda":
        print(f"✅ BALANCED MODE: Using GPU ({torch.cuda.get_device_name(0)})")
        # Removed cudnn.benchmark to prevent high-power tuning spikes
    else:
        print("⚠️ WARNING: No GPU detected. Training will be extremely slow on CPU.")

    ROOT_TRAIN = "coco_images/train2014"
    TRAIN_JSON = "coco_train_list.json"
    VOCAB_PATH = "vocabulary.pkl"
    
    # Optimized for ViT-B-16 + RTX 5050 Smooth Browsing
    BATCH_SIZE = 16 
    NUM_WORKERS = 2 
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 35
    SAVE_PATH = "checkpoints/best_model.pth"
    
    # --- Data ---
    if not os.path.exists(TRAIN_JSON):
        print(f"Error: {TRAIN_JSON} not found. Ensure you have downloaded the COCO annotations.")
        return

    with open(TRAIN_JSON, 'r') as f:
        full_data = json.load(f)
    
    # Build vocabulary logic
    if os.path.exists(VOCAB_PATH):
        import pickle
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Loaded existing vocabulary. Size: {len(vocab)}")
    else:
        print("Building new vocabulary...")
        all_captions = [item["caption"] for item in full_data]
        vocab = Vocabulary(freq_threshold=2) 
        vocab.build_vocabulary(all_captions)
        import pickle
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary built and saved. Size: {len(vocab)}")
        
    train_loader, train_dataset = get_loader(
        ROOT_TRAIN, TRAIN_JSON, vocab, 
        transform=get_transforms(is_train=True), 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True # Essential for GPU performance
    )

    # --- Model ---
    model = ImageCaptioningModel(
        vocab_size=len(vocab), 
        embed_dim=512, 
        num_heads=8, 
        num_layers=6
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # Automatic Mixed Precision (AMP) for Speed
    
    best_loss = float('inf')
    start_epoch = 0
    
    # --- Auto Resume Safety Feature ---
    if os.path.exists(SAVE_PATH):
        print(f"Found existing checkpoint at {SAVE_PATH}. Resuming training!")
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint.get('loss', float('inf'))
    else:
        print("Starting fresh training.")
    
    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for batch_idx, (images, captions) in enumerate(loop):
            # Rapid Data Transfer to GPU
            images = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)
            
            # Input to decoder is everything except <EOS>
            targets = captions[:, 1:]
            captions = captions[:, :-1]
            
            # Use AMP for significantly faster training on RTX series
            with autocast():
                outputs = model(images, captions)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save every epoch but only update 'best' if loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'loss': best_loss
            }, SAVE_PATH)
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()

