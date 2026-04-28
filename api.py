from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from model import ImageCaptioningModel
from data_loader import Vocabulary, get_transforms
from inference import CaptionGenerator
from PIL import Image
import io
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
# High-Performance Mode (RTX 50-series Enabled)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# Global generator instance
generator = None

# Manual load for stability
@app.on_event("startup")
def load_model():
    global generator
    print("DEBUG: load_model startup event triggered", flush=True)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"DEBUG: Attempting to load checkpoint from {CHECKPOINT_PATH}...", flush=True)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            print("DEBUG: Checkpoint loaded into memory.", flush=True)
            
            if 'vocab' in checkpoint:
                vocab = checkpoint['vocab']
            elif os.path.exists("vocabulary.pkl"):
                import pickle
                with open("vocabulary.pkl", "rb") as f:
                    vocab = pickle.load(f)
            else:
                raise Exception("No vocabulary available")
            
            embed_dim = checkpoint.get('embed_dim', 512)
            num_heads = checkpoint.get('num_heads', 8)
            num_layers = checkpoint.get('num_layers', 6)

            # Fallback inference if parameters weren't saved in older checkpoints
            if 'embed_dim' not in checkpoint and 'model_state_dict' in checkpoint:
                try:
                    state_dict = checkpoint['model_state_dict']
                    if 'encoder.cnn_projection.weight' in state_dict:
                        embed_dim = state_dict['encoder.cnn_projection.weight'].shape[0]
                    
                    layer_keys = [k for k in state_dict.keys() if 'decoder.transformer_decoder.layers' in k]
                    if layer_keys:
                        nums = [int(k.split('decoder.transformer_decoder.layers.')[1].split('.')[0]) 
                                for k in layer_keys if k.split('decoder.transformer_decoder.layers.')[1].split('.')[0].isdigit()]
                        if nums:
                            num_layers = max(nums) + 1
                except Exception as e:
                    print(f"DEBUG: Could not infer arch from state_dict: {e}")

            model = ImageCaptioningModel(
                vocab_size=len(vocab), 
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                num_layers=num_layers
            ).to(DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            generator = CaptionGenerator(model, vocab, DEVICE)
            print("INFO: Vision model successfully loaded on CPU (Stability Mode).", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
    else:
        print(f"WARNING: Checkpoint path not found: {CHECKPOINT_PATH}")

@app.post("/predict")
async def predict(file: UploadFile = File(...), beam_size: int = 3):
    if generator is None:
        return {"error": "Model not loaded. Please train first."}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate caption
        caption_list = generator.generate_caption_beam_search(image, beam_size=beam_size)
        
        # Aggressive join and cleanup
        words = [str(w) for w in caption_list if w and str(w) not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "undefined"]]
        
        # Perfect spacing joining
        final_caption = " ".join(words).strip()
        
        # Robust capitalization
        if final_caption:
            result = final_caption[0].upper() + final_caption[1:]
            if not result.endswith("."):
                result += "."
        else:
            result = "No caption generated."
            
        return {"caption": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": generator is not None}

if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8001)
