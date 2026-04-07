import gradio as gr
import torch
import torch.nn as nn
from model import ImageCaptioningModel
from data_loader import Vocabulary, get_transforms
from inference import CaptionGenerator
from PIL import Image
import os

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# Let's ensure a dummy model loads if no checkpoint exists
def load_app_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}! Please train the model first.")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Load vocab from pickle if it exists, otherwise from checkpoint
    if os.path.exists("vocabulary.pkl"):
        import pickle
        with open("vocabulary.pkl", "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = checkpoint['vocab']
    
    model = ImageCaptioningModel(
        vocab_size=len(vocab), 
        embed_dim=512, 
        num_heads=8, 
        num_layers=6
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

try:
    model, vocab = load_app_model()
    generator = CaptionGenerator(model, vocab, DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    # Placeholder for UI in case of failure
    model, vocab, generator = None, None, None

def predict(image, beam_size=3):
    if generator is None:
        return "Model not loaded. Please train the model first and ensure 'checkpoints/best_model.pth' exists."
    
    # Convert Gradio image to PIL
    pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Generate caption
    caption_list = generator.generate_caption_beam_search(pil_img, beam_size=int(beam_size))
    caption = " ".join(caption_list).capitalize() + "."
    
    return caption

# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🖼️ Hybrid CNN-Transformer Image Captioner")
    gr.Markdown("Upload an image and the model will generate a descriptive caption using a ResNet50-Transformer architecture.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image")
            beam_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Beam Size (Inference Quality)")
            btn = gr.Button("Generate Caption", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Generated Caption", placeholder="The caption will appear here...")

    btn.click(fn=predict, inputs=[input_image, beam_slider], outputs=output_text)
    
    gr.Examples(
        examples=[["coco_images/val2014/example.jpg"]], # Add example images if available
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(share=True)
