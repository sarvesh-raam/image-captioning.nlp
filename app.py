import gradio as gr
import torch
from PIL import Image

from api import resolve_checkpoint
from inference import CaptionGenerator, format_caption, load_model

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model, vocab = load_model(resolve_checkpoint(), DEVICE)
    generator = CaptionGenerator(model, vocab, DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None


def predict(image, beam_size=3):
    if generator is None:
        return "Model not loaded. Train the model first or check that the checkpoint can be downloaded."
    if image is None:
        return "Please upload an image."

    pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
    return format_caption(generator.generate_caption_beam_search(pil_img, beam_size=int(beam_size)))


# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# Hybrid CNN-Transformer Image Captioner")
    gr.Markdown("Upload an image and the model will generate a descriptive caption using a ResNet50-Transformer architecture.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image")
            beam_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Beam Size (Inference Quality)")
            btn = gr.Button("Generate Caption", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Caption", placeholder="The caption will appear here...")

    btn.click(fn=predict, inputs=[input_image, beam_slider], outputs=output_text)

if __name__ == "__main__":
    demo.launch()
