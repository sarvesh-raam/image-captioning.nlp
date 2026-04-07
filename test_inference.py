import torch
from model import ImageCaptioningModel
from data_loader import Vocabulary, get_transforms
from inference import CaptionGenerator
from PIL import Image
import os
import pickle

def test_on_car():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    VOCAB_PATH = "vocabulary.pkl"

    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(VOCAB_PATH):
        print("Model or vocab not found yet. Still training?")
        return

    # Load everything
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = ImageCaptioningModel(
        vocab_size=len(vocab), 
        embed_dim=512, 
        num_heads=8, 
        num_layers=6
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    generator = CaptionGenerator(model, vocab, DEVICE)
    
    # Let's test on the car image if available
    test_img_path = "car_test.jpg" # Need to download one if missing
    if not os.path.exists(test_img_path):
        import requests
        print("Downloading a car test image...")
        img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" # Using a dog for base, need a car
        url_car = "https://images.shutterstock.com/image-photo/modern-suv-car-driving-mountain-260nw-2115141029.jpg"
        try:
            with open(test_img_path, 'wb') as f:
                f.write(requests.get(url_car).content)
        except:
            print("Failed to download test car image.")
            return

    image = Image.open(test_img_path).convert("RGB")
    caption = generator.generate_caption_beam_search(image, beam_size=3)
    print(f"--- TEST RESULT ---")
    print(f"Generated: {' '.join(caption)}")

if __name__ == "__main__":
    test_on_car()
