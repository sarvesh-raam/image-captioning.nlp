import os
import requests
import zipfile
import json
import shutil
from tqdm import tqdm

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def setup_subset(num_images=500):
    # --- Config ---
    ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    IMG_BASE_URL = "http://images.cocodataset.org/train2014/COCO_train2014_"
    
    IMG_DIR = "coco_images/train2014"
    ANN_DIR = "annotations"
    
    if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
    if not os.path.exists(ANN_DIR): os.makedirs(ANN_DIR)

    # 1. Download Annotations
    print("Downloading COCO Annotations (approx 241MB)...")
    ann_zip = "annotations.zip"
    if not os.path.exists(ann_zip):
        download_file(ANN_URL, ann_zip)
    
    with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # 2. Parse Annotations for subset
    with open("annotations/captions_train2014.json", 'r') as f:
        data = json.load(f)
    
    images = data['images'][:num_images]
    img_ids = {img['id']: img['file_name'] for img in images}
    
    annotations = [ann for ann in data['annotations'] if ann['image_id'] in img_ids]
    
    # 3. Download images for subset
    print(f"Downloading {num_images} images...")
    subset_list = []
    for img in tqdm(images):
        img_filename = img['file_name']
        img_url = f"http://images.cocodataset.org/train2014/{img_filename}"
        img_path = os.path.join(IMG_DIR, img_filename)
        
        if not os.path.exists(img_path):
            try:
                img_data = requests.get(img_url).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
            except:
                print(f"Failed to download {img_filename}")
                continue

        # Find caption for this image
        img_anns = [a['caption'] for a in annotations if a['image_id'] == img['id']]
        if img_anns:
            subset_list.append({
                "image": img_filename,
                "caption": img_anns[0] # Use the first caption
            })

    # 4. Save list for trainer
    with open("coco_train_list.json", 'w') as f:
        json.dump(subset_list, f)
    
    print("Dataset subset setup complete.")
    print(f"Saved {len(subset_list)} image-caption pairs to coco_train_list.json")

if __name__ == "__main__":
    setup_subset(80000)
