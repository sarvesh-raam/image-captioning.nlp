import os
import requests
import zipfile
import json
from tqdm import tqdm
import concurrent.futures

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def setup_subset(num_images=80000):
    ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    IMG_DIR = "coco_images/train2014"
    ANN_DIR = "annotations"
    
    if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
    if not os.path.exists(ANN_DIR): os.makedirs(ANN_DIR)

    # 1. Download Annotations
    ann_zip = "annotations.zip"
    if not os.path.exists(ann_zip):
        print("Downloading COCO Annotations (approx 241MB)...")
        download_file(ANN_URL, ann_zip)
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    # 2. Parse Annotations
    with open("annotations/captions_train2014.json", 'r') as f:
        data = json.load(f)
    
    images = data['images'][:num_images]
    
    # Fast Dictionary Lookup for Captions (Massive Speedup)
    caption_dict = {}
    for ann in data['annotations']:
        if ann['image_id'] not in caption_dict:
            caption_dict[ann['image_id']] = ann['caption']
            
    # 3. Multithreaded Image Downloader
    print(f"Downloading {num_images} images using FAST Multi-threading...")
    subset_list = []

    def process_image(img):
        img_filename = img['file_name']
        img_url = f"http://images.cocodataset.org/train2014/{img_filename}"
        img_path = os.path.join(IMG_DIR, img_filename)
        
        # Download if we haven't already
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url, timeout=15)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
            except:
                return None
                
        # Connect to its specific caption
        cap = caption_dict.get(img['id'])
        if cap:
            return {"image": img_filename, "caption": cap}
        return None

    # Run 50 downloads at the exact same time!
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(tqdm(executor.map(process_image, images), total=len(images)))
        
    for r in results:
        if r is not None:
            subset_list.append(r)

    # 4. Save list for trainer
    with open("coco_train_list.json", 'w') as f:
        json.dump(subset_list, f)
    
    print(f"Dataset setup complete. Saved {len(subset_list)} image pairs.")

if __name__ == "__main__":
    setup_subset(80000)
