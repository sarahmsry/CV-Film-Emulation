import sys
sys.path.append('src')
import cv2
import numpy as np
from integration import FilmEmulationPipeline
from utils import load_image, load_dataset, save_image
from PIL import Image

pipeline = FilmEmulationPipeline('portra_400')
pipeline.load_model('results/models/portra400_model/')  # adjust path as needed

test_images = load_dataset("data/digital_samples/")

for i, img in enumerate(test_images):
    # resize images for processing if they are > 2000px on h or w
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    result = pipeline.transform(img)
    comparison = np.hstack((img, result))
    save_image(comparison, f"results/comparisons/digital_sample_{i+1}_comparison.jpg")

print("Saved comparison images to results/comparisons/")