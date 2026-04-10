"""
Utility functions for loading, saving, and processing images, as well as displaying 
comparisons and computing statistics. Used across multiple modules for consistent and efficient image handling.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt


def load_image(path: str) -> np.ndarray:
    """Load image and convert from BGR to RGB"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str):
    """Save image, converting from RGB to BGR"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def load_dataset(directory: str, max_images: int = None) -> List[np.ndarray]:
    """Load all images from a directory"""
    image_dir = Path(directory)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    images = []
    for ext in extensions:
        for img_path in image_dir.glob(ext):
            if max_images and len(images) >= max_images:
                break
            try:
                images.append(load_image(str(img_path)))
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    
    return images


def show_comparison(original: np.ndarray, processed: np.ndarray, 
                   title1: str = "Original", title2: str = "Processed"):
    """Display side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB color space"""
    # OpenCV expects BGR, so convert RGB->BGR->LAB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return img_lab.astype(np.float32)


def lab_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB color space"""
    img_lab_uint8 = np.clip(img, 0, 255).astype(np.uint8)  # cv2 needs uint8 input
    img_bgr = cv2.cvtColor(img_lab_uint8, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb, 0, 255).astype(np.uint8)


def compute_image_stats(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
   ''' 
   computes the mean and std across the height and width dimensions for each channel,
   resulting in a mean and std for each color channel (L, A, B).
    Input:
         img: input image in LAB color space as a numpy array with shape (Height, Width, 3 channels)    
    Returns:
         mean: mean color values for each channel (L, A, B)
         std: standard deviation for each channel (L, A, B)
   '''
   mean = np.mean(img, axis=(0, 1))
   std = np.std(img, axis=(0, 1))
   return mean, std


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range"""
    img_normalized = img - img.min()
    img_normalized = img_normalized / (img_normalized.max() + 1e-8) * 255
    return img_normalized.astype(np.uint8)