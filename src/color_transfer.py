"""
Color transfer implementation using Reinhard et al. method
"""

import numpy as np
from typing import List
from utils import rgb_to_lab, lab_to_rgb, compute_image_stats


class ColorTransfer:
    """
    Implements color transfer from film samples to digital images
    Based on Reinhard et al. "Color Transfer between Images" (2001)
    """
    
    def __init__(self):
        self.target_mean = None
        self.target_std = None
        
    def fit(self, film_samples: List[np.ndarray]):
        """
        Learn color statistics from film samples
        
        Args:
            film_samples: List of film images in RGB format
        """
        # Convert all samples to LAB
        lab_samples = [rgb_to_lab(img) for img in film_samples]
        
        # Compute average statistics across all samples
        means = []
        stds = []
        
        for lab_img in lab_samples:
            mean, std = compute_image_stats(lab_img)
            means.append(mean)
            stds.append(std)
        
        # Average across all film samples
        self.target_mean = np.mean(means, axis=0)
        self.target_std = np.mean(stds, axis=0)
        
        print(f"Learned color stats from {len(film_samples)} samples")
        print(f"Target mean (LAB): {self.target_mean}")
        print(f"Target std (LAB): {self.target_std}")
        
    def transform(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Apply color transfer to a digital image
        
        Args:
            image: Input image in RGB format
            strength: Transfer strength (0-1), 1.0 = full transfer
            
        Returns:
            Transformed image in RGB format
        """
        if self.target_mean is None:
            raise ValueError("Must call fit() before transform()")
        
        # Convert to LAB
        lab_img = rgb_to_lab(image)
        
        # Compute source statistics
        source_mean, source_std = compute_image_stats(lab_img)
        
        # Apply color transfer
        # Step 1: Subtract source mean
        lab_transformed = lab_img - source_mean
        
        # Step 2: Scale by std ratio
        std_ratio = self.target_std / (source_std + 1e-8)
        lab_transformed = lab_transformed * std_ratio
        
        # Step 3: Add target mean
        lab_transformed = lab_transformed + self.target_mean
        
        # Blend with original based on strength
        lab_transformed = lab_img * (1 - strength) + lab_transformed * strength
        
        # Convert back to RGB
        result = lab_to_rgb(lab_transformed)
        
        return result
    
    def save_stats(self, filepath: str):
        """Save learned statistics to file"""
        np.savez(filepath, mean=self.target_mean, std=self.target_std)
        print(f"Saved color stats to {filepath}")
    
    def load_stats(self, filepath: str):
        """Load learned statistics from file"""
        data = np.load(filepath)
        self.target_mean = data['mean']
        self.target_std = data['std']
        print(f"Loaded color stats from {filepath}")