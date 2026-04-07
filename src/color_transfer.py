import numpy as np
from typing import List

#from string.templatelib import convert
from utils import rgb_to_lab, lab_to_rgb, compute_image_stats
from pathlib import Path

class ColorTransfer:
    """
    Implements color transfer from film samples to digital images.
    Based on Reinhard et al. "Color Transfer between Images" (2001).
    This transfers the color distribution (mean and standard deviation) from film photos to digital images.
    """

    def __init__(self):
        self.target_mean = None  # stores mean color values in LAB space from film samples (average color)
        self.target_std = None   # stores standard deviation of colors in LAB space from film samples (variance of colors)

    def fit(self, film_samples: List[np.ndarray]):
        '''
        Training function. Analyze film samples to learn color statistics. 
        Args: 
            film_samples: List of film images in RGB format
        Returns:
            None (stores learned color stats in class variables)
        '''
        lab_samples = []
        for img in film_samples: 
            ''' 
            converts each image into LAB color space to analyze color distribution. 
            (lab separates lightness (luminance) from color, making it easier to analyze and transfer colors)
            '''
            lab_img = rgb_to_lab(img)
            lab_samples.append(lab_img)
        
        means = []
        stds = []
        for lab_img in lab_samples:
            mean, std = compute_image_stats(lab_img) # compute mean and std for each color channel (L - lightness, A - greens and reds, B - blues and yellows)
            means.append(mean)
            stds.append(std)

        # takes the average of all of the means and std's across all images for the film stock
        # to give ONE target mean and std representing each film stock.

        # ex: L = ___ A = ___ B = ___; film stock typically has L brightness, A color balance 
        # (positive or negative = red or greeen, respectively), and B color balance (positive or negative = yellow 
        # or blue, respectively)
        # std represents color variance - higher std = more color variation, lower std = more consistent colors
        self.target_mean = np.mean(means, axis=0)
        self.target_std = np.mean(stds, axis=0)

        print(f"Learned color stats (LAB) from {len(film_samples)} samples - Mean: {self.target_mean} Std: {self.target_std}")
        

    def transform(self, image: np.ndarray, strength: float= 1.0) -> np.ndarray:
        '''
        Apply color transfer to digital input photo to match the film color distribution.

        Input:
            image: input digital image in RGB format
            strength: transfer strength 0-1; 1 = full transfer, 0 = no change

        Output:
            transformed image in RGB format with film-like colors 
        '''

        # transfer cannot be applied without fitting first; check if target mean and std have been learned before applying color transfer
        if self.target_mean is None or self.target_std is None:
            raise ValueError("ColorTransfer model has not been fitted with film samples. Call fit() first.")
        
        lab_img = rgb_to_lab(image) # convert input image to LAB color space
        source_mean, source_std = compute_image_stats(lab_img) # get source image color statistics (mean and std for digital input photo)

        ### Reinhard color transfer algorithm:
        # Step 1: 
        # Center the colors by subtracting the source mean from each pixel's color values
        # (essentially normalizes image my shifting all pixel values so the average becomes 0)
        lab_transformed = lab_img - source_mean

        # Step 2: 
        # Scale by std ratio; calculate ratio between target std (from film samples) 
        # and source std (from input photo) for each channel, then multiply pixel values by this ratio

        # this adjusts the color distribution to match the variance of the target film stock
        '''
        ex: 
            target std (from film stock) = [15.0, 10.0 , 12,0] (L, A, B channels)
            source std (from input photo) = [20.5, 15.2, 18.3] (L, A, B channels)
            std ratio = target std / source std = [1.37, 1.52, 1.53]
            this means lightness will vary by 1.37x, red-green balance will vary by 1.52x, 
            and blue-yellow balance will vary by 1.53x compared to the original input photo, 
            to match the film stock's color variance 
        '''
        std_ratio = self.target_std/(source_std + 1e-8)  # Add to avoid division by zero
        lab_transformed *= std_ratio 

        # Step 3: 
        # Add target mean (from film samples) to shift colors to match the average color of the target film stock
        lab_transformed = lab_transformed + self.target_mean
    
        # Step 4:
        # Linear interpolation between the original and transformed results based on strength input (0-1) to allow for partial transfer
        # Ex: strength = 0.5 would blend 50% of the film stock color characteristics with 50% of the original input photo's colors
        lab_transformed = lab_img * (1 - strength) + lab_transformed * strength

        rgb_transformed  = lab_to_rgb(lab_transformed) # convert back to RGB color space for output
        return rgb_transformed
    

    def save_stats(self, filepath: str):
        '''save learned color stats to avoid re-fitting; can load later to apply color transfer'''
        np.savez(filepath, mean=self.target_mean, std=self.target_std)
        print(f"Saved color stats to {filepath}")

    def load_stats(self, filepath: str):
        '''load previously saved color stats to apply transfer without refitting'''
        data = np.load(filepath)
        self.target_mean = data['mean']
        self.target_std = data['std']
        print(f"Loaded color stats from {filepath}")