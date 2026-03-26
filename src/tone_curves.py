"""
Tone curve extraction and application
"""

import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class ToneCurves:
    """
    Extracts and applies characteristic tone curves from film samples
    """
    
    def __init__(self):
        self.curve_params = None
        
    @staticmethod
    def sigmoid_curve(x, midpoint, steepness, max_val):
        """Sigmoid tone curve function"""
        return max_val / (1 + np.exp(-steepness * (x - midpoint)))
    
    @staticmethod
    def gamma_curve(x, gamma):
        """Simple gamma curve"""
        return 255 * np.power(x / 255.0, gamma)
    
    def analyze_film_contrast(self, film_samples: list):
        """
        Analyze contrast characteristics from film samples
        
        Args:
            film_samples: List of film images in RGB format
        """
        # Extract luminance histograms
        luminance_samples = []
        
        for img in film_samples:
            # Convert to grayscale for luminance
            gray = np.mean(img, axis=2)
            luminance_samples.append(gray)
        
        # Compute average histogram
        hist_sum = np.zeros(256)
        for lum in luminance_samples:
            hist, _ = np.histogram(lum.flatten(), bins=256, range=(0, 255))
            hist_sum += hist
        
        avg_hist = hist_sum / len(luminance_samples)
        
        # Estimate curve parameters from histogram shape
        # This is a simplified approach - fit to cumulative histogram
        cumsum = np.cumsum(avg_hist)
        cumsum_normalized = cumsum / cumsum[-1] * 255
        
        # Fit sigmoid to the cumulative distribution
        x_data = np.arange(256)
        try:
            params, _ = curve_fit(
                self.sigmoid_curve,
                x_data,
                cumsum_normalized,
                p0=[128, 0.05, 255],  # initial guess
                maxfev=10000
            )
            self.curve_params = {
                'type': 'sigmoid',
                'midpoint': params[0],
                'steepness': params[1],
                'max_val': params[2]
            }
        except:
            # Fallback to gamma
            print("Sigmoid fit failed, using gamma curve")
            self.curve_params = {'type': 'gamma', 'gamma': 1.2}
        
        print(f"Extracted curve parameters: {self.curve_params}")
        
    def apply_curve(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Apply learned tone curve to an image
        
        Args:
            image: Input image in RGB format
            strength: Curve application strength (0-1)
            
        Returns:
            Image with tone curve applied
        """
        if self.curve_params is None:
            raise ValueError("Must call analyze_film_contrast() first")
        
        # Create lookup table
        lut = np.arange(256, dtype=np.float32)
        
        if self.curve_params['type'] == 'sigmoid':
            lut = self.sigmoid_curve(
                lut,
                self.curve_params['midpoint'],
                self.curve_params['steepness'],
                self.curve_params['max_val']
            )
        else:  # gamma
            lut = self.gamma_curve(lut, self.curve_params['gamma'])
        
        # Blend with linear curve based on strength
        linear = np.arange(256, dtype=np.float32)
        lut = linear * (1 - strength) + lut * strength
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        
        # Apply LUT to image
        result = cv2.LUT(image, lut)
        
        return result
    
    def visualize_curve(self):
        """Plot the learned tone curve"""
        if self.curve_params is None:
            print("No curve to visualize")
            return
        
        x = np.arange(256)
        if self.curve_params['type'] == 'sigmoid':
            y = self.sigmoid_curve(
                x,
                self.curve_params['midpoint'],
                self.curve_params['steepness'],
                self.curve_params['max_val']
            )
        else:
            y = self.gamma_curve(x, self.curve_params['gamma'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, x, 'r--', label='Linear (Original)', alpha=0.5)
        plt.plot(x, y, 'b-', label='Film Curve', linewidth=2)
        plt.xlabel('Input Value')
        plt.ylabel('Output Value')
        plt.title('Tone Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()