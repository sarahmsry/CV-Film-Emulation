import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import warnings


class GrainSynthesis:
    """
    Synthesizes and applies film grain using frequency-domain analysis and luminance-dependent grain patterns. 
    This class generates grain that mimics the characteristics of real film grain, including size, intensity, and distribution based on image luminance.
    """
    
    def __init__(self, intensity: float = 0.1, size: float = 1.0):
        """
        Args:
            intensity: Grain intensity (0-1)
            size: Grain size multiplier
        """
        self.intensity = intensity
        self.size = size
        self.film_stats = None  # Store learned grain statistics from film samples
        
    def generate_grain(self, shape: tuple, luminance: np.ndarray = None) -> np.ndarray:
        """
        Generate film grain pattern
        
        Args:
            shape: Output shape (height, width)
            luminance: Optional luminance map for luminance-dependent grain
            
        Returns:
            Grain pattern
        """
        h, w = shape[:2]
        
        # Generate base noise
        grain = np.random.normal(0, 1, (h, w))
        
        # Apply Gaussian blur to create grain "clumps"
        sigma = self.size
        grain = gaussian_filter(grain, sigma=sigma)
        
        # Normalize
        grain = grain - grain.mean()
        grain = grain / (grain.std() + 1e-8)
        
        # Make grain intensity depend on luminance (more grain in shadows)
        if luminance is not None:
            # Normalize luminance to 0-1
            lum_normalized = luminance / 255.0
            # More grain in darker areas
            grain_mask = 1.0 - lum_normalized * 0.7
            grain = grain * grain_mask
        
        return grain
    
    def apply_grain(self, image: np.ndarray) -> np.ndarray:
        """
        Apply grain to an image
        
        Args:
            Image in RGB format
            
        Returns:
            Image with grain applied
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Compute luminance for luminance-dependent grain
        luminance = np.mean(img_float, axis=2)
        
        # Generate grain
        grain = self.generate_grain(image.shape[:2], luminance)
        
        # Apply grain to each channel
        for c in range(3):
            grain_channel = grain * self.intensity * 255
            img_float[:, :, c] = img_float[:, :, c] + grain_channel
        
        # Clip and convert back
        result = np.clip(img_float, 0, 255).astype(np.uint8)
        
        return result
    
    def extract_grain_from_image(self, image: np.ndarray, blur_radius: int = 5) -> np.ndarray:
        """
        Extract grain pattern from a film photo by subtracting blurred version.
        This isolates the noise/grain from the underlying image content.
        
        Args:
            image: Film photo in RGB format
            blur_radius: Radius for blur kernel (larger = more smoothing)
            
        Returns:
            Extracted grain pattern
        """
        img_float = image.astype(np.float32)
        
        # Create blurred version (removes grain, keep content)
        img_blurred = cv2.GaussianBlur(img_float, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Extract grain by subtracting
        grain = img_float - img_blurred
        
        return grain
    
    def analyze_grain_properties(self, image: np.ndarray) -> Dict:
        """
        Analyze grain properties from film photos.
        Extracts intensity, frequency characteristics, and luminance dependence.
        
        Args:
            image: Film photo in RGB format
            
        Returns:
            Dictionary with grain statistics
        """
        # Extract grain from each channel
        grain_r = self.extract_grain_from_image(image[:, :, 0:1])
        grain_g = self.extract_grain_from_image(image[:, :, 1:2])
        grain_b = self.extract_grain_from_image(image[:, :, 2:3])
        
        # Combine into single grain pattern (average across channels)
        grain = np.mean([grain_r, grain_g, grain_b], axis=0)
        
        # Calculate statistics
        grain_std = np.std(grain)
        grain_mean = np.mean(grain)
        
        # Analyze frequency domain (FFT)
        grain_fft = fftshift(fft2(grain.squeeze()))
        grain_power = np.abs(grain_fft) ** 2
        grain_freqs = np.mean(grain_power)
        
        # Analyze luminance dependence
        luminance = np.mean(image.astype(np.float32), axis=2)
        
        # Correlation between grain and luminance
        grain_flat = grain.flatten()
        lum_flat = luminance.flatten()
        
        # Normalize for correlation
        grain_norm = (grain_flat - grain_flat.mean()) / (grain_flat.std() + 1e-8)
        lum_norm = (lum_flat - lum_flat.mean()) / (lum_flat.std() + 1e-8)
        
        luminance_dependence = np.abs(np.mean(grain_norm * lum_norm))
        
        stats = {
            'grain_intensity': grain_std,
            'grain_mean_offset': grain_mean,
            'frequency_content': grain_freqs,
            'luminance_dependence': luminance_dependence,
            'dynamic_range': np.std(grain)
        }
        
        return stats
    
    def load_and_analyze_film_samples(self, film_directory: str) -> Dict:
        """
        Load film photos and analyze their grain characteristics.
        
        Args:
            film_directory: Path to directory containing film samples 
            
        Returns:
            Aggregated statistics from all samples
        """
        film_path = Path(film_directory)
        
        if not film_path.exists():
            print(f"Directory not found: {film_directory}")
            return None
        
        # Find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in extensions:
            image_files.extend(film_path.glob(ext))
        
        if not image_files:
            print(f"No images found in {film_directory}")
            return None
        
        print(f"Found {len(image_files)} film samples to analyze...")
        
        all_stats = []
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Analyze grain
                stats = self.analyze_grain_properties(img_rgb)
                all_stats.append(stats)
                
                print(f"Analyzed: {img_path.name}")
                
            except Exception as e:
                print(f"Error analyzing {img_path.name}: {e}")
                continue
        
        if not all_stats:
            return None
        
        aggregated_stats = {
            'samples_analyzed': len(all_stats),
            'avg_grain_intensity': np.mean([s['grain_intensity'] for s in all_stats]),
            'avg_grain_mean': np.mean([s['grain_mean_offset'] for s in all_stats]),
            'avg_frequency': np.mean([s['frequency_content'] for s in all_stats]),
            'avg_luminance_dependence': np.mean([s['luminance_dependence'] for s in all_stats]),
            'grain_intensity_range': (
                min([s['grain_intensity'] for s in all_stats]),
                max([s['grain_intensity'] for s in all_stats])
            )
        }
        
        self.film_stats = aggregated_stats
        
        print(f"\n=== Film Grain Analysis Summary ===")
        print(f"Samples analyzed: {aggregated_stats['samples_analyzed']}")
        print(f"Average grain intensity: {aggregated_stats['avg_grain_intensity']:.4f}")
        print(f"Luminance dependence: {aggregated_stats['avg_luminance_dependence']:.4f}")
        print(f"Grain intensity range: {aggregated_stats['grain_intensity_range'][0]:.4f} - {aggregated_stats['grain_intensity_range'][1]:.4f}")
        
        return aggregated_stats
    
    def apply_film_grain(self, image: np.ndarray, film_directory: str = None) -> np.ndarray:
        """
        Apply grain learned from film photos to a digital image.
        
        Args:
            image: Digital image in RGB format
            film_directory: Path to film samples. If None, uses previously learned stats.
            
        Returns:
            Image with learned film grain applied
        """
        if film_directory:
            self.load_and_analyze_film_samples(film_directory)
        
        if self.film_stats is None:
            print("Warning: No film statistics available. Using default grain.")
            return self.apply_grain(image)
        
        # Use learned statistics to set grain parameters
        grain_intensity = self.film_stats['avg_grain_intensity'] / 255.0
        self.intensity = np.clip(grain_intensity, 0.01, 1.0)
        
        # Apply with learned parameters
        result = self.apply_grain(image)
        
        return result