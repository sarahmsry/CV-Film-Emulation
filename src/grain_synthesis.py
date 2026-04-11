import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
import cv2

from utils import lab_to_rgb, rgb_to_lab

class GrainSynthesis:
    """
    Analyzes and applies film grain using fast fourier transform (FFT) and 
    luminance-dependent grain patterns. This class generates grain that mimics 
    the real film grain, including size, intensity, and distribution based on image luminance (brightness).
    """
    
    # initialize with default grain parameters
    def __init__(self, intensity: float = 0.1, size: float = 0.2):
        """
        Args:
            intensity: Grain intensity (default 0.1, increase for stronger grain)
            size: Grain size multiplier (default 0.2 for fine grain clumps, increase for coarser clumps)
        """
        self.intensity = intensity
        self.size = size
        self.film_stats = False  # track whether or not real film samples have been analyzed to learn grain characteristics

    def estimate_grain_size(self, grain: np.ndarray) -> float:
            """
            Estimate grain size using frequency analysis. Grain with more high-frequency 
            content is finer, while grain with more low-frequency content is coarser.
            
            Args:
                grain: Extracted grain pattern from a film image
                
            Returns:
                Estimated grain size (higher = coarser)
            """
            h, w = grain.shape
            crop_size = min(512, h, w) # crop to manageable size for FFT
            center_h, center_w = h//2, w//2 # center of the image
            half_crop = crop_size // 2

            # array slicing to get centered crop for FFT analysis
            grain_crop = grain[center_h - half_crop:center_h + half_crop, center_w - half_crop:center_w + half_crop]
            
            # Get actual dimensions of the crop (may differ slightly from crop_size due to array slicing)
            actual_h, actual_w = grain_crop.shape

            fft = fft2(grain_crop) # convert to frequency domain from space domain
            fft_shifted = fftshift(fft) # shift low frequencies to center, higher frequencies to edges
            magnitude = np.abs(fft_shifted) # get strength of each frequency

            # Create coordinate grid based on actual crop dimensions
            center_x = actual_w // 2
            center_y = actual_h // 2
            y, x = np.ogrid[:actual_h, :actual_w] # create coordinate grid for distance calculation
            radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) # calculate distance from center for each frequency

            radial_profile = []
            # for each distance (r) from the center, find all pixels at r, get their magnitudes, and compute average magnitude for that radius which represents the strength of that frequency band
            max_radius = min(center_x, center_y)
            for r in range(1, int(max_radius)):
                mask = np.isclose(radius, r, atol=0.5)  # Use isclose for fuzzy float matching instead of exact equality
                if np.any(mask):
                    radial_profile.append(np.mean(magnitude[mask]))
            
            if len(radial_profile) > 5:  # Ensure radial_profile is long enough to slice [5:]
                radial_profile = np.array(radial_profile)
                peak_freq = np.argmax(radial_profile[5:]) + 5 # find index of largest value (highest grain frequency), and ignore very low frequencies which can be affected by image content rather than grain

                # inverse relationship between frequency and grain size: higher frequency = finer grain, lower frequency = coarser grain
                # 30 is arbitrary scaling facotr; can be adjusted
                # add 1 to peak_freq to avoid division by zero in case of no clear peak
                grain_size = 30.0 / (peak_freq + 1) 
                return np.clip(grain_size, 0.3, 3.0) # clip to reasonable range for grain size
            else: 
                return 1.0 # default grain size if no clear peak found (weak/insufficient grain signal)
                
    def analyze_film_grain(self, film_samples: list):   
        """
        Analyze grain characteristics from film samples to learn parameters for synthesis.
        
        Args:
            film_samples: List of film images in RGB format
        """
        print(f"Analyzing grain from {len(film_samples)} film samples...") 
        intensities = []
        sizes = []

        for index, film_img in enumerate(film_samples):
            # convert to grayscale to analyze grain in one color channel rather than three
            gray = cv2.cvtColor(film_img, cv2.COLOR_RGB2GRAY).astype(np.float32) # allow negative values for grain extraction
            smooth = gaussian_filter(gray, sigma=5)  # smooth to remove grain, keep content (higher sigma = more smoothing)
            grain = gray - smooth  # extract grain by subtracting smoothed version

            grain_intensity = np.std(grain) # standard deviation as measure of grain intensity (low std = little grain, high std = high grain)
            intensities.append(grain_intensity)

            grain_size = self.estimate_grain_size(grain) # estimate grain size using frequency analysis
            sizes.append(grain_size)

            # progress report for every 10 processed film samples 
            if index % 10 == 0: 
                print(f"Analyzed sample {index+1}/{len(film_samples)} - Intensity: {grain_intensity:.4f}, Size: {grain_size:.4f}")

        self.intensity = np.mean(intensities) / 255.0  # average all intensity measurements and normalize 0-1 (easier to work with)
        self.size = np.mean(sizes) # average all size measurements
        self.film_stats = True # flag to indicate real film samples have been learned from

        print(f"Grain analysis complete - Average Intensity: {self.intensity:.4f}, Average Size: {self.size:.4f}")
        
            
    def generate_grain(self, shape: tuple, luminance: np.ndarray = None) -> np.ndarray:
        """
        Generate film grain pattern

        Args:
            shape: Output shape (height, width)
            luminance: Optional luminance map for luminance-dependent grain

        Returns:
            Array of grain values 
        """

        h, w = shape[:2]
        grain = np.random.normal(0,1,(h,w)) # generate base noise with normal distribution (mean=0, std=1)
        sigma = self.size # use learned grain size to set blur amount (higher sigma = more blurring = coarser grain and vice versa)
        grain = gaussian_filter(grain, sigma=sigma) # blur noise to create grain "clumps" (higher sigma = more blurring = coarser grain)
        grain = grain - grain.mean() # center at zero (mean can be slightly off due to randomness, so subtract it to ensure grain is zero-centered)
        grain = grain / grain.std() + 1e-8 # normalize to unit standard deviation (add small epsilon to avoid division by zero)

        if luminance is not None:
            lum_normalized = luminance / 255.0 # normalize luminance (brightness) to 0-1 (1 white, 0 black)
            grain_mask = 1.0 - lum_normalized * 0.7 # create mask to make grain stronger in darker areas (shadows) and weaker in brighter areas (highlights) (mask = 1 = full grain, mask = 0.3 = reduced grain)
            grain = grain * grain_mask # apply mask to grain to create luminance-dependent grain pattern
        return grain

    def apply_grain(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Apply grain to an image

        Input:
            Image: Input digital image in RGB format
            Strength: How strongly to apply the grain (0-2); 2 = double grain, 1 = normal grain (learned intensity), 0 = no grain

        Returns:
            Original image with grain applied in RGB format 
        """
        if not self.film_stats:
            print("Warning: No film grain characteristics learned, so using default grain values.")

        if strength == 0:
            return image # if strength is 0, return original image without grain
        
        lab = rgb_to_lab(image) # convert to LAB color space to apply grain to luminance channel only
        lab_float = lab.astype(np.float32) # convert to float for grain application (allow negative values)
        luminance = lab[:,:,0] # extract luminance channel to apply grain

        for c in range(3): # apply grain to each channel, but modulate by luminance to create more realistic grain pattern that is stronger in shadows and weaker in highlights
            grain = self.generate_grain(image.shape[:2], luminance) # generate grain pattern based on learned characteristics and image luminance
            lab_float[:,:,c] += grain * self.intensity * strength * 255.0 * 0.5 # apply grain to each channel, scaled by learned intensity and user-specified strength (multiply by 255 to scale back to pixel value range)
        
        result = lab_to_rgb(lab_float.astype(np.uint8)) # convert back to RGB color space
        return result

    def save_stats(self, filepath: str):
        '''save learned grain stats to avoid re-fitting; can load later to apply grain synthesis'''
        np.savez(filepath, intensity=self.intensity, size=self.size, film_stats=self.film_stats)
        print(f"Saved grain stats to {filepath}")

    def load_stats(self, filepath: str):
        '''load previously saved grain stats to apply grain synthesis without refitting'''
        data = np.load(filepath, allow_pickle=True)
        self.intensity = float(data['intensity'])
        self.size = float(data['size'])
        self.film_stats = bool(data['film_stats']) 
        print(f"Loaded grain stats from {filepath}")