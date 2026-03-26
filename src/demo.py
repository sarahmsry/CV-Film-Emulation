### Demo script for testing on a sample digital image

import sys
sys.path.append('src')

from utils import load_image, save_image, show_comparison
from color_transfer import ColorTransfer
from grain_synthesis import GrainSynthesis
import matplotlib.pyplot as plt

def main():
    # print("=== Film Emulation Test ===\n")
    
    # print("Loading test image...")
    # try:
    #     img = load_image("data/digital/digitaltest1.jpg")
    # except:
    #     print("No test image found.")
    #     return
    
    # print(f"Image shape: {img.shape}\n")
    
    # # Apply grain
    # print("Applying film grain...")
    # grain_synth = GrainSynthesis(intensity=0.1, size=1)
    # img_with_grain = grain_synth.apply_grain(img)
    
    # # Show results
    # show_comparison(img, img_with_grain, "Original", "With Film Grain")
    
    # print("\n Demo complete!")

    # Create instance
    grain_synth = GrainSynthesis()

    print("=== Film Grain Analysis Demo ===\n")
    
    # Analyze your film samples (e.g., from portra folder)
    print("Step 1: Analyze film samples")
    grain_synth.load_and_analyze_film_samples("data/film_samples/portra")
    
    # Load a digital image to apply learned grain to
    print("\nStep 2: Loading digital image")
    try:
        digital_image = load_image("data/digital/digitaltest1.jpg")
        print(f"Image loaded: {digital_image.shape}")
    except:
        print("No test image found. Creating synthetic test image...")
        import numpy as np
        digital_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Apply the learned grain to the digital image
    print("\nStep 3: Applying learned film grain...")
    result = grain_synth.apply_film_grain(digital_image)
    
    # Show comparison
    show_comparison(digital_image, result, "Original Digital", "With Film Grain")
    
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()