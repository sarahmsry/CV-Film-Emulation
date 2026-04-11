import sys
sys.path.append('src')

from integration import FilmEmulationPipeline
from utils import load_image, load_dataset
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load trained model
    print("1. Loading Portra 400 model...")
    pipeline = FilmEmulationPipeline('portra_400')
    pipeline.load_model('results/models/portra400_model/')  # adjust path as needed
    
    # Load test image
    print("2. Loading test image...")
    digital_img = load_image("data/digital_samples/digital8.jpg")
    
    # Apply full emulation
    print("3. Applying film emulation...")
    result = pipeline.transform(digital_img)
    
    # Create comparison visualization
    print("4. Creating visualization for before and after comparison...\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Add spacing between subplots
    
    # Original
    axes[0, 0].imshow(digital_img)
    axes[0, 0].set_title('Original Digital', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Color only
    color_only = pipeline.color_transfer.transform(digital_img)
    axes[0, 1].imshow(color_only)
    axes[0, 1].set_title('+ Color Transfer', fontsize=11)
    axes[0, 1].axis('off')
    
    # Color + Curves
    color_curve = pipeline.tone_curves.analyze_curve(color_only)
    axes[0, 2].imshow(color_curve)
    axes[0, 2].set_title('+ Tone Curves', fontsize=11)
    axes[0, 2].axis('off')
    
    # Color + Curves + Grain (full)
    axes[1, 0].imshow(result)
    axes[1, 0].set_title('+ Grain (FINAL)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Show parameters
    params_text = f"Portra 400 Parameters:\n\n"
    params_text += f"Color: LAB mean={np.round(pipeline.color_transfer.target_mean, 2)}\n"
    params_text += f"Tone: {pipeline.tone_curves.curve_params['type']}\n"
    params_text += f"Grain: intensity={pipeline.grain_synthesis.intensity:.3f}\n"
    params_text += f"         size={pipeline.grain_synthesis.size:.2f}"
    
    axes[1, 1].text(0.05, 0.5, params_text, fontsize=9, family='monospace', 
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    # final comparison
    axes[1, 2].imshow(np.hstack([digital_img, result]))
    axes[1, 2].set_title('Before | After', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Film Emulation Pipeline - Kodak Portra 400', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('results/demo_output.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Demo complete; saved to results/demo_output.jpg")

if __name__ == "__main__":
    main()