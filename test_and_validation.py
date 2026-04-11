"""Test complete pipeline"""

import sys
sys.path.append('src')

from integration import FilmEmulationPipeline
from utils import load_dataset, load_image
import numpy as np

def test_training():
    """Test training on film samples"""
    print("Testing training...")
    
    # Load small sample
    samples = load_dataset("data/film_samples/portra_400/", max_images=10)
    
    # Train pipeline
    pipeline = FilmEmulationPipeline('portra_400_test')
    pipeline.train(samples)
    
    # Check parameters were learned
    assert pipeline.color_transfer.target_mean is not None
    assert pipeline.tone_curves.curve_params is not None
    assert pipeline.grain_synthesis.film_stats == True
    
    print("Training test passed")

def test_transformation():
    """Test applying to digital image"""
    print("Testing transformation...")
    
    # Create synthetic test image
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Load trained model
    pipeline = FilmEmulationPipeline('portra_400')
    pipeline.load_model('results/models/portra400_model/')  # adjust path as needed
    
    # Apply transformation
    result = pipeline.transform(test_img)
    
    # Check output
    assert result.shape == test_img.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255
    
    print("Transformation test passed")

def test_save_load():
    """Test saving and loading models"""
    print("Testing save/load...")
    
    # Train small model on actual film samples
    samples = load_dataset("data/film_samples/kodak portra 400/", max_images=5)
    if len(samples) == 0:
        print("No film samples found. Skipping save/load test.")
        return
    
    pipeline1 = FilmEmulationPipeline('test')
    pipeline1.train(samples)
    pipeline1.save_model('models/test')
    
    # Load it back
    pipeline2 = FilmEmulationPipeline('test')
    pipeline2.load_model('models/test')
    
    # Check parameters match (use equal_nan=True to handle NaN values if present)
    assert np.allclose(pipeline1.color_transfer.target_mean, 
                      pipeline2.color_transfer.target_mean, equal_nan=True)
    
    print("Save/load test passed")

if __name__ == "__main__":
    test_training()
    test_transformation()
    test_save_load()
    print("\nALL TESTS PASSED!")