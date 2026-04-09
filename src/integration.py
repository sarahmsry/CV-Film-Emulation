"""
Film emulation pipeline - integrates all modules
"""
import numpy as np
from typing import List
import os
from pathlib import Path

from color_transfer import ColorTransfer
from tone_curves import ToneCurves
from grain_synthesis import GrainSynthesis
from utils import load_image, save_image, load_dataset


class FilmEmulationPipeline:
    def __init__(self, film_name: str):
        '''
        Input: 
            film_stock_name: name of the film stock to emulate (in this case, portra 400, gold 200, or velvia 50)
        '''
        self.film_name = film_name
        self.color_transfer = ColorTransfer()
        self.tone_curves = ToneCurves()
        self.grain_synthesis = GrainSynthesis()
        self.is_trained = False

    def train(self, film_samples: List[np.ndarray]):
        '''
        Train all modules on film samples by calling the fit/analyze functions in each module to learn
        the film stock's specific characteristics (color distribution, tone curve parameters, grain structure) 

        Input:
            film_samples: list of film stock sample images in RGB format 

        Output:
            None (stores learned parameters in class variables)
        '''
        print(f'training {self.film_name} emulation pipeline...')
        print(f'processing {len(film_samples)} film samples...')

        # Step 1: Train color transfer 
        print('Step 1/3: Learning color characteristics...')
        self.color_transfer.fit(film_samples)
        print('Color transfer trained\n')

        # Step 2: Extract tone curves
        print('Step 2/3: Extracting tone curves...')
        self.tone_curves.analyze_film_contrast(film_samples)
        print('Tone curves extracted\n')

        # Step 3: Analyze grain
        print('Step 3/3: Analyzing film grain structure...')
        self.grain_synthesis.analyze_film_grain(film_samples)

        self.is_trained = True 
        print(f'Training complete for {self.film_name}!\n')


    def transform(self, image: np.ndarray, color_strength: float = 1.0, curve_strength: float = 1.0, grain_strength: float = 1.0) -> np.ndarray:
        '''
        Apple film emulation to a digital image by applying the transformations from each module (color transfer, 
        tone curves, grain synthesis). Strength parameters allow for intensity adjustments

        Input:
            image: original digital image in RGB format 
            color_strength: intensity of color transfer (0-1)
            curve_strength: intensity of tone curve application (0-1)
            grain_strength: intensity of grain application (0-1)

        Output:
            Film-emulated image in RGB format
        '''

        if not self.is_trained:
            raise ValueError("Pipeline not trained; must call train() first.") 
        
        result = image.copy() # copy original image to apply tranformations 

        # Step 1: color transfer
        if color_strength > 0:
            result = self.color_transfer.transform(result, strength=color_strength)

        # Step 2: tone curves
        if curve_strength > 0:
            result = self.tone_curves.analyze_curve(result, strength=curve_strength)
        
        # Step 3: grain synthesis
        if grain_strength > 0:
            result = self.grain_synthesis.apply_grain(result, strength=grain_strength)

        return result


    def save_model(self, directory: str):   
        '''
        Save trained pipeline parameters to a directory for later use. Saves color transfer stats, 
        tone curve parameters, and grain synthesis parameters.

        Input:
            directory: directory to save model files (will be created if it doesn't exist)

        Output:
            None (saves model files to disk)
        '''
        if not self.is_trained:
            raise ValueError("Cannot save untrained pipeline")
        
        os.makedirs(directory, exist_ok=True) # create directory if it doesn't exist

        # Save each component's parameters using save methods in each module; .npz files can be reloaded to prevent retraining 
        self.color_transfer.save_stats(f"{directory}/color_stats.npz")
        self.tone_curves.save_stats(f"{directory}/tone_curve_params.npz")
        self.grain_synthesis.save_stats(f"{directory}/grain_synthesis_params.npz")

        print(f"Saved {self.film_name} model to {directory}")


    def load_model(self, directory: str):
        '''
        Load trained pipeline parameters from a directory to apply transformations without retraining.
        
        Input: 
            directory: directory to load .npz model files from 

        Output:
            None (updates self.is_trained to True)
        '''

        self.color_transfer.load_stats(f"{directory}/color_stats.npz")
        self.tone_curves.load_stats(f"{directory}/tone_curve_params.npz")
        self.grain_synthesis.load_stats(f"{directory}/grain_synthesis_params.npz")

        print(f"Loaded {self.film_name} model from {directory}")
   