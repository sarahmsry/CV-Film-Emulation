import sys
sys.path.append('src')

from utils import load_image, save_image, show_comparison, load_dataset
from tone_curves import ToneCurves

def main():
    tone_curves = ToneCurves()
    film_data = load_dataset("data/film_samples/kodak gold 200/") # load film samples to fit model
    tone_curves.analyze_film_contrast(film_data) # fit model to learn tone curve parameters from film samples
    digital_image = load_image("data/digital_samples/digital8.jpg") # load digital image to apply transfer 
    result = tone_curves.analyze_curve(digital_image, strength=1) # apply tone curve
    tone_curves.visualization(film_data) # visualize learned curve parameters
    show_comparison(digital_image, result, "Original Digital", "With Tone Curve")
    print(tone_curves.curve_params)

if __name__ == "__main__":
    main()