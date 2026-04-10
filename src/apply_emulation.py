'''cmd line interface for user to apply film emulation to any image'''

import sys 
sys.path.append('src')
import argparse
from utils import load_image, save_image
from integration import FilmEmulationPipeline

def main():
    parser = argparse.ArgumentParser(description="Apply film emulation to a digital image using a trained model.")
    parser.add_argument("--input", type=str, help="Path to the input digital image")
    parser.add_argument("--output", type=str, help="Path to save the output image with film emulation applied")
    parser.add_argument("--film", type=str, default="portra400_model", help="Type of film stock to emulate (e.g., 'portra400_model', 'gold200_model', 'velvia50_model')")
    parser.add_argument("--color", type=float, default=1.0, help="Color adjustment factor (0-1); 1 = full color transfer, 0 = no color change")
    parser.add_argument('--curve', type=float, default=1.0, help='Curve strength (0-1)')
    parser.add_argument('--grain', type=float, default=1.0, help='Grain strength (0-2); 2 = double grain, 1 = normal grain, 0 = no grain')

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.film} model...")
    pipeline = FilmEmulationPipeline(args.film)
    pipeline.load_model(f'results/models/{args.film}')

    # Load and process image
    print(f"Processing {args.input}...")
    img = load_image(args.input)

    # Apply film emulation with specified strength parameters for color, curve, and grain adjustments from user input
    result = pipeline.transform(img, color_strength=args.color, curve_strength=args.curve, grain_strength=args.grain)
    save_image(result, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()