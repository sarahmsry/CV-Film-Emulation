# Film Emulation Project

A computer vision project for emulating the characteristics of various iconic film stocks in digital images using classical computer vision techniques. This project analyzes film samples and applies their color, tone, and grain characteristics to digital images.

## Project Goals

- Analyze various film stocks (Kodak Portra 400, Kodak Gold 200, Fujifilm Velvia 50) 
- Implement color transfer technique using Reinhard algorithm to match film color distributions
- Develop accurate tone curve model for film contrast characteristics
- Create realistic grain synthesis algorithm
- Apply all characteristics to any digital image

## System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: Minimum 4GB (8GB+ recommended for smooth processing)
- **Storage**: Minimum 2GB for dependencies, models, and sample data
- **Processor**: Any modern CPU (Intel i5+, AMD Ryzen 5+, or equivalent)

### Tested Configurations
- **Windows 10/11** with Python 3.11 (fully tested)
- **macOS** with Python 3.9+ 
- **Linux** (Ubuntu 20.04+) with Python 3.8+

### Software Dependencies
All dependencies are automatically installed via `pip install -r requirements.txt`:
- **numpy** (>=1.20.0) - Numerical arrays and linear algebra
- **opencv-python** (>=4.5.0) - Image processing and analysis
- **scikit-image** (>=0.18.0) - Advanced image algorithms
- **scipy** (>=1.5.0) - Scientific computing and optimization
- **pillow** (>=8.0.0) - Image I/O and manipulation
- **matplotlib** (>=3.3.0) - Visualization and plotting


## Installation

### 1. Clone or set up the project directory

```bash
cd CV-Film-Emulation
```

### 2. Create a Python virtual environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
CV-Film-Emulation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── apply_emulation.py        # Apply trained models to images (CLI tool)
├── demo.py                   # Demo/visualization script
├── train_all.py              # Training script for all film stocks
├── test_and_validation.py    # Unit tests for the pipeline
├── evaluate.py               # Evaluation utilities
├── visualize_tone_curves.py  # Tone curve visualization tool
├── src/                      # Source code directory
│   ├── utils.py             # Utility functions (image I/O, color conversions)
│   ├── color_transfer.py    # Color transfer module
│   ├── tone_curves.py       # Tone curve analysis and application
│   ├── grain_synthesis.py   # Film grain synthesis
│   ├── integration.py       # Main pipeline integration
│   └── test.py              # Additional testing utilities
├── data/                     # Data directory
│   ├── digital_samples/     # Digital test images
│   └── film_samples/        # Film reference images
│       ├── fuji velvia 50/
│       ├── kodak gold 200/
│       └── kodak portra 400/
├── results/                  # Output directory
│   ├── models/              # Saved trained model parameters (created during training)
│   │   ├── portra400_model/
│   │   ├── gold200_model/
│   │   └── velvia50_model/
│   └── *.jpg                # Processed output images
└── .git/                    # Git repository
```

## Quick Start

### Running the Demo

The demo script provides a complete visualization of the film emulation pipeline with before/after comparisons and learned parameters:

```bash
# From the project root directory
python demo.py
```

**What the demo does:**
- Loads a pre-trained Portra 400 model from `results/models/portra400_model/`
- Loads a test digital image
- Applies the complete film emulation pipeline
- Displays a 2×3 comparison grid showing:
  1. Original digital image
  2. After color transfer
  3. After tone curve application
  4. Final result with all effects (color + curves + grain)
  5. Learned film parameters (color stats, tone curve, grain characteristics)
  6. Side-by-side before/after comparison
- Saves the output to `results/demo_output.jpg` 

### Training Models (Optional - Pre-trained models included)

To retrain models on film samples:

```bash
python train_all.py
```

This trains all three film stock models and saves them to `results/models/`:
- `portra400_model/` - Kodak Portra 400
- `gold200_model/` - Kodak Gold 200  
- `velvia50_model/` - Fuji Velvia 50

Each model directory contains:
- `color_stats.npz` - Learned color transfer parameters
- `tone_curve_params.npz` - Learned tone curve coefficients
- `grain_synthesis_params.npz` - Learned grain characteristics

### Applying Film Emulation with Custom Parameters

Use the CLI tool to apply film emulation to any digital image with custom effect strengths:

```bash
python apply_emulation.py --film MODEL_NAME --input INPUT_PATH --output OUTPUT_PATH [OPTIONS]
```

#### Examples

**Basic usage (default parameters):**
```bash
python apply_emulation.py --film "portra400_model" --input "data/digital_samples/your_image.jpg"
```
Output will be saved to `results/output.jpg`

**With custom output path:**
```bash
python apply_emulation.py --film "gold200_model" --input "data/digital_samples/test.jpg" --output "results/my_result.jpg"
```

**Fine-tuned effects (50% strength on each effect):**
```bash
python apply_emulation.py --film "velvia50_model" --input "data/digital_samples/photo.jpg" --output "results/subtle.jpg" --color 0.5 --curve 0.5 --grain 0.5
```

**Extra strong grain (Portra 400 with double grain, full color/curve):**
```bash
python apply_emulation.py --film "portra400_model" --input "data/digital_samples/photo.jpg" --output "results/grainy.jpg" --grain 2.0 --color 1.0 --curve 1.0
```

**No grain, full color and tone curves:**
```bash
python apply_emulation.py --film "gold200_model" --input "data/digital_samples/photo.jpg" --output "results/clean.jpg" --grain 0.0 --color 1.0 --curve 1.0
```

#### CLI Parameters

**Required:**
- `--film`: Film model to use. Options:
  - `portra400_model` (Kodak Portra 400 - warm, saturated)
  - `gold200_model` (Kodak Gold 200 - bright, vibrant)
  - `velvia50_model` (Fuji Velvia 50 - cool, contrasty)

- `--input`: Path to input digital image (JPEG or PNG)

**Optional:**
- `--output`: Path where output will be saved (default: `results/output.jpg`)
- `--color`: Color transfer strength 
  - Range: 0.0 to 1.0
  - 0.0 = no color change
  - 1.0 = full film color characteristics (default)
  
- `--curve`: Tone curve strength
  - Range: 0.0 to 1.0
  - 0.0 = no contrast adjustment
  - 1.0 = full film tone curve applied (default)
  
- `--grain`: Film grain strength
  - Range: 0.0 to 2.0
  - 0.0 = no grain
  - 1.0 = normal learned grain intensity (default)
  - 2.0 = double grain intensity

## Module Documentation

### Color Transfer (`color_transfer.py`)
Transfers color characteristics from film samples to digital images. Based on Reinhard et al. "Color Transfer between Images" (2001).
- Converts images to LAB color space
- Matches color mean and standard deviation
- Preserves natural color relationships

### Tone Curves (`tone_curves.py`)
Models and applies the unique contrast characteristics of film stocks.
- Analyzes luminance distribution of film samples
- Fits sigmoid curves to tone characteristics
- Applies S-curve for increased contrast (typical of film)

### Grain Synthesis (`grain_synthesis.py`)
Synthesizes realistic film grain patterns.
- Analyzes grain structure from film samples
- Generates procedural grain matching film characteristics
- Blends grain with image while preserving details

### Integration Pipeline (`integration.py`)
Combines all three modules into a single unified pipeline for complete film emulation.

## Data Format

### Required Image Format
- **Format**: JPEG, PNG, or other common formats
- **Color Space**: RGB (converted from BGR if needed)
- **Dimensions**: Recommended minimum 800x600 pixels
- **Range**: Pixel values 0-255 (8-bit)

### Film Sample Organization
Film samples should be organized as follows:
```
data/film_samples/
└── [film_stock_name]/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg  
```

## Troubleshooting

### Incorrect or Weak Results

**Issue: Film emulation effect is barely visible**
- Increase effect strengths: `--color 1.0 --curve 1.0 --grain 1.0`
- Try a different film model (some may be more subtle than others)
- Ensure digital image has sufficient color variation and detail

**Issue: Result looks over-processed or unnatural**
- Reduce effect strengths: `--color 0.7 --curve 0.7 --grain 0.5`
- Disable individual effects where needed:
  ```bash
  python apply_emulation.py --film "portra400_model" --input "image.jpg" --grain 0.0  # Color + tone curves only
  ```
- Try different film model that may match your image better

## Output

Processed images and model results are saved in the `results/` directory. The directory is created automatically if it doesn't exist.

## Testing

### Running Automated Tests

A comprehensive test suite is provided to validate the pipeline functionality:

```bash
python test_and_validation.py
```

**Tests included:**

1. **Training Test** - Validates that the pipeline can train on film samples
   - Loads 10 film samples from `data/film_samples/portra_400/`
   - Trains all three modules (color transfer, tone curves, grain synthesis)
   - Verifies learned parameters are not None

2. **Transformation Test** - Validates that trained models can process images
   - Creates a synthetic test image
   - Loads pre-trained Portra 400 model
   - Applies film emulation transformation
   - Verifies output shape, data type, and value range (0-255)

3. **Save/Load Test** - Validates model persistence
   - Trains a model on 5 samples from `data/film_samples/kodak portra 400/`
   - Saves trained parameters to disk
   - Loads parameters back
   - Verifies loaded and saved parameters match

**Expected Output:**
```
Testing training...
Training test passed
Testing transformation...
Transformation test passed
Testing save/load...
Save/load test passed

✓ ALL TESTS PASSED!
```

### What Tests Verify

- ✓ Pipeline can successfully train on real film samples
- ✓ Color transfer statistics are learned correctly
- ✓ Tone curve parameters are extracted
- ✓ Grain characteristics are analyzed
- ✓ Film emulation produces valid output images
- ✓ Learned model parameters can be saved and loaded without loss of data

## Presentation Guide

### For Live Demonstrations

**Best Practices:**
1. **Pre-run the demo** before your presentation to ensure all models are trained and files are present
2. **Run demo.py** to show the complete pipeline with visualization:
   ```bash
   python demo.py
   ```
   This generates a professional-looking comparison visualization in `results/demo_output.jpg`

3. **Highlight the key steps** shown in the demo output:
   - Original digital photo
   - Color transfer from film
   - Tone curve adjustments
   - Grain synthesis
   - Final result

4. **Show learned parameters** from the demo output to demonstrate how the system analyzes film samples

5. **Live CLI demo** (optional):
   ```bash
   python apply_emulation.py --film "gold200_model" --input "data/digital_samples/your_image.jpg" --output "results/demo_live.jpg" --color 1.0 --curve 1.0 --grain 0.5
   ```

### Pre-Presentation Checklist

- [ ] Run tests to verify everything works: `python test_and_validation.py` (should see "✓ ALL TESTS PASSED!")
- [ ] Verify all dependencies installed: `pip list | grep -E "numpy|opencv|scipy|pillow|matplotlib"`
- [ ] Test `python demo.py` runs successfully
- [ ] Check that `results/models/` contains all three trained models
- [ ] Have sample images ready in `data/digital_samples/`
- [ ] Pre-generate some example outputs to have as backup visuals

## Output

Processed images and model results are saved in the `results/` directory. The directory is created automatically if it doesn't exist.

## Future Enhancements

- [ ] Support for additional film stocks
- [ ] Batch processing for multiple images
- [ ] Web-based interface


