# Film Emulation Project

A computer vision project for emulating the characteristics of various film stocks in digital images.

## Project Structure

```
├── data/                    # Data storage
│   ├── film_samples/        # Reference film image samples
│   │   ├── portra/          # Kodak Portra 400 film samples
│   │   ├── velvia/          # Fujifilm Velvia film samples
│   │   └── gold/            # Kodak Gold 200 film samples
│   └── digital/             # Original digital images for processing
├── src/                     # Source code
│   ├── color_transfer.py    # Color transfer module
│   ├── tone_curves.py       # Tone curve implementation
│   ├── grain_synthesis.py   # Film grain synthesis
│   └── utils.py             # Utility functions
├── results/                 # Output results and processed images
├── requirements.txt         # Project dependencies
└── README.md               
```

## Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Project Goals

- Analyze and characterize various film stocks (Portra, Velvia, Tri-X)
- Implement color transfer techniques
- Develop accurate tone curve models
- Create realistic grain synthesis algorithms

## Development

Start with the notebooks for exploration and testing new approaches:
- `notebooks/week1_exploration.ipynb` - Initial data exploration

## Output

Processed images and results will be saved in the `results/` directory.
