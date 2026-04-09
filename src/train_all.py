import sys 
import numpy as np
from utils import load_dataset
from integration import FilmEmulationPipeline

def main():
    print("Training ALL models on film samples...")

    print("Training Kodak Portra 400 Samples...")
    portra_samples = load_dataset("data/film_samples/kodak portra 400/")
    portra_pipeline = FilmEmulationPipeline("Kodak Portra 400")
    portra_pipeline.train(portra_samples)
    portra_pipeline.save_model("results/models/portra400_model/")

    print("Training Kodak Gold 200 Samples...")
    gold_samples = load_dataset("data/film_samples/kodak gold 200/")  
    gold_pipeline = FilmEmulationPipeline("Kodak Gold 200")
    gold_pipeline.train(gold_samples)
    gold_pipeline.save_model("results/models/gold200_model/")

    print("Training Fuji Velvia 50 Samples...")
    velvia_samples = load_dataset("data/film_samples/fuji velvia 50/")
    velvia_pipeline = FilmEmulationPipeline("Fuji Velvia 50")
    velvia_pipeline.train(velvia_samples)
    velvia_pipeline.save_model("results/models/velvia50_model/")

    print("COMPLETE: All models trained and saved!")      

if __name__ == "__main__":
    main()  