# MLP Pet Adoption Prediction

Multi-layer perceptron (MLP) trained on the PetFinder-Mini dataset to predict pet adoption speed. Classifies whether a pet will be adopted quickly, slowly, or not at all based on profile attributes.

## Overview

Uses the PetFinder-Mini dataset (a curated subset of the Petfinder.my competition data) containing pet profiles with features like breed, age, colour, health, and description length. An MLP classifier predicts the adoption outcome class.

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras, pandas, NumPy, scikit-learn, matplotlib
- **Model format:** `.h5` (weights)
- **Environment:** Jupyter Notebook

## Key Concepts

- Tabular data preprocessing (encoding, scaling)
- Multi-layer perceptron architecture design
- Dropout regularisation
- Multi-class classification with softmax
- Model weight serialisation (`.h5`)

## Project Structure

```
ai-mlp-pet-adoption-prediction/
├── mlp_pet_adoption.ipynb          # Training and evaluation notebook
├── mlp_model.weights.h5      # Saved model weights
└── petfinder-mini.csv        # Dataset
```

## How to Run

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib jupyter
jupyter notebook mlp_pet_adoption.ipynb
```

To load saved weights:

```python
model.load_weights("mlp_model.weights.h5")
```

## Dataset

- **Source:** PetFinder-Mini (derived from Kaggle PetFinder.my Adoption Prediction)
- **Features:** Type, Age, Breed, Gender, Color, Maturity Size, Fur Length, Health, Fee, PhotoAmt, etc.
- **Target:** AdoptionSpeed (0 = same day, 1 = within a week, 2 = within a month, 3 = within 3 months, 4 = no adoption)

## Environment

Developed and tested with:

- Python 3.9+
- Jupyter Notebook / JupyterLab

Install dependencies:

```bash
pip install -r requirements.txt      # if provided
# or manually: pip install numpy pandas matplotlib scikit-learn torch torchvision
```

Open notebooks in order — each notebook builds on outputs from the previous one.
