# MLP Pet Adoption Prediction

University coursework: a multi-layer perceptron (MLP) trained on the PetFinder-Mini dataset to predict whether a pet will be adopted, built with TensorFlow/Keras.

## Overview

Uses the PetFinder-Mini dataset (a curated subset of the Petfinder.my competition data) containing pet profiles with features like breed, age, colour, health, and description length. The original dataset's `AdoptionSpeed` column (0 = same day, 1 = within a week, 2 = within a month, 3 = within 3 months, 4 = no adoption) is simplified into a binary target: `0` if the pet was never adopted (`AdoptionSpeed == 4`), `1` otherwise. The MLP is trained to predict this binary outcome.

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras (notebook pins `tensorflow==2.11.0`), pandas, NumPy, scikit-learn, matplotlib
- **Model format:** `.h5` (weights)
- **Environment:** Jupyter Notebook (developed on Google Colab)

## Key Concepts

- Tabular data preprocessing with Keras `feature_column` / `DenseFeatures`
- Multi-layer perceptron architecture design
- Dropout regularisation
- Binary classification with a sigmoid output layer
- Model weight serialisation (`.h5`)

## Project Structure

```
ai-mlp-pet-adoption-prediction/
├── mlp_pet_adoption.ipynb    # Training and evaluation notebook
├── mlp_model.weights.h5      # Saved model weights
└── petfinder-mini.csv        # Dataset
```

## Dataset

- **Source:** PetFinder-Mini (derived from Kaggle PetFinder.my Adoption Prediction)
- **Rows:** 5,770 pet profiles
- **Features used:** Type, Age, Breed1, Gender, Color1, Color2, MaturitySize, FurLength, Vaccinated, Sterilized, Health, Fee, PhotoAmt (the free-text `Description` column is dropped)
- **Target:** binary `target` column derived from `AdoptionSpeed` (see Overview)
- **Split:** 3,692 train / 924 validation / 1,154 test (80/20 train-test split, then 80/20 train-validation split)

## Setup & Run

```bash
pip install tensorflow==2.11.0 pandas numpy scikit-learn matplotlib jupyter
jupyter notebook mlp_pet_adoption.ipynb
```

The notebook was written for Google Colab and by default loads the dataset from Google Drive:

```python
dataframe = pd.read_csv("drive/MyDrive/lab6/petfinder-mini.csv")
```

To run locally, comment out that line and uncomment the local alternative already present in the notebook, which reads `petfinder-mini.csv` from the repo root:

```python
dataframe = pd.read_csv("petfinder-mini.csv")
```

To load the saved model weights instead of retraining:

```python
model.load_weights("mlp_model.weights.h5")
```

## Results

Measured results from the notebook's stored outputs (`mlp_pet_adoption.ipynb`):

**Model architecture** (`model.summary()`):

| Layer | Output | Params |
|---|---|---|
| dense_features (DenseFeatures) | multiple | 1,144 |
| dense_18 (Dense, relu, 128 units) | multiple | 19,328 |
| dense_19 (Dense, relu, 128 units) | multiple | 16,512 |
| dropout_6 (Dropout, rate 0.1) | multiple | 0 |
| dense_20 (Dense, sigmoid, 1 unit) | multiple | 129 |

Total params: 37,113 (all trainable)

**Training** — Adam optimizer (lr=0.001), binary cross-entropy loss, batch size 5, 30 epochs. Final epoch (30/30):

- Training loss: 0.4051, training accuracy: 0.8115
- Validation loss: 0.5818, validation accuracy: 0.7511

**Test set evaluation** (`model.evaluate(test_ds)`):

- Test loss: 0.5877
- Test accuracy: **0.7565** (75.65%)

These are the raw numbers as printed in the notebook's output cells; no metrics in this README are estimated.
