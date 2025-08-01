# Flood Risk Prediction System 🌊

A machine learning project that predicts flood risk levels based on environmental and geographical features using a custom neural network implemented in PyTorch.

## Overview

The Flood Risk Prediction System uses a custom neural network (FloodNet) trained on environmental data to classify flood risk into predefined categories (Low, Medium, High). The system processes numerical, categorical, and text-based features and includes data preprocessing, model training, evaluation, and an interactive command-line interface for real-time predictions.

Key capabilities:
- Handles imbalanced data using weighted random sampling
- Employs early stopping to prevent overfitting
- Provides detailed validation and testing metrics (F1 score)
- Interactive interface with confidence scores and probability breakdown

## Features

- **Data Preprocessing**: Handles numerical (Elevation, Rainfall), ordinal (Vegetation, Urbanization), one-hot encoded (Soil, Wetlands), and text (ProximityToWaterBody) features
- **Outlier Handling**: Clips and log-transforms numerical features to manage outliers
- **Class Imbalance**: Uses WeightedRandomSampler to address imbalanced flood risk classes
- **Model Architecture**: Custom PyTorch neural network with embedding layers for categorical features
- **Evaluation**: Uses weighted F1 score for validation and testing with confusion matrix visualization
- **Interactive Predictions**: Real-time flood risk predictions with confidence scores
- **Input Validation**: Ensures user inputs match valid categories with suggestions for close matches

## Prerequisites

### Software Requirements
- **Python**: Version 3.8 or higher
- **Hardware**: GPU recommended for faster training (CUDA-compatible), but CPU is supported

### Required Libraries
```bash
pip install numpy pandas torch scikit-learn matplotlib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses a custom dataset (`flood_risk_dataset_final.csv`) with the following structure:

### Features
- **Numerical Features**: Elevation, Rainfall
- **Ordinal Features**: Vegetation, Urbanization, Drainage, Slope, StormFrequency, Deforestation, Infrastructure, Encroachment, Season
- **One-Hot Encoded Features**: Soil, Wetlands
- **Text Feature**: ProximityToWaterBody
- **Target**: FloodRisk (categorical: Low, Medium, High)

## Installation & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script**:
   ```bash
   python cmodel_1.py
   ```

## How It Works

The script performs the following steps:

1. **Loads and Preprocesses Data**: Reads the dataset, handles outliers, and encodes features
2. **Splits Data**: Divides dataset into training (80%), validation (10%), and test (10%) sets with stratified sampling
3. **Trains the Model**: Trains FloodNet for up to 40 epochs with early stopping (patience=6) based on validation F1 score
4. **Evaluates the Model**: Computes weighted F1 score on test set and displays confusion matrix
5. **Interactive Prediction**: Enters loop for real-time flood risk predictions

The best model weights are automatically saved as `best_floodnet_model.pth`.

## Interactive Example

```
🌊 FLOOD RISK PREDICTION SYSTEM 🌊
📋 Valid options for each feature:
Vegetation: Dense, Sparse, Moderate, Missing
Urbanization: Low, Medium, High, Missing
…

Enter environmental details to predict flood risk:
(Type 'help' to see valid options for any field)

Vegetation: Dense
Urbanization: Low
…
Rainfall (numeric): 50
Elevation (numeric): 100

🌊 Predicted Flood Risk Level: Low
📊 Confidence: 92.3%
📈 Risk Breakdown:
  Low: 92.3%
  Medium: 6.5%
  High: 1.2%

Would you like to predict again? (Yes/No): No
Goodbye! 👋
```

### Tips for Interactive Mode
- Type `help` during input prompts to see valid options for categorical features
- Press `Ctrl+C` or answer "No" to the "predict again" prompt to exit

## Model Architecture

The FloodNet model processes features as follows:

### Feature Processing
- **Text Feature**: Embedded into an 8-dimensional vector
- **Ordinal Features**: Each embedded into a 4-dimensional vector
- **One-Hot Features**: Passed directly as binary vectors
- **Numerical Features**: Scaled and concatenated with other features

### Network Structure
- **Input layer**: Size depends on the number of features
- **Hidden layers**: 
  - 128 units (ReLU activation, 30% dropout)
  - 64 units (ReLU activation, 10% dropout)
- **Output layer**: Number of flood risk classes

## Model Performance

The model uses weighted F1 score for evaluation to handle class imbalance effectively. Performance metrics include:
- Validation and test F1 scores
- Confusion matrix visualization
- Early stopping based on validation performance

## File Structure

```
flood-risk-prediction/
├── cmodel_1.py              # Main script
├── flood_risk_dataset_final.csv  # Dataset
├── requirements.txt         # Dependencies
├── best_floodnet_model.pth  # Saved model weights (generated)
└── README.md               # This file
```

## Contributing

Feel free to fork this project and submit pull requests for improvements.

## Author

**Mobolaji Opeyemi Bolatito**  
📧 Contact: opeblow2021@gmail.com

---

*This project demonstrates the application of deep learning techniques for environmental risk assessment and disaster preparedness.*
