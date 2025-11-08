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
   pip install -r requirement.txt
   ```

2. **Run the script**:
   ```bash
   python main.py
   ```

## How It Works

The script performs the following steps:

1. **Loads and Preprocesses Data**: Reads the dataset, handles outliers, and encodes features
2. **Splits Data**: Divides dataset into training (80%), validation (10%), and test (10%) sets with stratified sampling
3. **Trains the Model**: Trains FloodNet for up to 40 epochs with early stopping (patience=6) based on validation F1 score
4. **Evaluates the Model**: Computes weighted F1 score on test set and displays confusion matrix
5. **Interactive Prediction**: Enters loop for real-time flood risk predictions

The best model weights are automatically saved as `best_floodnet_model.pth`.
Please select an option:
1. Start Model Training (via main.py)
2. Start New Prediction (Interactive)
3. Exit Application

Enter your choice (1, 2, or 3): 1 

[SYSTEM MESSAGE] Starting training pipeline...

Epoch 1/40
Batch [ 10/100] Loss: 0.691
Batch [ 20/100] Loss: 0.685
...
Epoch 40/40
Training Complete.
Model checkpoint saved to: best_floodnet_model.pth

--- 🌊 FLOOD RISK PREDICTION SYSTEM 🌊 ---
... (Menu repeats, user selects 3 to)


3. Scenario 2: Option 2 (Make Interactive Prediction)
This is the flow that requires feature input.

--- 🌊 FLOOD RISK PREDICTION SYSTEM 🌊 ---

Please select an option:
1. Start Model Training (via main.py)
2. Start New Prediction (Interactive)
3. Exit Application

Enter your choice (1, 2, or 3): 2

[SYSTEM MESSAGE] Loading best_floodnet_model.pth... ✅ Model ready.

✅ Valid options for each feature:
   Vegetation: Dense, Sparse, Moderate, Missing
   Urbanization: Low, Medium, High, Missing

Enter environmental details to predict flood risk:
(Type 'help' to see valid options for any field)

Vegetation: Dense
Urbanization: Low
---
Rainfall (numeric): 50
Elevation (numeric): 100

⏳ Processing input and calculating risk...
---
✅ Predicted Flood Risk Level: Low
   Confidence: 92.3%
   Risk Breakdown: [Shows feature contribution to the prediction]

--- 🌊 FLOOD RISK PREDICTION SYSTEM 🌊 ---
... (Menu repeats, user selects 3 to exit)
4. Scenario 3: Option 3 (Exit)
This is the simple exit flow.

--- 🌊 FLOOD RISK PREDICTION SYSTEM 🌊 ---

Please select an option:
1. Start Model Training (via main.py)
2. Start New Prediction (Interactive)
3. Exit Application

Enter your choice (1, 2, or 3): 3

[SYSTEM MESSAGE] Exiting application. Goodbye!

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

```neural_network/
├── data/                             # Stores your input CSV data
├── src/
│   ├── data_preprocessing.py         # Data cleaning and feature encoding
│   ├── dataset.py                    # Defines PyTorch Dataset and DataLoader
│   ├── model.py                      # Contains the FloodNet(nn.Module) class
│   ├── train.py                      # Main training loop (calls train_model)
│   └── predict.py                    # (Optional) Separate script for final evaluation/prediction
├── main.py                           # The primary script to run training
├── requirement.txt                   # Dependency list
└── best_floodnet_model.pth
```

## Contributing

Feel free to fork this project and submit pull requests for improvements.

## Author

**Mobolaji Opeyemi Bolatito**  
📧 Contact: opeblow2021@gmail.com

---

*This project demonstrates the application of deep learning techniques for environmental risk assessment and disaster preparedness.*
