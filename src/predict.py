import numpy as np 
import pandas as pd
import torch
from .data_preprocessing import (
    df,
    text_feature,
    ordinal_features,
    onehot_features,
    numerical_features,
    label_enc,
    text_enc,
    ord_enc,
    onehot_enc,
    scaler,
    valid_categories
    
)

from .utils import show_valid_options,validate_input,get_closest_match
from .model import FloodNet

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=FloodNet().to(device)

def get_user_input():
    print("\nEnter environmental details to predict flood risk:")
    print("(Type 'help' to see valid options for any field)")
    user_input_data = {}
    prompts = {
        'Vegetation': "Vegetation: ",
        'Urbanization': "Urbanization: ",
        'Drainage': "Drainage: ",
        'Slope': "Slope: ",
        'StormFrequency': "Storm Frequency: ",
        'Deforestation': "Deforestation: ",
        'Infrastructure': "Infrastructure: ",
        'Encroachment': "Encroachment: ",
        'Season': "Season: ",
        'Soil': "Soil: ",
        'Wetlands': "Wetlands: ",
        'ProximityToWaterBody': "Proximity to water body: ",
        'Rainfall': "Rainfall (numeric): ",
        'Elevation': "Elevation (numeric): "
    }
    for feature, prompt in prompts.items():
        while True:
            value = input(prompt).strip()
            if value.lower() == 'help':
                if feature in valid_categories:
                    print(f"Valid options for {feature}: {', '.join(valid_categories[feature])}")
                continue
            if feature in ['Rainfall', 'Elevation']:
                try:
                    user_input_data[feature] = [float(value)]
                    break
                except ValueError:
                    print("Please enter a valid number.")
                    continue
            else:
                value = value.title()
                if validate_input(feature, value):
                    user_input_data[feature] = [value]
                    break
                else:
                    closest = get_closest_match(feature, value)
                    if closest:
                        print(f"Did you mean '{closest}'? Using that instead.")
                        user_input_data[feature] = [closest]
                        break
                    else:
                        print(f"Invalid input for {feature}.")
                        if feature in valid_categories:
                            print(f"Valid options: {', '.join(valid_categories[feature])}")
                        print("Please try again or type 'help' for options.")
    return pd.DataFrame(user_input_data)

def preprocess_user_input(user_df):
    try:
        for col in ordinal_features + onehot_features + [text_feature]:
            if col in user_df.columns:
                user_df[col] = user_df[col].fillna("Missing")
        ordinals = ord_enc.transform(user_df[ordinal_features]).astype(np.int64)
        onehots = onehot_enc.transform(user_df[onehot_features]).astype(np.float32)
        text_code = text_enc.transform(user_df[[text_feature]]).astype(np.int64).squeeze(1)
        numerical = user_df[numerical_features].copy()
        for col in ["Elevation", "Rainfall"]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            numerical[col] = numerical[col].clip(lower=max(q1 - 1.5 * iqr, 0), upper=q3 + 1.5 * iqr)
            mn = numerical[col].min()
            numerical[col] = np.log1p(numerical[col] + abs(mn) + 1e-6)
        numerical = scaler.transform(numerical).astype(np.float32)
        return (
            torch.tensor(text_code, dtype=torch.long).to(device),
            torch.tensor(ordinals, dtype=torch.long).to(device),
            torch.tensor(onehots, dtype=torch.float).to(device),
            torch.tensor(numerical, dtype=torch.float).to(device),
        )
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def predict_from_input():
    model.eval()
    user_df = get_user_input()
    text_tensor, ord_tensor, onehot_tensor, num_tensor = preprocess_user_input(user_df)
    with torch.no_grad():
        out = model(text_tensor, ord_tensor, onehot_tensor, num_tensor)
        probabilities = torch.softmax(out, dim=1)
        pred = out.argmax(1).item()
    risk_level = label_enc.inverse_transform([pred])[0]
    confidence = probabilities[0][pred].item()
    print(f"\n Predicted Flood Risk Level: {risk_level}")
    print(f" Confidence: {confidence:.1%}")
    print("\n Risk Breakdown:")
    for i, class_name in enumerate(label_enc.classes_):
        prob = probabilities[0][i].item()
        print(f"   {class_name}: {prob:.1%}")