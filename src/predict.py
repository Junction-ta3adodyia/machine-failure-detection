import argparse
import pandas as pd
import numpy as np
import json

from utils import load_model


#Parse command line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained model'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='../models/best_model.pkl',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--input_data', 
        type=str, 
        required=True,
        help='Path to input data CSV or JSON file with machine parameters'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='predictions.json',
        help='Path to output JSON file for predictions'
    )
    
    return parser.parse_args()


# Load input data from file.
def load_input_data(input_path):
  
 
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            json_data = json.load(f)
        
        # Handle both single instance and multiple instances
        if isinstance(json_data, dict):
            data = pd.DataFrame([json_data])
        else:
            data = pd.DataFrame(json_data)
    else:
        raise ValueError("Input file must be .csv or .json")
    
    # Rename columns if needed
    column_mapping = {
        'Air temperature [K]': 'airtemp',
        'Process temperature [K]': 'processtemp',
        'Rotational speed [rpm]': 'rpm',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'toolwear'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in data.columns:
            data = data.rename(columns={old_col: new_col})
    
    required_columns = ['airtemp', 'processtemp', 'rpm', 'torque', 'toolwear']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return data[required_columns]


# Make predictions using the trained model.
def predict_failures(model, data):
    
    
    
    # Make predictions
    predictions = model.predict(data).tolist()
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(data).tolist()
        confidence = [prob[1] if pred == 1 else prob[0] for pred, prob in zip(predictions, probabilities)]
    except:
        confidence = [None] * len(predictions)
    
    # Create output dictionary
    results = {
        'predictions': [
            {
                'id': i,
                'failure_detected': bool(pred),
                'confidence': conf * 100 if conf is not None else None
            }
            for i, (pred, conf) in enumerate(zip(predictions, confidence))
        ],
        'summary': {
            'total_samples': len(predictions),
            'failures_detected': sum(predictions),
            'failure_rate': sum(predictions) / len(predictions) * 100
        }
    }
    
    return results


# Main function for prediction.
def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    print(f"Loading input data from {args.input_data}...")
    data = load_input_data(args.input_data)
    print(f"Loaded {len(data)} samples")
    
    print("Making predictions...")
    results = predict_failures(model, data)
    
    print(f"Writing predictions to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary: {results['summary']['failures_detected']} failures detected "
          f"out of {results['summary']['total_samples']} samples "
          f"({results['summary']['failure_rate']:.2f}% failure rate)")


if __name__ == "__main__":
    main()
