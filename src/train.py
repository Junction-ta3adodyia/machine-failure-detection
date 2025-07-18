import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_data, preprocess_data, prepare_features, save_model
from model_selection import ModelSelection
from tune_model import tune_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train machine failure detection models'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='../input/predictive_maintenance.csv',
        help='Path to the input data CSV file'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.3,
        help='Proportion of data to use for testing'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--tune', 
        action='store_true',
        help='Whether to tune the best model'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../models',
        help='Directory to save trained models'
    )
    
    return parser.parse_args()


# Main function to train models
def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    
    print("Preprocessing data...")
    data, label_encoder = preprocess_data(data)
    
    print("Preparing features...")
    X, y = prepare_features(data)
    
    print(f"Splitting data with test_size={args.test_size}, random_state={args.random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print("Selecting best model...")
    model_selector = ModelSelection(X_train, y_train, X_test, y_test)
    model_selector.fit_all_models()
    
    model_name, best_model, accuracy, runtime = model_selector.get_best_model()
    
    print("\n===== Best Model Results =====")
    print(f"Best model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training runtime: {runtime:.2f}s")
    print("Classification Report:")
    print(model_selector.get_classification_report())
    
    # Save best model
    best_model_path = os.path.join(args.output_dir, "best_model.pkl")
    save_model(best_model, best_model_path)
    
    # Tune model if requested
    if args.tune:
        print(f"\nTuning {model_name}...")
        tuned_model = tune_model(model_name, best_model, X_train, y_train)
        
        # Evaluate tuned model
        from sklearn.metrics import accuracy_score
        y_pred_tuned = tuned_model.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        
        print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
        print(f"Improvement: {tuned_accuracy - accuracy:.4f}")
        
        # Save tuned model
        tuned_model_path = os.path.join(args.output_dir, "tuned_model.pkl")
        save_model(tuned_model, tuned_model_path)


if __name__ == "__main__":
    main()
