# Machine Failure Detection

This project implements a machine learning system for predictive maintenance to detect machine failures based on sensor data and operational parameters.

## Project Structure

```
├── input/
│   └── predictive_maintenance.csv          # Input dataset
│
├── src/
│   ├── train.py          # Script for training models
│   ├── predict.py        # Script for making predictions
│   ├── model_selection.py # Model selection functionality
│   ├── tune_model.py     # Model tuning functionality
│   └── utils.py          # Utility functions
│
├── models/
│   ├── best_model.pkl    # Best performing model
│   └── tuned_model.pkl   # Hyperparameter-tuned model
│
├── notebooks/
│   └── machine-failure-detection.ipynb    # Jupyter notebook for EDA and experiments
│
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Dataset

The dataset contains measurements from machines in a manufacturing setting, including:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Target variable (machine failure)
- Type of failure (when applicable)

## Getting Started

### Prerequisites

- Python 3.7+
- Dependencies listed in requirements.txt

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Training Models

To train models with default parameters:

```bash
python src/train.py
```

For hyperparameter tuning:

```bash
python src/train.py --tune
```

Additional options:

```bash
python src/train.py --help
```

### Making Predictions

To make predictions on new data:

```bash
python src/predict.py --input_data path/to/predictive_maintenance.csv
```

Or with a JSON input:

```bash
python src/predict.py --input_data path/to/data.json --model_path models/tuned_model.pkl
```

## Model Performance

Multiple classification models are evaluated, including:

- Random Forest
- Gradient Boosting
- AdaBoost
- Stacked Ensembles
- Neural Networks (MLP)
- Support Vector Machines
- K-Nearest Neighbors
- XGBoost
- LightGBM

Performance metrics include accuracy, precision, recall, F1-score, and runtime.
