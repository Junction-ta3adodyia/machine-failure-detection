import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Load the dataset from the given filepath.
def load_data(filepath='../input/predictive_maintenance.csv'):
    return pd.read_csv(filepath)


# Preprocess the dataset by removing unnecessary columns and renaming features.
def preprocess_data(data):

    # Drop unnecessary columns
    data = data.drop(["UDI", "Product ID"], axis=1)
    
    # Rename columns for better readability
    data = data.rename(columns={
        'Air temperature [K]': 'airtemp',
        'Process temperature [K]': 'processtemp',
        'Rotational speed [rpm]': 'rpm',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'toolwear'
    })
    
    # Encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(data['Target'])
    data['Target'] = label_encoder.transform(data['Target'])
    
    return data, label_encoder


# Prepare features for model training by selecting relevant columns.
def prepare_features(data):

    X = data.drop(['Failure Type', 'Target', 'Type'], axis=1)
    y = data['Target']
    return X, y


# Plot a confusion matrix for model evaluation.
def plot_confusion_matrix(y_true, y_pred, model_classes):
   
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# Save a trained model to disk.
def save_model(model, filename='../models/model.pkl'):

    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


# Load a trained model from disk.
def load_model(filename='../models/model.pkl'):

    import pickle
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
