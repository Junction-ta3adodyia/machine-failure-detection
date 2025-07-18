import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Model imports
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb

from utils import save_model


"""
Class to select the best model for machine failure detection by comparing multiple
models based on accuracy and runtime.
"""
class ModelSelection:
    
# Initialize the ModelSelection class.
    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.get_models()
        
    # Initialize all models to be tested.
    def get_models(self):
        # Basic models
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Ada Boost': AdaBoostClassifier(),
            'Bagging': BaggingClassifier(),
            'Multi Layer Perceptron': MLPClassifier(),
            'Logistic Regression': LogisticRegression(),
            'k Nearest Neighbours': KNeighborsClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'XGBoost': XGBClassifier(),
            'SVM': SVC(probability=True)
        }
        
        # Add stacked model
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('svm', SVC(random_state=42))
        ]
        self.models['Stacked (RF & SVM)'] = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression()
        )
        
        return list(self.models.values())
    
    # Fit all models and evaluate performance.
    def fit_all_models(self):  
        model_names = []
        model_instances = []
        model_acc = []
        model_time = []
        
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            start = time.time()
            
            # Special handling for KNN to find optimal k
            if name == 'k Nearest Neighbours':
                accuracy = []
                for j in range(1, 50):  # Try different k values
                    knn = KNeighborsClassifier(n_neighbors=j)
                    knn.fit(self.X_train, self.y_train)
                    pred = knn.predict(self.X_test)
                    accuracy.append([accuracy_score(self.y_test, pred), j])
                
                # Find best k
                best_acc, best_k = max(accuracy, key=lambda x: x[0])
                model = KNeighborsClassifier(n_neighbors=best_k)
                print(f"  Best k for KNN: {best_k}")
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            stop = time.time()
            runtime = stop - start
            
            model_names.append(name)
            model_instances.append(model)
            model_acc.append(acc)
            model_time.append(runtime)
            
            print(f"  Accuracy: {acc:.4f}, Runtime: {runtime:.2f}s")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Model': model_names,
            'Instance': model_instances,
            'Accuracy': model_acc,
            'Runtime (s)': model_time
        })
        
        # Sort by accuracy (descending) then runtime (ascending)
        self.results_df = results.sort_values(
            by=['Accuracy', 'Runtime (s)'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        return self.results_df
    
    # Get the best performing model.
    def get_best_model(self):

        if not hasattr(self, 'results_df'):
            self.fit_all_models()
        
        best_row = self.results_df.iloc[0]
        return (
            best_row['Model'],
            best_row['Instance'],
            best_row['Accuracy'],
            best_row['Runtime (s)']
        )
    
    # Save the best model to disk.
    def save_best_model(self, filepath='../models/best_model.pkl'):
        _, best_model, _, _ = self.get_best_model()
        save_model(best_model, filepath)

    # Generate classification report for a model.    
    def get_classification_report(self, model=None):

        if model is None:
            _, model, _, _ = self.get_best_model()
        
        y_pred = model.predict(self.X_test)
        return classification_report(self.y_test, y_pred)
