"""
Model tuning module for machine failure detection project.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils import save_model


def tune_random_forest(model, X_train, y_train, cv=5):
    """
    Tune Random Forest hyperparameters.
    
    Args:
        model: RandomForestClassifier model
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        The tuned model
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# Tune Gradient Boosting hyperparameters.
def tune_gradient_boosting(model, X_train, y_train, cv=5):


    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


# Tune SVM hyperparameters.
def tune_svm(model, X_train, y_train, cv=5):

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly'],
        'probability': [True]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# Tune model hyperparameters based on the model type.
def tune_model(model_name, model, X_train, y_train, cv=5):

    if "Random Forest" in model_name:
        return tune_random_forest(model, X_train, y_train, cv)
    elif "Gradient Boosting" in model_name:
        return tune_gradient_boosting(model, X_train, y_train, cv)
    elif "SVM" in model_name:
        return tune_svm(model, X_train, y_train, cv)
    else:
        print(f"No specific tuning function for {model_name}, returning original model")
        return model


# Save the tuned model to disk.
def save_tuned_model(model, filepath='../models/tuned_model.pkl'):
    save_model(model, filepath)
