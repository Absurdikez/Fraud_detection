import time
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pickle

def train_model_with_timer(model, param_grid, model_name, X_train, y_train):
    """
    Trains model with grid search and progress tracking.
    """
    print(f"\n🔹 {model_name} - Training model...\n")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"\n✅ Training complete! Total time: {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}\n")
    
    return grid_search

def save_model(model, filename):
    """
    Saves trained model to disk.
    """
    with open(filename, "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved as '{filename}'")

def load_model(filename):
    """
    Loads trained model from disk.
    """
    with open(filename, 'rb') as model_file:
        return pickle.load(model_file)