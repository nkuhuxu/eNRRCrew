import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import os

class YieldPredictor:
    def __init__(self, model_path: str = 'best_model.joblib', features_path: str = 'selected_features.txt'):
        """
        Initialize the YieldPredictor with paths to the model and features list.
        
        Args:
            model_path (str): Path to the saved model file
            features_path (str): Path to the file containing selected features
        """
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.selected_features = None
        self._load_resources()

    def _load_resources(self):
        """Load the model and selected features."""
        try:
            self.model = load(self.model_path)
            with open(self.features_path, 'r') as f:
                self.selected_features = [line.strip() for line in f]
        except Exception as e:
            raise Exception(f"Error loading resources: {str(e)}")

    def generate_sample_data(self, original_data_path: str, n_samples: int = 10) -> pd.DataFrame:
        """
        Generate sample data from the original dataset.
        
        Args:
            original_data_path (str): Path to the original dataset
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        try:
            original_data = pd.read_csv(original_data_path)
            np.random.seed(42)
            sample_data = original_data.sample(n=n_samples)
            return sample_data
        except Exception as e:
            raise Exception(f"Error generating sample data: {str(e)}")

    def validate_data(self, new_data: pd.DataFrame, original_data: pd.DataFrame) -> dict:
        """
        Validate the new data against the original dataset.
        
        Args:
            new_data (pd.DataFrame): New data to validate
            original_data (pd.DataFrame): Original dataset for comparison
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "original_columns_count": len(original_data.columns),
            "new_columns_count": len(new_data.columns),
            "missing_columns": list(set(original_data.columns) - set(new_data.columns)),
            "data_types": new_data.dtypes.to_dict(),
            "null_counts": new_data.isnull().sum().to_dict()
        }
        return validation_results

    def predict(self) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data_path (str): Path to the new data file
            
        Returns:
            pd.DataFrame: Prediction results
        """
        try:
            # Read new data
            new_data = pd.read_csv('new_data.csv')
            
            # Ensure all selected features exist
            for feature in self.selected_features:
                if feature not in new_data.columns:
                    new_data[feature] = 0.0
            
            # Prepare features for prediction
            X = new_data[self.selected_features]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Original_Index': new_data.index,
                'Prediction': predictions,
                'Probability': probabilities
            })
            
            # Combine with original data
            final_results = pd.concat([new_data, results_df], axis=1)
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}") 