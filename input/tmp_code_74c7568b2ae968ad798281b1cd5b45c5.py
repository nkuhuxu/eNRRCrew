import pandas as pd
import joblib

# Load the new data
data = pd.read_csv('new_data.csv')

# Load the pre-trained model
model = joblib.load('model.pkl')

# Assuming the model expects specific features, we need to select them
# Replace 'feature_1', 'feature_2', ... with actual feature names used in the model
features = data[['Applied Potential (Faraday Efficiency)', 'pH_acidic', 'Electrocatalyst', 'Faraday efficiency']]  # Example features

# Make predictions
predictions = model.predict(features)

# Add predictions to the original data
data['Predicted Yield'] = predictions

# Save the results to a new CSV file
data.to_csv('predicted_yield_results.csv', index=False)