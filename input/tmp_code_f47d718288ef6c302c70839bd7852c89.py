import pandas as pd
import joblib

# Load the new data
new_data = pd.read_csv('new_data.csv')

# Load the trained model
model = joblib.load('model.pkl')

# Assuming the model expects certain features, we need to select them from new_data
# Replace 'feature_1', 'feature_2', ... with actual feature names used in the model
features = new_data[['feature_1', 'feature_2', 'feature_3']]  # Adjust this line based on your model's requirements

# Make predictions
predictions = model.predict(features)

# Add predictions to the new_data DataFrame
new_data['Predicted_Yield'] = predictions

# Save the results to a new CSV file
new_data.to_csv('predicted_yield.csv', index=False)