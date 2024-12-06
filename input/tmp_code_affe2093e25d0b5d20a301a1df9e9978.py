import pandas as pd
import joblib  # Directly import joblib

# Load the new data
new_data = pd.read_csv('./input/new_data.csv')

# Load the trained model (replace 'model.pkl' with your model file)
model = joblib.load('model.pkl')

# Preprocess the new data if necessary (e.g., handle missing values, encode categorical variables)
# This step will depend on how your model was trained

# Make predictions
predictions = model.predict(new_data)

# Add predictions to the new data DataFrame
new_data['Predicted_Yield'] = predictions

# Save the results to a new CSV file
new_data.to_csv('./output/predicted_yield.csv', index=False)