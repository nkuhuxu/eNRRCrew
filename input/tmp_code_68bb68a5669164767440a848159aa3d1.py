import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Assuming the pre-trained model is saved as 'pretrained_model.pkl'
model = joblib.load('pretrained_model.pkl')

# Prepare the features for prediction
# Selecting relevant features based on the model's requirements
features = data[['Faraday efficiency', 'Applied Potential (Faraday Efficiency)', 
                 'Applied Potential (NH3 Yield)', 'Electrocatalyst', 
                 'pH_acidic', 'pH_alkaline', 'Yield_mg_edited']]

# Make predictions
predictions = model.predict(features)

# Add predictions to the original dataframe
data['Predicted_Yield'] = predictions

# Save the results to a new CSV file
data.to_csv('predicted_yield_results.csv', index=False)