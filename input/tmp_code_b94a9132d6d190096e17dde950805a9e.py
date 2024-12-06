import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data_path = './input/new_data.csv'
data = pd.read_csv(data_path)

# Select features and target variable
features = data.drop(columns=['Yield_mg', 'Yield_cm', 'Yield_mg_edited', 'Yield_cm_edited'])
target = data['Yield_mg']  # Assuming we want to predict Yield_mg

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Output predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Yield_mg'])
predictions_df.to_csv('./output/predicted_yield.csv', index=False)