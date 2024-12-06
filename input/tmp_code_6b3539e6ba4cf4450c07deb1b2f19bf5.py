import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Select relevant features and target variable
# Assuming 'Yield_mg' is the target variable
features = data.drop(columns=['Yield_mg', 'Yield_cm', 'N-15 labeling_mentioned', 'N-15 labeling_nan'])
target = data['Yield_mg']

# Handle missing values in features and target
features.fillna(0, inplace=True)
target.dropna(inplace=True)

# Ensure features and target are aligned after dropping NaNs
features = features.loc[target.index]

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')