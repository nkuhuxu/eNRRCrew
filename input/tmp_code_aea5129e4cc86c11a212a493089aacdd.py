import pandas as pd

# Load the CSV file
file_path = 'data_include_morphology_electrocatalyst.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()