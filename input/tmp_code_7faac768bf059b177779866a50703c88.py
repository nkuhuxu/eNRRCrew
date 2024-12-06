import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Display the column names
column_names = data.columns.tolist()
column_names