import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Find the catalyst with the highest NH3 yield
highest_yield = data.loc[data['Applied Potential (NH3 Yield)'].idxmax()]

highest_yield