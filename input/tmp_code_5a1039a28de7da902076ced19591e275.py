import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between Applied Potential (Faraday Efficiency) and Applied Potential (NH3 Yield)
correlation = data['Applied Potential (Faraday Efficiency)'].corr(data['Applied Potential (NH3 Yield)'])

correlation