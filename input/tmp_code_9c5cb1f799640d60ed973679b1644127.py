import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between Applied Potential (NH3 Yield) and Faraday efficiency
correlation = data['Applied Potential (NH3 Yield)'].corr(data['Faraday efficiency'])

# Output the correlation result
correlation