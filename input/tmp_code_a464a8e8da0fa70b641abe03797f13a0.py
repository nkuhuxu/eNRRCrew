import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between 'Applied Potential (Faraday Efficiency)' and 'Faraday efficiency'
correlation = data['Applied Potential (Faraday Efficiency)'].corr(data['Faraday efficiency'])

# Print the correlation result
print("Correlation between Applied Potential (Faraday Efficiency) and Faraday efficiency:", correlation)