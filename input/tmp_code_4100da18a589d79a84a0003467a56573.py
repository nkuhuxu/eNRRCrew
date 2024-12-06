import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Clean the Yield_mg column to extract numeric values
data['Yield_mg'] = data['Yield_mg'].str.extract('([0-9.]+)').astype(float)

# Calculate the correlation between Yield_mg and Applied Potential (NH3 Yield)
correlation = data['Yield_mg'].corr(data['Applied Potential (NH3 Yield)'])

# Output the correlation result
print(f'The correlation between Yield_mg and Applied Potential (NH3 Yield) is: {correlation}')