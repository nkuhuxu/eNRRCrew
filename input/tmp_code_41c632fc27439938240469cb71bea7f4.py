import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Clean the 'Yield_mg' column to extract numeric values
data['Yield_mg'] = data['Yield_mg'].str.extract('(\d+\.?\d*)')[0].astype(float)

# Calculate the correlation between 'Applied Potential (NH3 Yield)' and 'Yield_mg'
correlation = data['Applied Potential (NH3 Yield)'].corr(data['Yield_mg'])

# Print the correlation result
print("Correlation between Applied Potential (NH3 Yield) and Yield (mg):", correlation)