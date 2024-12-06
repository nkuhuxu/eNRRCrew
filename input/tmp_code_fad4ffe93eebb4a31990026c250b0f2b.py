import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between Yield_mg and Applied Potential (NH3 Yield)
correlation = data['Yield_mg'].corr(data['Applied Potential (NH3 Yield)'])

# Output the correlation result
print(f'The correlation between Yield_mg and Applied Potential (NH3 Yield) is: {correlation}')