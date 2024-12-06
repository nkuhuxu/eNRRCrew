import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between Yield_mg_edited and Applied Potential (NH3 Yield)
correlation = data['Yield_mg_edited'].corr(data['Applied Potential (NH3 Yield)'])

# Output the correlation coefficient
print(f'The correlation coefficient between Yield_mg_edited and Applied Potential (NH3 Yield) is: {correlation}')