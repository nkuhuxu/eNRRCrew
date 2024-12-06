import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between 'Applied Potential (NH3 Yield)' and 'Yield_mg_edited'
correlation = data['Applied Potential (NH3 Yield)'].corr(data['Yield_mg_edited'])

# Output the correlation coefficient
print(f'The correlation coefficient between Applied Potential (NH3 Yield) and Yield_mg_edited is: {correlation}')