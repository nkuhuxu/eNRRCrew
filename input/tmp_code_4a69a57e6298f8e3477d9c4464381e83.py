import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the relevant columns
yield_mg_edited = data['Yield_mg_edited']
applied_potential = data['Applied Potential (NH3 Yield)']

# Calculate the correlation
correlation = yield_mg_edited.corr(applied_potential)

# Display the correlation
print(f'The correlation between Yield (mg edited) and Applied Potential (NH3 Yield) is: {correlation}')