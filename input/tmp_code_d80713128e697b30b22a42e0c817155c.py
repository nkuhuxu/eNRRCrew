import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average applied potential where faradaic efficiency is not null
average_applied_potential = data['Applied Potential (Faraday Efficiency)'][data['Faraday efficiency'].notnull()].mean()

# Output the result
average_applied_potential