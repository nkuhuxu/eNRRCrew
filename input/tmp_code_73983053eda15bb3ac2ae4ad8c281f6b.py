import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Display the column names to identify the correct ones
print(data.columns)

# Assuming the correct column names are 'Applied Potential' and 'Faradaic Efficiency'
# Calculate the average applied potential where faradaic efficiency is not null
average_applied_potential = data['Applied Potential'][data['Faradaic Efficiency'].notnull()].mean()

# Output the result
average_applied_potential