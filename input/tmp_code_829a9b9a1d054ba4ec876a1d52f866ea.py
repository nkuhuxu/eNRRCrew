import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average electrode potential for entries with faradaic efficiency
average_electrode_potential = data['Electrode Potential'][data['Faradaic Efficiency'].notnull()].mean()

# Output the result
print(average_electrode_potential)