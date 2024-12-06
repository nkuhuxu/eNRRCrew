import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average electrode potential for faradaic efficiency
average_potential = data['faradaic_efficiency'].mean()

average_potential