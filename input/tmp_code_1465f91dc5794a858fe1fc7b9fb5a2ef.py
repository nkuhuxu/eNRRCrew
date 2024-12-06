import pandas as pd

# Load the CSV file
file_path = 'data_include_morphology_electrocatalyst.csv'
data = pd.read_csv(file_path)

# Calculate the average Faradaic efficiency
average_faradaic_efficiency = data['Faraday efficiency'].mean()

# Output the average Faradaic efficiency
print("Average Faradaic Efficiency:", average_faradaic_efficiency)