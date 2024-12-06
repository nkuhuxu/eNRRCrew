import pandas as pd

# Load the CSV file
file_path = 'data_include_morphology_electrocatalyst.csv'
data = pd.read_csv(file_path)

# Check the columns to find the one related to Faradaic efficiency
print(data.columns)

# Assuming the column name for Faradaic efficiency is 'Faradaic Efficiency'
average_faradaic_efficiency = data['Faradaic Efficiency'].mean()

# Output the average Faradaic efficiency
print("Average Faradaic Efficiency:", average_faradaic_efficiency)