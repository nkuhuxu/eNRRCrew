import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Check the columns to find the one related to Faraday efficiency
print(data.columns)

# Assuming the column name for Faraday efficiency is 'Faraday Efficiency'
average_faraday_efficiency = data['Faraday Efficiency'].mean()

# Output the average Faraday efficiency
print(average_faraday_efficiency)