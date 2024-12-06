import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average Faraday efficiency
average_faraday_efficiency = data['Faraday efficiency'].mean()

# Print the average Faraday efficiency
print("Average Faraday Efficiency:", average_faraday_efficiency)