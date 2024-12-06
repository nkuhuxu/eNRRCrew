import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average Faraday efficiency
average_faraday_efficiency = data['Faraday Efficiency'].mean()

# Output the result
average_faraday_efficiency