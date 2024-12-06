import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the average of the 'Applied Potential (Faraday Efficiency)' column
average_potential = data['Applied Potential (Faraday Efficiency)'].mean()

average_potential