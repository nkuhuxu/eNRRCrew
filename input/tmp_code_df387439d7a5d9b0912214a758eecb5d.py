import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Get the overall statistics of the 'Faraday efficiency' column
faraday_efficiency_stats = data['Faraday efficiency'].describe()

# Print the statistics
print(faraday_efficiency_stats)