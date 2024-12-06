import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Check if the 'Faraday efficiency' column exists and display its contents
if 'Faraday efficiency' in data.columns:
    faraday_efficiency_data = data['Faraday efficiency']
    print(faraday_efficiency_data)
else:
    print("The 'Faraday efficiency' column does not exist in the CSV file.")