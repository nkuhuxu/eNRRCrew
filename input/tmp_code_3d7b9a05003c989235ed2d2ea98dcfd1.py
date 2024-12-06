import pandas as pd

# Load the CSV file
file_path = './input/data_include_morphology_electrocatalyst.csv'
data = pd.read_csv(file_path)

# Identify the Electrode Potential Column
# Assuming the column name is 'Electrode Potential', you may need to adjust this based on the actual column name in your CSV.
electrode_potential_column = 'Electrode Potential'

# Calculate the Average
average_electrode_potential = data[electrode_potential_column].mean()

# Output the result
print("Average Electrode Potential:", average_electrode_potential)