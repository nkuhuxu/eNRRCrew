import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Assuming the relevant column for elements is named 'Element'
# Count the occurrences of each element
element_counts = data['Element'].value_counts()

# Get the two most common elements
top_two_elements = element_counts.head(2)

# Output the names of the two elements
top_two_elements.index.tolist()