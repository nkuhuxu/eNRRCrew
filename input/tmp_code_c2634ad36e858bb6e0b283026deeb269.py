import pandas as pd

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Assuming the column containing the elements is named 'Element'
element_counts = data['Element'].value_counts()

# Get the two most common elements
top_two_elements = element_counts.head(2)
top_two_elements_names = top_two_elements.index.tolist()

top_two_elements_names