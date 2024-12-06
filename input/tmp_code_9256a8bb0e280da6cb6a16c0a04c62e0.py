import pandas as pd

# Load the CSV file
df = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Assuming there is a column named 'element' that contains the elements
most_used_element = df['element'].value_counts().idxmax()
most_used_count = df['element'].value_counts().max()

most_used_element, most_used_count