import pandas as pd

# Load the CSV file
df = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Combine all elements columns into a single list
elements_columns = [f'Elements of electrocatalyst_{i}' for i in range(7)]
elements = df[elements_columns].values.flatten()

# Filter out any NaN values and count the occurrences of each element
elements = [element for element in elements if pd.notna(element)]
element_counts = pd.Series(elements).value_counts()

# Get the most used elements
most_used_elements = element_counts.head()

print(most_used_elements)