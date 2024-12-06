import pandas as pd
from collections import Counter

# Load the CSV file
df = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the element columns
element_columns = [f'Elements of electrocatalyst_{i}' for i in range(7)]
elements = df[element_columns].values.flatten()

# Filter out NaN values and count occurrences
elements = [elem for elem in elements if pd.notna(elem)]
element_counts = Counter(elements)

# Get the most common elements
most_common_elements = element_counts.most_common()

# Print the results
print(most_common_elements)