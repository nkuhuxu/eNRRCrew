import pandas as pd

# Load the CSV file
df = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Combine all elements columns into a single list
elements_columns = [f'Elements of electrocatalyst_{i}' for i in range(7)]
all_elements = df[elements_columns].values.flatten()

# Count the occurrences of each element
element_counts = pd.Series(all_elements).value_counts()

# Get the most used element
most_used_element = element_counts.idxmax()
most_used_count = element_counts.max()

print(f"The most used element is: {most_used_element} with {most_used_count} occurrences.")