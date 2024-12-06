import csv
from collections import Counter

# Initialize a counter for chemical elements
element_counter = Counter()

# Read the CSV file
with open('data_include_morphology_electrocatalyst.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Assuming the chemical elements are listed in a column named 'element'
        element = row['element']
        element_counter[element] += 1

# Find the most common element
most_common_element = element_counter.most_common(1)

# Output the most used chemical element
most_used_element = most_common_element[0] if most_common_element else None
print(most_used_element)