import csv
from collections import Counter

# Initialize a counter for chemical elements
element_counter = Counter()

# Read the CSV file with UTF-8 encoding
with open('data_include_morphology_electrocatalyst.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Iterate through each row in the CSV
    for row in reader:
        # Check all element columns
        for i in range(7):  # Assuming there are 7 element columns (0 to 6)
            element_column = f'Elements of electrocatalyst_{i}'
            element = row[element_column]
            if element:  # Only count non-empty elements
                element_counter[element] += 1

# Find the most common element
most_common_element = element_counter.most_common(1)

# Output the most used chemical element
most_used_element = most_common_element[0] if most_common_element else None
print(most_used_element)