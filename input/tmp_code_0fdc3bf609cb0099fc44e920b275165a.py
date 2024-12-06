import csv
from collections import Counter

# Initialize a counter for chemical elements
element_counter = Counter()

# Read the CSV file and print the header to identify the correct column
with open('data_include_morphology_electrocatalyst.csv', mode='r') as file:
    reader = csv.DictReader(file)
    headers = reader.fieldnames  # Get the headers
    print("Headers in the CSV file:", headers)  # Print headers for identification

    # Assuming the chemical elements are listed in a column that we will identify
    for row in reader:
        # Replace 'element' with the correct column name after identifying it
        element = row['Your_Correct_Column_Name']  # Update this line after identifying the correct column
        element_counter[element] += 1

# Find the most common element
most_common_element = element_counter.most_common(1)

# Output the most used chemical element
most_used_element = most_common_element[0] if most_common_element else None
print(most_used_element)