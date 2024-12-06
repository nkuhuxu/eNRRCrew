import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'data_include_morphology_electrocatalyst.csv'
data = pd.read_csv(file_path)

# Extract the elements columns
elements_columns = [f'Elements of electrocatalyst_{i}' for i in range(7)]
elements_data = data[elements_columns]

# Melt the DataFrame to have a single column of elements
melted_elements = elements_data.melt(value_name='Element').dropna()

# Count the occurrences of each element
element_counts = melted_elements['Element'].value_counts()

# Plot the results
plt.figure(figsize=(10, 6))
element_counts.plot(kind='bar')
plt.title('Most Commonly Used Constituent Elements of Electrocatalysts')
plt.xlabel('Elements')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()