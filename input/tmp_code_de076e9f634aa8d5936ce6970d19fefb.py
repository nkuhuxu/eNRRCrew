import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Sample a smaller subset of the data
data_sample = data.sample(n=50, random_state=1)  # Sample 50 random rows

# Extract the relevant columns
yield_mg_edited = data_sample['Yield_mg_edited']
applied_potential = data_sample['Applied Potential (NH3 Yield)']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential, yield_mg_edited, alpha=0.7)
plt.title('Relationship between Yield (mg) and Applied Potential (NH3 Yield)')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Yield (mg edited)')
plt.grid()
plt.tight_layout()  # Optimize layout
plt.show()