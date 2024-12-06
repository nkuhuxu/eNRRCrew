import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the relevant columns
yield_mg_edited = data['Yield_mg_edited']
applied_potential = data['Applied Potential (NH3 Yield)']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential, yield_mg_edited, alpha=0.7)
plt.title('Relationship between Yield (mg) and Applied Potential (NH3 Yield)')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Yield (mg edited)')
plt.grid()
plt.show()