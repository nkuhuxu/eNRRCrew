import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the relevant columns
applied_potential_yield = data['Applied Potential (NH3 Yield)']
yield_mg_edited = data['Yield_mg_edited']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential_yield, yield_mg_edited, alpha=0.7)
plt.title('Relationship between Applied Potential (NH3 Yield) and Yield_mg_edited')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Yield_mg_edited')
plt.grid()
plt.show()