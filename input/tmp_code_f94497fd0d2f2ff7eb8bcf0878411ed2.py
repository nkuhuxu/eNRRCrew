import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract relevant columns
applied_potential = data['Applied Potential (Faraday Efficiency)']
faraday_efficiency = data['Faraday efficiency']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential, faraday_efficiency, alpha=0.7)
plt.title('Relationship between Applied Potential and Faraday Efficiency')
plt.xlabel('Applied Potential (Faraday Efficiency)')
plt.ylabel('Faraday Efficiency')
plt.grid(True)
plt.show()