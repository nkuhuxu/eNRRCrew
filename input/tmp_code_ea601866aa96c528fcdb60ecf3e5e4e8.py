import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract relevant columns
applied_potential_yield = data['Applied Potential (NH3 Yield)']
faraday_efficiency = data['Faraday efficiency']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential_yield, faraday_efficiency, alpha=0.7)
plt.title('Relationship between Applied Electrode Potential for Yield and Faradaic Efficiency')
plt.xlabel('Applied Potential (NH3 Yield) (V)')
plt.ylabel('Faraday Efficiency (%)')
plt.grid(True)
plt.show()