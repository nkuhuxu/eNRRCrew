import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the relevant columns
applied_potential = data['Applied Potential (Faraday Efficiency)']
faraday_efficiency = data['Faraday efficiency']

# Calculate the correlation coefficient
correlation = np.corrcoef(applied_potential, faraday_efficiency)[0, 1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(applied_potential, faraday_efficiency, alpha=0.7)
plt.title('Relationship between Applied Potential and Faraday Efficiency')
plt.xlabel('Applied Potential (Faraday Efficiency)')
plt.ylabel('Faraday Efficiency')
plt.grid()
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.axvline(x=0, color='r', linestyle='--')  # Add a vertical line at x=0
plt.text(0.1, 0.9, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)

# Show the plot
plt.show()