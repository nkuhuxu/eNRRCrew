import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Extract the relevant columns
x = data['Applied Potential (Faraday Efficiency)']
y = data['Faraday efficiency']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title('Relationship between Applied Potential and Faraday Efficiency')
plt.xlabel('Applied Potential (Faraday Efficiency)')
plt.ylabel('Faraday Efficiency')
plt.grid()
plt.show()