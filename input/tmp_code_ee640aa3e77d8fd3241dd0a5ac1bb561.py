import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Applied Potential (NH3 Yield)'], data['Faraday efficiency'], alpha=0.7)
plt.title('Faraday Efficiency vs Applied Potential (NH3 Yield)')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Faraday Efficiency')
plt.grid(True)
plt.show()