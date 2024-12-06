import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Calculate the correlation between Faraday efficiency and Applied Potential (Faraday Efficiency)
correlation = data['Faraday efficiency'].corr(data['Applied Potential (Faraday Efficiency)'])
print(f'Correlation between Faraday efficiency and Applied Potential (Faraday Efficiency): {correlation}')

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(data['Applied Potential (Faraday Efficiency)'], data['Faraday efficiency'], alpha=0.5)
plt.title('Faraday Efficiency vs Applied Potential (Faraday Efficiency)')
plt.xlabel('Applied Potential (Faraday Efficiency)')
plt.ylabel('Faraday Efficiency')
plt.grid()
plt.show()