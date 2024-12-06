import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Convert columns to numeric and drop NaN values
data['Applied Potential (NH3 Yield)'] = pd.to_numeric(data['Applied Potential (NH3 Yield)'], errors='coerce')
data['Yield_cm'] = pd.to_numeric(data['Yield_cm'], errors='coerce')
data = data.dropna(subset=['Applied Potential (NH3 Yield)', 'Yield_cm'])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Applied Potential (NH3 Yield)'], data['Yield_cm'], alpha=0.5)
plt.title('Relationship between Applied Potential for NH3 Yield and NH3 Yield')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('NH3 Yield (cm)')
plt.grid()
plt.show()