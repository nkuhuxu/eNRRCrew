import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Applied Potential (NH3 Yield)'], data['Yield_mg_edited'], alpha=0.5)
plt.title('Relationship between Applied Potential (NH3 Yield) and Yield_mg_edited')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Yield_mg_edited')
plt.grid(True)
plt.show()