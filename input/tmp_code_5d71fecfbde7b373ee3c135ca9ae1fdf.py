import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Drop rows with NaN values in the relevant columns to avoid errors
data = data.dropna(subset=['Applied Potential (Faraday Efficiency)', 'Faraday efficiency'])

# Extract relevant columns
x = data['Applied Potential (Faraday Efficiency)']
y = data['Faraday efficiency']

# Calculate the correlation coefficient
correlation = x.corr(y)

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y)
plt.title('Relationship between Applied Potential and Faraday Efficiency')
plt.xlabel('Applied Potential (Faraday Efficiency)')
plt.ylabel('Faraday Efficiency')
plt.axhline(y=y.mean(), color='r', linestyle='--', label='Mean Faraday Efficiency')
plt.axvline(x=x.mean(), color='g', linestyle='--', label='Mean Applied Potential')
plt.legend()
plt.grid()
plt.savefig('faraday_efficiency_relationship.png')
plt.show()

# Output the correlation coefficient
correlation