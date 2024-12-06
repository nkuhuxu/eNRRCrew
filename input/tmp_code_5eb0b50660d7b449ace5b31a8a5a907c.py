import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Inspect the first few rows of the data
print(data.head())

# Descriptive statistics for yield_mg_edited and Applied Potential (NH3 Yield)
yield_stats = data[['Yield_mg_edited', 'Applied Potential (NH3 Yield)']].describe()
print(yield_stats)

# Correlation analysis
correlation = data['Yield_mg_edited'].corr(data['Applied Potential (NH3 Yield)'])
print(f'Correlation between Yield_mg_edited and Applied Potential (NH3 Yield): {correlation}')

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Applied Potential (NH3 Yield)', y='Yield_mg_edited', data=data)
plt.title('Scatter Plot of Yield_mg_edited vs Applied Potential (NH3 Yield)')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Yield_mg_edited')
plt.grid()
plt.show()