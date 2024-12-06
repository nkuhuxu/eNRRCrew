import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data_include_morphology_electrocatalyst.csv')

# Plotting the relationship between Applied Potential (NH3 Yield) and Yield
plt.figure(figsize=(10, 6))
plt.scatter(data['Applied Potential (NH3 Yield)'], data['Faraday efficiency'], color='blue', alpha=0.5)
plt.title('Relationship between Applied Potential and Faraday Efficiency')
plt.xlabel('Applied Potential (NH3 Yield)')
plt.ylabel('Faraday Efficiency')
plt.grid()
plt.tight_layout()

# Show the plot
import streamlit as st
st.pyplot(plt.gcf())