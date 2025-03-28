import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Load the data from CSV ===
df_voltage = pd.read_csv("voltage_interpolation.csv")

# Extract Row number and Voltage values
rows = df_voltage['Row'].values
voltage = df_voltage['Voltage'].values

# === Fit linear regression ===
X = rows.reshape(-1, 1)  # Row number as the independent variable
y = voltage  # Voltage as the dependent variable
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# === Regression prediction ===
y_pred = model.predict(X)
equation = f'Voltage = {slope:.4f} * Row + {intercept:.4f}'

# === Plotting ===
plt.figure(figsize=(12, 6))

# --- Plot: Voltage vs Row number without connecting lines
plt.scatter(rows, voltage, color='blue', label='Original Data Points')

# Plot the regression line
plt.plot(rows, y_pred, color='red', label='Linear Regression')

plt.xlabel('Water Level (ml)')
plt.ylabel('Voltage (V)')
plt.title('Voltage (V) vs Water Level (ml)')
plt.legend()
plt.grid(True)

# Display the regression equation on the plot
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Show the plot
plt.tight_layout()
plt.show()
