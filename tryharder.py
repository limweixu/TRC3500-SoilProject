import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# === Constants ===
SATURATION_VOLUME = 52  # ml volume at 100% moisture

# Load the data
df = pd.read_csv("DATA2.csv", header=None)

# Extract volumes and ADC values
volumes = df[0].astype(int)  # Convert volumes to integers
adc_values = df[1].astype(float)  # Ensure ADC values are floats

# Prepare data for linear regression
X = volumes.values.reshape(-1, 1)
y = adc_values.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# === Calculate R-squared ===
r_squared = model.score(X, y)

# Extract regression parameters
slope = model.coef_[0]
intercept = model.intercept_

# Calculate predicted values
y_pred = model.predict(X)

# === Deviation Calculation ===
# Calculate absolute deviations
absolute_deviations = np.abs(y - y_pred)

# Calculate full-scale ADC resolution (max - min of ADC values)
full_scale_adc = np.max(adc_values) - np.min(adc_values)

# Calculate percentage deviations
percentage_deviations = (absolute_deviations / full_scale_adc) * 100

# Find maximum deviation
max_absolute_deviation = np.max(absolute_deviations)
max_percentage_deviation = np.max(percentage_deviations)

# Index of maximum deviation
max_deviation_index = np.argmax(absolute_deviations)

# Calculate confidence interval for slope
def calculate_slope_confidence_interval(X, y, confidence=0.90):
    n = len(X)
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_error_estimate = np.sqrt(np.sum(residuals**2) / (n - 2))
    xx_sum = np.sum((X - np.mean(X))**2)
    std_error_slope = std_error_estimate / np.sqrt(xx_sum)
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 2)
    return t_value * std_error_slope

# Calculate confidence interval
margin_of_error = calculate_slope_confidence_interval(X, y)


# === Moisture Percentage Calculation ===
def volume_to_moisture(volume):
    """
    Convert volume to moisture percentage
    
    Parameters:
    volume (float): Volume in ml
    
    Returns:
    float: Moisture percentage (0-100%)
    """
    moisture = (volume / SATURATION_VOLUME) * 100
    return max(0, min(100, moisture))  # Clamp between 0 and 100

def adc_to_volume(adc_value):
    """
    Convert ADC value to volume using linear regression
    
    Parameters:
    adc_value (float): ADC reading to convert
    
    Returns:
    float: Estimated volume in ml
    """
    volume = (adc_value - intercept) / slope
    return volume

def adc_to_moisture(adc_value):
    """
    Convert ADC value directly to moisture percentage
    
    Parameters:
    adc_value (float): ADC reading to convert
    
    Returns:
    float: Moisture percentage (0-100%)
    """
    volume = adc_to_volume(adc_value)
    return volume_to_moisture(volume)

# Visualization
plt.figure(figsize=(15, 6))

# --- Plot 1: Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(volumes, adc_values, color='blue', label='Original Data Points')

# Main regression line
plt.plot(volumes, y_pred, color='red', label='Linear Regression')

# Calculate vertical offsets (±10% of the mean ADC value)
mean_adc = np.mean(y_pred)
vertical_offset = mean_adc * 0.1

# Plot offset lines with same slope but vertical displacement
plt.plot(volumes, y_pred + vertical_offset, 
         color='green', linestyle='--', label='+10% Vertical Offset')
plt.plot(volumes, y_pred - vertical_offset, 
         color='green', linestyle='--', label='-10% Vertical Offset')

plt.xlabel('Volume (ml)')
plt.ylabel('ADC Value')
plt.title('Soil Moisture Linear Regression')
plt.legend()
plt.grid(True)

# Add regression equation and confidence details
equation = f'ADC = {slope:.4f} * Volume + {intercept:.4f}'
confidence_text = f'R-squared: {r_squared:.4f}\n'
confidence_text += f'Slope Confidence Interval (90%): ±{margin_of_error:.6f}\n'
confidence_text += f'Slope Variation: {(margin_of_error/slope)*100:.2f}%\n'
confidence_text += f'Max Deviation: {max_percentage_deviation:.4f}% of Full Scale'
plt.text(0.05, 0.95, equation + '\n' + confidence_text, 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# --- Plot 2: Moisture Percentage
plt.subplot(1, 2, 2)
moisture_percentages = [volume_to_moisture(vol) for vol in volumes]
plt.plot(volumes, moisture_percentages, marker='o', color='purple')
plt.xlabel('Volume (ml)')
plt.ylabel('Moisture (%)')
plt.title('Estimated Soil Moisture Percentage')
plt.grid(True)

# plt.tight_layout()
plt.show()

# Print additional deviation details
print("\nDeviation Analysis:")
print(f"Full-scale ADC Resolution: {full_scale_adc:.2f}")
print(f"Maximum Absolute Deviation: {max_absolute_deviation:.4f}")
print(f"Maximum Percentage Deviation: {max_percentage_deviation:.4f}%")
print(f"Volume at Max Deviation: {volumes[max_deviation_index]} ml")
print(f"Actual ADC: {adc_values[max_deviation_index]:.4f}")
print(f"Predicted ADC: {y_pred[max_deviation_index]:.4f}")



# Interactive conversion loop
def main():
    print("Soil Moisture ADC to Volume & Moisture Converter")
    print(f"Regression Equation: {equation}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Saturation Point: {SATURATION_VOLUME} ml\n")
    
    while True:
        try:
            # Prompt for ADC input
            adc_input = float(input("\nEnter ADC Value (0 to exit): "))
            
            # Exit condition
            if adc_input == 0:
                print("Exiting...")
                break
            
            # Convert ADC to volume and moisture
            volume = adc_to_volume(adc_input)
            moisture = adc_to_moisture(adc_input)
            
            # Round to nearest 2ml for practical measurement
            estimated_volume = round(volume)
            
            print(f"ADC Input: {adc_input}")
            print(f"Exact Volume Detected: {volume:.2f} ml")
            print(f"Estimated Volume (rounded to ml): {estimated_volume} ml")
            print(f"Estimated Soil Moisture: {moisture:.2f}%")
        
        except ValueError:
            print("Invalid input. Please enter a valid numeric ADC value.")

# Run the main function
if __name__ == "__main__":
    main()

# Print out regression details for verification
print("\nRegression Details:")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Slope Confidence Interval (90%): ±{margin_of_error:.6f}")
print(f"Slope Variation: {(margin_of_error/slope)*100:.2f}%")
