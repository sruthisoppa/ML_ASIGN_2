import numpy as np
import matplotlib.pyplot as plt

# Dataset
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Step 1: Analytic Solution for Linear Regression
N = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the coefficients
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * x_mean

# Predictions
y_pred = beta_0 + beta_1 * x

# Calculate SSE
SSE = np.sum((y - y_pred) ** 2)

# Calculate R^2
SST = np.sum((y - y_mean) ** 2)
R2 = 1 - SSE / SST

print("Analytic Solution:")
print(f"Beta 0 (Intercept): {beta_0}")
print(f"Beta 1 (Slope): {beta_1}")
print(f"SSE: {SSE}")
print(f"R^2: {R2}")

# Step 2: Gradient Descent Implementation

# Initialize coefficients
beta_0_gd = 0
beta_1_gd = 0
alpha = 0.01  # Learning rate
epochs = 1000  # Number of iterations

# Full-batch Gradient Descent
for epoch in range(epochs):
    y_pred_gd = beta_0_gd + beta_1_gd * x
    error = y_pred_gd - y
    beta_0_gd -= alpha * (1/N) * np.sum(error)
    beta_1_gd -= alpha * (1/N) * np.sum(error * x)

# Calculate SSE and R^2 for Gradient Descent
SSE_gd = np.sum((y - y_pred_gd) ** 2)
R2_gd = 1 - SSE_gd / SST

print("\nGradient Descent Solution:")
print(f"Beta 0 (Intercept): {beta_0_gd}")
print(f"Beta 1 (Slope): {beta_1_gd}")
print(f"SSE: {SSE_gd}")
print(f"R^2: {R2_gd}")

# Plotting the results
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Analytic Solution')
plt.plot(x, y_pred_gd, color='green', linestyle='--', label='Gradient Descent Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
