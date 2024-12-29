import numpy as np
import matplotlib.pyplot as plt

# Define the function to evaluate directly
def func(x, y):
    """
    Function to be integrated: Z = X * exp(-X^2 - Y^2)
    """
    return x * np.exp(-x**2 - y**2)

# Monte Carlo Integration Parameters
num_samples = 1000000  # Number of Monte Carlo samples
x_min, x_max = -2, 2.5  # X bounds
y_min, y_max = -2, 2.5  # Y bounds
z_min, z_max = -0.4, 0.4  # Z bounds (from visual inspection or prior knowledge)

# Generate random samples for X, Y, Z
x_rand = np.random.uniform(x_min, x_max, num_samples)
y_rand = np.random.uniform(y_min, y_max, num_samples)
z_rand = np.random.uniform(z_min, z_max, num_samples)

# Evaluate the function directly at sampled (x, y) points
z_func = func(x_rand, y_rand)

# Determine how many points fall under the surface
inside_positive = (0 < z_rand) & (z_rand < z_func)
inside_negative = (z_func < z_rand) & (z_rand < 0)

# Count the number of points in positive and negative regions
num_inside_positive = np.sum(inside_positive)
num_inside_negative = np.sum(inside_negative)

# Calculate the integration volume
volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

# Estimate the integral using Monte Carlo method
integral_estimate = (num_inside_positive - num_inside_negative) / num_samples * volume



# Visualization (optional)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the wireframe for the function
X, Y = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
Z = func(X, Y)
ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5, alpha=0.7, label="Surface")

# Scatter sampled points under the curve (optional visualization)
ax.scatter(x_rand[inside_positive], y_rand[inside_positive], z_rand[inside_positive],
           color='red', s=0.01, label="Points Under Curve")
ax.scatter(x_rand[inside_negative], y_rand[inside_negative], z_rand[inside_negative],
           color='green', s=0.01, label="Points Below Surface")

# Set labels and show plot
ax.set_title("Monte Carlo Integration")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()
