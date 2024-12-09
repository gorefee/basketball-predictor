import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Example dataset (you can replace this with real data)
data = {
    'FieldGoal%': [45.3, 47.8, 50.1, 42.0, 48.5],
    'Assists': [5.2, 7.1, 8.5, 6.0, 6.8],
    'Rebounds': [8.1, 9.2, 7.5, 10.1, 8.0],
    'ThreePointAttempts': [5.0, 6.5, 4.2, 7.3, 6.0],
    'Altitude': [1000, 2000, 1500, 2500, 1200],
    'HomeCourtAdvantage': [1, 0, 1, 0, 1],
    'Performance': [20.3, 25.4, 22.8, 21.5, 23.0]  # Target variable
}

# Convert to NumPy arrays
X = np.array([
    data['FieldGoal%'],
    data['Assists'],
    data['Rebounds'],
    data['ThreePointAttempts'],
    data['Altitude'],
    data['HomeCourtAdvantage']
]).T  # Shape (n_samples, n_features)

y = np.array(data['Performance'])  # Target variable

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale features
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))  # Add intercept term

# Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize coefficients
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)  # Partial derivatives
        theta -= learning_rate * gradient  # Update coefficients

        # Debugging: Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Mean Squared Error = {np.mean(errors ** 2):.4f}")
    return theta

# Train the model
theta = gradient_descent(X_scaled, y, learning_rate=0.01, iterations=1000)

# Save the model and scaler
joblib.dump({'theta': theta, 'scaler': scaler}, 'basketball_model.pkl')
print("Model trained and saved as basketball_model.pkl")
