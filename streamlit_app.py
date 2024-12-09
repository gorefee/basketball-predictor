import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('basketball_model.pkl')
theta = model['theta']  # Gradient Descent coefficients
scaler = model['scaler']  # Scaler for feature normalization

# Example training data
data = {
    'FieldGoal%': [45.3, 47.8, 50.1, 42.0, 48.5],
    'Assists': [5.2, 7.1, 8.5, 6.0, 6.8],
    'Rebounds': [8.1, 9.2, 7.5, 10.1, 8.0],
    'ThreePointAttempts': [5.0, 6.5, 4.2, 7.3, 6.0],
    'Altitude': [1000, 2000, 1500, 2500, 1200],
    'HomeCourtAdvantage': [1, 0, 1, 0, 1],
    'Performance': [20.3, 25.4, 22.8, 21.5, 23.0]
}
df = pd.DataFrame(data)
X = df[['FieldGoal%', 'Assists', 'Rebounds', 'ThreePointAttempts', 'Altitude', 'HomeCourtAdvantage']].values
y = df['Performance'].values

# Scale features
X_scaled = scaler.transform(X)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))  # Add intercept term

# Closed-Form Solution
X_transpose = X_scaled.T
theta_closed_form = np.linalg.inv(X_transpose @ X_scaled) @ X_transpose @ y

# Streamlit App
st.title("Basketball Performance Predictor")
st.markdown("""
Compare Gradient Descent with the Closed-Form (OLS) method, and analyze Partial Derivatives.
""")

# User Input Fields
st.header("Input Player Stats")
field_goal_percentage = st.number_input("Field Goal Percentage (%)", min_value=0.0, max_value=100.0, value=47.0, step=0.1)
assists = st.number_input("Assists per Game", min_value=0.0, max_value=15.0, value=6.5, step=0.1)
rebounds = st.number_input("Rebounds per Game", min_value=0.0, max_value=20.0, value=5.2, step=0.1)
three_point_attempts = st.number_input("3-Point Attempts per Game", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
altitude = st.number_input("Altitude (ft)", min_value=0.0, max_value=6000.0, value=52.0, step=10.0)
home_court_advantage = st.number_input("Home Court Advantage (Yes=1, No=0)", min_value=0, max_value=1, value=1, step=1)

# Combine input variables
raw_input_data = np.array([field_goal_percentage, assists, rebounds, three_point_attempts, altitude, home_court_advantage])
scaled_input_data = scaler.transform([raw_input_data])  # Scale features
scaled_input_data = np.hstack(([1], scaled_input_data.flatten()))  # Add intercept term

# Predict Performance
if st.button("Predict Performance"):
    # Gradient Descent Prediction
    predicted_performance_gd = scaled_input_data @ theta
    st.write(f"**Gradient Descent Predicted Performance:** {round(predicted_performance_gd, 2)} points")

    # Closed-Form Solution Prediction
    predicted_performance_closed = scaled_input_data @ theta_closed_form
    st.write(f"**Closed-Form Predicted Performance:** {round(predicted_performance_closed, 2)} points")

    # Compare Predictions
    st.markdown("### Comparison of Predicted Performance")
    fig, ax = plt.subplots()
    methods = ["Gradient Descent", "Closed-Form"]
    predictions = [predicted_performance_gd, predicted_performance_closed]

    # Create a horizontal bar chart
    ax.barh(methods, predictions, color=["skyblue", "steelblue"])
    ax.set_xlabel("Predicted Points")
    ax.set_title("Comparison of Predicted Performance")
    for i, v in enumerate(predictions):
        ax.text(v + 0.1, i, f"{v:.2f}", va="center")  # Annotate bars

    st.pyplot(fig)

# Display Partial Derivatives
st.header("Partial Derivatives")
if st.button("Analyze Partial Derivatives"):
    st.write("""
    These represent the **impact of each variable** on performance (holding others constant).
    """)
    variables = ['Intercept', 'FieldGoal%', 'Assists', 'Rebounds', 'ThreePointAttempts', 'Altitude', 'HomeCourtAdvantage']
    for i, var in enumerate(variables):  # Correctly map variable names to coefficients
        st.write(f"- {var}: {theta[i]:.4f} (Impact per unit change)")

# Gradient Descent Optimization Simulation
st.header("Gradient Descent Optimization")
learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
iterations = st.slider("Iterations", 100, 1000, 500, step=100)

if st.button("Simulate Gradient Descent"):
    m, n = X_scaled.shape
    theta_sim = np.zeros(n)
    error_history = []

    for t in range(iterations):
        predictions = X_scaled @ theta_sim
        errors = predictions - y
        gradient = (1 / m) * (X_scaled.T @ errors)
        theta_sim -= learning_rate * gradient
        error_history.append(np.mean(errors ** 2))  # Mean Squared Error

    st.line_chart(error_history)
    st.write("### Final Simulated Coefficients (Gradient Descent)")
    variables = ['Intercept', 'FieldGoal%', 'Assists', 'Rebounds', 'ThreePointAttempts', 'Altitude', 'HomeCourtAdvantage']
    for i, coeff in enumerate(theta_sim):
        st.write(f"{variables[i]}: {coeff:.4f}")
    
    st.write("### Closed-Form Coefficients")
    for i, coeff in enumerate(theta_closed_form):
        st.write(f"{variables[i]}: {coeff:.4f}")

# Model Comparison
st.header("Model Comparison: Gradient Descent vs Closed-Form")
# Gradient Descent Errors
y_pred_gd = X_scaled @ theta
mse_gd = mean_squared_error(y, y_pred_gd)
mae_gd = mean_absolute_error(y, y_pred_gd)

# Closed-Form Errors
y_pred_closed = X_scaled @ theta_closed_form
mse_closed = mean_squared_error(y, y_pred_closed)
mae_closed = mean_absolute_error(y, y_pred_closed)

# Display Metrics
st.write("### Mean Squared Error (MSE)")
st.write(f"Gradient Descent: {mse_gd:.2f}")
st.write(f"Closed-Form: {mse_closed:.2f}")

st.write("### Mean Absolute Error (MAE)")
st.write(f"Gradient Descent: {mae_gd:.2f}")
st.write(f"Closed-Form: {mae_closed:.2f}")
