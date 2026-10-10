
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Ice Cream Revenue Prediction Project

# Create sample dataset
data = {
    "Temperature": [15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
    "Revenue": [200, 250, 300, 350, 420, 500, 560, 610, 700, 780]
}

df = pd.DataFrame(data)

# Display dataset
print("Ice Cream Revenue Dataset:")
print(df)

# Prepare data for training
X = df[["Temperature"]]
y = df["Revenue"]

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict revenue at 27 degrees Celsius
temperature = pd.DataFrame({"Temperature": [27]})
predicted_revenue = model.predict(temperature)

print("\nTemperature:", 27, "degrees Celsius")
print("Predicted Revenue:", round(predicted_revenue[0], 2))

# Visualize actual and predicted revenue
plt.scatter(X["Temperature"], y, label="Actual Revenue")
plt.plot(X["Temperature"], model.predict(X),
         label="Regression Line")

plt.xlabel("Temperature (°C)")
plt.ylabel("Revenue")
plt.title("Ice Cream Revenue Prediction")
plt.legend()
plt.show()
