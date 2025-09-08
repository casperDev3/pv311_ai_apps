from  sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[50], [80], [100], [120], [150], [180], [200]]) # Feature: Size in square meters
y = np.array([150, 200, 250, 300, 350, 400, 450]) # Target: Price in dollars

model = LinearRegression()
model.fit(X, y)



print(f"Price prediction for 149 sqm: ${model.predict([[149]])[0]:.2f}")