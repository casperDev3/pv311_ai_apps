from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[2015, 50000, 2.0, 150],
             [2018, 30000, 3.0, 200],
              [2020, 20000, 4.0, 250],
              [2016, 40000, 2.5, 180],
              [2019, 25000, 3.5, 220],
              [2021, 15000, 4.5, 270]]) # Features: Year, Mileage, Engine Size (L), Horsepower

y = np.array([15000, 20000, 25000, 18000, 22000, 27000]) # Target: Price in dollars

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Втрати = MSE + α * (сума квадратів коефіцієнтів)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

print(f"Price prediction for a 2017 car with 35000 miles, 3.0L engine, 210 HP: ${ridge.predict(scaler.transform([[2022, 35000, 3.0, 200]]))[0]:.2f}")