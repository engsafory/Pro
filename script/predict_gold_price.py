import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Load the historical gold price data
file_path = input("Enter the file path of the gold price data: ")
data = pd.read_csv(file_path)

# Preprocess the data
data.dropna(inplace=True)
data['Date'] = pd.to_datetime(data['Date']).astype(int)/ 10**9
X = data['Date'].values.reshape(-1,1)
y = data['Price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the gold price
y_pred = model.predict(X_test)

# Evaluate the model's performance
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualize the actual vs predicted gold prices
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()