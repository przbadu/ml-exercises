import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("../data/kaggle/Salary.csv")

df.head()
df.describe()
df.isna().sum()

fig, ax = plt.subplots()
ax = df["Salary"].plot(kind="hist", bins=30)
ax = df["YearsExperience"].plot(kind="hist")
plt.show()

X = df[["YearsExperience"]]
y = df["Salary"]

# Train and Score data in Original Data
model = LinearRegression()
model.fit(X, y)
model.score(X, y)


# split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print("Y TEST: ", y_test)
print("Y PREDICT: ", y_pred)

# Calculate the error scores of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")
