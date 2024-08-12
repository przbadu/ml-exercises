import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score


def plot_prediction(X, y, pred, model_name="LinearRegression"):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X, y, label="Actual")
    ax.plot(X, pred, "r", label="Predicted")
    ax.set_title(f"Prediction using {model_name}")
    ax.set_xlabel("Yeas of Experience")
    ax.set_ylabel("Salary")
    legend = ax.legend(loc="upper center", shadow=True, fontsize="x-large")
    legend.get_frame().set_facecolor("C0")
    plt.show()


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
pred = model.predict(X)
model.score(X, y)
# plot actual vs prediction
plot_prediction(X, y, pred)
print(r2_score(y, pred))

#################################
# Decision Tree
#################################
model3 = DecisionTreeRegressor(random_state=42, max_leaf_nodes=8)
model3.fit(X, y)
y_pred2 = model3.predict(X)
plot_prediction(X, y, y_pred2, model_name="DecisionTreeRegressor")
# Plot decision tree
plt.figure(figsize=(20, 10), dpi=100)
plot_tree(
    model3,
    feature_names=["YearsExperience"],
    class_names=["Salary"],
    rounded=True,
    filled=True,
)
plt.show()
