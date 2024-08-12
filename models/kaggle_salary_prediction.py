import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier


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
lr_model = LinearRegression()
lr_model.fit(X, y)
pred = lr_model.predict(X)
lr_model.score(X, y)
# plot actual vs prediction
plot_prediction(X, y, pred)
print(r2_score(y, pred))

#################################
# Decision Tree
#################################
dtr_model = DecisionTreeRegressor(random_state=42, max_leaf_nodes=8)
dtr_model.fit(X, y)
y_pred2 = dtr_model.predict(X)
print(r2_score(y, y_pred2))
plot_prediction(X, y, y_pred2, model_name="DecisionTreeRegressor")
# Plot decision tree
plt.figure(figsize=(20, 10), dpi=100)
plot_tree(
    dtr_model,
    feature_names=["YearsExperience"],
    class_names=["Salary"],
    rounded=True,
    filled=True,
)
plt.show()

################################
# k Nearest Neighbor
################################
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X, y)
y_pred_knn = knn_model.predict(X)
print(r2_score(y, y_pred_knn))
plot_prediction(X, y, y_pred_knn, model_name="kNN")
