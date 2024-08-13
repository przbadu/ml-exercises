import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

heart = pd.read_csv("../data/heart.csv")

# Force the response into a binary indicator:
# heart['target'] = 1 * (heart['target'] == 'Yes')

heart_train, heart_val = train_test_split(heart, train_size=0.75, random_state=42)
print(heart_train.shape, heart_val.shape)

x_train = heart_train[["age"]]
y_train = heart_train["target"]

# k-NN model
knn20 = KNeighborsClassifier(n_neighbors=20)
knn20.fit(x_train, y_train)


# there are two types of predictions in classification models in sklearn
# model.predict for pure classifications, and model.predict_proba for probabilities

# create the predictions based on the train data
yhat20_class = knn20.predict(x_train)
yhat20_prob = knn20.predict_proba(x_train)

print(yhat20_class[1:10])
print(yhat20_prob[1:10, :])

# Simple logistic regression model fitting
logit = LogisticRegression(penalty=None, max_iter=1000)
logit.fit(x_train, y_train)
print("Logistic Regression Estimated Betas (B0, B1):", logit.intercept_, logit.coef_)

logit.predict_proba(x_train)

# Define the equivalent validation variables from `heart_val`
x_val = heart_val[["age"]]
y_val = heart_val["target"]

# Compute the training & validation accuracy using the estimator.score() function
knn20_train_accuracy = knn20.score(x_train, y_train)
knn20_val_accuracy = knn20.score(x_val, y_val)
logit_train_accuracy = logit.score(x_train, y_train)
logit_val_accuracy = logit.score(x_val, y_val)

print("k-NN Train & Validation Accuracy:", knn20_train_accuracy, knn20_val_accuracy)
print(
    "Logisitic Train & Validation Accuracy:", logit_train_accuracy, logit_val_accuracy
)

# set-up the dummy x for plotting: we extend it a little bit beyond the range of observed values
# x = np.linspace(np.min(heart[['Age', ]])-10, 100+10, 200)
x = np.linspace(np.min(heart[["age"]]) - 10, 80 + 10, 200).reshape(-1, 1)

yhat_class_knn20 = knn20.predict(x)
yhat_prob_knn20 = knn20.predict_proba(x)[:, 1]

yhat_class_logit = logit.predict(x)
yhat_prob_logit = logit.predict_proba(x)[:, 1]

# plot the observed data.  Note: we offset the validation points to make them more clearly differentiated from train
plt.plot(x_train, y_train, "o", alpha=0.1, label="Train Data")
plt.plot(x_val, 0.94 * y_val + 0.03, "o", alpha=0.1, label="Validation Data")

# plot the prediction
plt.plot(x, yhat_class_knn20, label="knn20 Classifications")
plt.plot(x, yhat_prob_knn20, label="knn20 Probabilities")
plt.plot(x, yhat_class_logit, label="logit Classifications")
plt.plot(x, yhat_prob_logit, label="logit Probabilities")

# Put the lower-left part of the legend 5% to the right along the x-axis, and 45% up along the y-axis
plt.legend(loc=(0.05, 0.45))

# Don't forget your axis labels!
plt.xlabel("age")
plt.ylabel("Heart disease (AHD)")

plt.show()
