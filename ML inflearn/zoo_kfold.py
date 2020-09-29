import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


zoo_df = pd.read_csv("./zoo.csv")

target = zoo_df.iloc[:,-1]
features = zoo_df.iloc[:, 1 :-1]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)

dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

print(pred)

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
