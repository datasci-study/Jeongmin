import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

zoo_df = pd.read_csv("./zoo.csv")

zoo_data = zoo_df

print(zoo_data.head(3))

zoo_label = zoo_data["class_type"]
zoo_data = zoo_data.drop(columns=['class_type','animal_name'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(zoo_data, zoo_label,
                                                    test_size=0.2, random_state=11)

dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)