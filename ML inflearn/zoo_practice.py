import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

zoo_df = pd.read_csv("./zoo.csv")

zoo_data = zoo_df

print(zoo_data.head(3))

zoo_target = zoo_data["class_type"]
zoo_data = zoo_data.drop(columns=['class_type','animal_name'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(zoo_data, zoo_target,
                                                    test_size=0.2, random_state=11)  # test_size = 0.5 accuracy = 0.9608

dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

print(pred)

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

features = zoo_data.to_numpy()
target = zoo_target.to_numpy()
dt1_clf = DecisionTreeClassifier(random_state=111)

kf = KFold(n_splits=5)
kf.get_n_splits(features)
cv_accuracy = []
print('zoo dataset shape : ', zoo_data.shape[0])

n_iter = 0
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]

    dt1_clf.fit(X_train, y_train)
    pred = dt1_clf.predict(X_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test.pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증데이터 크기 : {3}"
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스: {1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)

print("\n## 평균 검증 정확도: ", np.mean(cv_accuracy))
