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

zoo_ftr = features.to_numpy()
zoo_tar = target.to_numpy()

kf = KFold(n_splits=5)
kf.get_n_splits(zoo_ftr)
kf.get_n_splits(zoo_tar)
cv_accuracy = []

print('데이터 세트 크기:',zoo_ftr.shape[0])
print('target 세트 크기:',zoo_tar.shape[0])

n_iter = 0
for train_index, test_index in kf.split(zoo_ftr):
    X_train, X_test = zoo_ftr[train_index], zoo_ftr[test_index]
    y_train, y_test = zoo_tar[train_index], zoo_tar[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    accuracy = round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증데이터 크기 : {3}"
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스: {1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)


print("\n## 평균 검증 정확도: ", np.mean(cv_accuracy))