import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate


zoo_df = pd.read_csv("./zoo.csv")

target = zoo_df.iloc[:,-1]
features = zoo_df.iloc[:, 1 :-1]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=5)

dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

print(pred)

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

zoo_ftr = features.to_numpy()
zoo_tar = target.to_numpy()

dt1_clf = DecisionTreeClassifier(max_depth=5)

dt1_clf.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True) # 5 0.89
cv_accuracy = []

print('데이터 세트 크기:',zoo_ftr.shape[0])
print('target 세트 크기:',zoo_tar.shape[0])

print('\nKFold의 검증결과')
n_iter = 0
for train_index, test_index in kf.split(zoo_ftr):
    X_train, X_test = zoo_ftr[train_index], zoo_ftr[test_index]
    y_train, y_test = zoo_tar[train_index], zoo_tar[test_index]

    dt1_clf.fit(X_train, y_train)
    pred = dt1_clf.predict(X_test)
    n_iter += 1

    accuracy = round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]


    print("\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증데이터 크기 : {3}"
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)


print("\n## 평균 검증 정확도: ", np.mean(cv_accuracy))

print('\nStratifiedKFold의 검증결과')
dt_clf = DecisionTreeClassifier(random_state=5)

SKf = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

# StratifiedKFold의 split( ) 호출시 반드시 레이블 데이터 셋도 추가 입력 필요
for train_index, test_index in SKf.split(zoo_ftr, zoo_tar):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = zoo_ftr[train_index], zoo_ftr[test_index]
    y_train, y_test = zoo_tar[train_index], zoo_tar[test_index]

    # 학습 및 예측
    dt1_clf.fit(X_train, y_train)
    pred = dt1_clf.predict(X_test)

    # 반복 시 마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy))


# 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개
print('\ncross_val_score : ')
scores = cross_val_score(dt_clf, zoo_ftr, zoo_tar
                         , scoring='accuracy' ,cv=3)
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))