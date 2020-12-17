
# 필요한 라이브러리 import
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
# 데이터 불러오기
zoo_df = pd.read_csv("./zoo.csv")
# 데이터 전처리
target = zoo_df.iloc[:,-1]
features = zoo_df.iloc[:, 1 :-1]
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=5)
# model = DecisionTreeClassifier()
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

print(pred)

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
# numpy로 변환
zoo_ftr = features.to_numpy()
zoo_tar = target.to_numpy()

dt1_clf = DecisionTreeClassifier(max_depth=5)

dt1_clf.fit(X_train, y_train)
# kfold validation
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
# StratifiedKFold
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

X_train, X_test, y_train, y_test = train_test_split(zoo_ftr, zoo_tar,
                                                    test_size=0.2, random_state=50)
girdDT = DecisionTreeClassifier()
parameter = {'Max_depth' : [1, 2, 3], 'Min_samples_split': [1, 2]}
grid_girdDT = GridSearchCV(estimator=girdDT, cv=3, param_grid=parameter, refit=True, return_train_score=True)

scores_df = pd.DataFrame(grid_girdDT.cv_results_)
scores_df.columns = ['params', 'mean_test_score', 'rank_test_score','split0_test_score', 'split1_test_score', 'split2_test_score']

print('GridSearchCV 최적 파라미터:', grid_girdDT.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_girdDT.best_score_))

# refit=True로 설정된 GridSearchCV 객체가 fit()을 수행 시 학습이 완료된 Estimator를 내포하고 있으므로 predict()를 통해 예측도 가능.
pred = grid_girdDT.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))