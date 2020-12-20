import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold  # K-fold CV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GroupKFold, cross_val_score
from functools import partial

df_1 = df.copy()
df_4 = df.copy()

df_1.drop(['도시', '설계수량', '도급단가합계', '공구별면적'], axis=1, inplace=True)
df_4.drop(['공사', 'modified공종명', '주거면적', '주거면적(%)', '공구별세대수',
           '기간(월)', 'hdb_clusters', '도-단cluster', '공-단cluster', '도공-단cluster',
           '도-단비cluster', '공-단비cluster', '도공-단비cluster', '도-수비cluster',
           '공-수비cluster', '도공-수비cluster'], axis=1, inplace=True)

# 라벨 인코더
le = preprocessing.LabelEncoder()
df_2 = df_1.apply(le.fit_transform)

# 원핫 인코더
enc = preprocessing.OneHotEncoder()
enc.fit(df_2)
df_3 = enc.transform(df_2).toarray()
df_3 = pd.DataFrame(df_3)

df_svr = pd.concat([df_4, df_3], axis=1)

# train, test 나누기
train_df = df_svr[df_svr['도시'] != '동탄5']
test_df = df_svr[df_svr['도시'] == '동탄5']
train_df = train_df[train_df['도급단가합계'] > 0]
test_df = test_df[test_df['도급단가합계'] > 0]

X_train = train_df.drop('도급단가합계', axis=1)
y_train = np.log1p(train_df['도급단가합계'])

X_test = test_df.drop('도급단가합계', axis=1)
y_test = test_df['도급단가합계']

# 공구명 삭제
X_train.drop('도시', axis=1, inplace=True)
X_test.drop('도시', axis=1, inplace=True)

# Group KFold 정의
group_kfold = GroupKFold(n_splits=5)  # GroupKFold
groups = X_train['공구별면적']  # Groupclass


def svr_cv(C, gamma, epsilon, x_data=None, y_data=None, n_splits=5, output='model'):
    score = 0
    models = []
    i = 0
    for train_index, valid_index in group_kfold.split(x_data, groups=groups):
        # print("TRAIN:", train_index, "TEST:", valid_index)
        print('Fold : ', i)
        x_train, y_train = x_data.iloc[train_index], y_data.iloc[train_index]
        x_valid, y_valid = x_data.iloc[valid_index], y_data.iloc[valid_index]

        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)

        model.fit(x_train, y_train)
        models.append(model)

        pred = model.predict(x_valid)
        true = y_valid
        RMSE = (mean_squared_error(true, pred) ** 0.5) * (-1)
        score += RMSE / n_splits
        i += 1

    if output == 'score':
        return score
    if output == 'model':
        return models


# func_fixed = partial(svr_cv, C = 10, gamma = 10, epsilon = 0.1, x_data=X_train, y_data=y_train, n_splits=5, output='score')

# score = svr_cv(C = 10, gamma = 10, epsilon = 0.1, x_data = X_train, y_data = y_train, n_splits = 5, output = 'score')
# print(score)


'''
Cuckoo Search
'''
from random import *
import math


def initNest(n):
    x = []
    for i in range(n):
        c = round(uniform(1350, 1450), 3)
        gamma = round(uniform(1350, 1450), 3)
        eps = round(uniform(0, 1), 3)
        x.append([c, gamma, eps])
    return x


def levyflight():
    return math.pow(uniform(0.0001, 0.9999), -1.0 / 3.0)


n = 25
nests = initNest(n)
maxGen = 1000
Gen = 0
pa = 0.2

minerror = 98765432
min_nest = []

for i in range(1, maxGen + 1):
    print('---------------', Gen, '---------------')
    k = randint(0, n - 1)
    cuckooNest = np.array(nests[k]) + levyflight()

    model_nests = SVR(kernel='rbf', C=nests[k][0], gamma=nests[k][1], epsilon=nests[k][2])
    model_nests.fit(X_train, y_train)
    y_pred = model_nests.predict(X_test)
    fj = mean_squared_error(y_test, y_pred)

    model_cuckoo = SVR(kernel='rbf', C=cuckooNest[0], gamma=cuckooNest[1], epsilon=cuckooNest[2])
    model_cuckoo.fit(X_train, y_train)
    y_pred = model_cuckoo.predict(X_test)
    fi = mean_squared_error(y_test, y_pred)

    if (fi > fj):
        nests[k] = cuckooNest
    if (random() <= pa):
        nests[k] = initNest(1)[0]

    for j in range(n):
        model_nests = SVR(kernel='rbf', C=nests[j][0], gamma=nests[j][1], epsilon=nests[j][2])
        model_nests.fit(X_train, y_train)
        y_pred = model_nests.predict(X_test)
        fj = mean_squared_error(y_test, y_pred)

        if (fj < minerror):
            minerror = fj
            min_nest = nests[j]
    Gen += 1

print('-----------------------------------')
print(minerror, min_nest)