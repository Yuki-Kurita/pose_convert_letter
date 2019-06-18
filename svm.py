# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pickle

def make_dataset():
    data_list = os.listdir("./data")
    coo_data = {}
    coo_vec = {}
    count = 0
    vec_list = [['LCS-X', '4X', '2X'], ['LCS-Y', '4Y', '2Y'], ['LSE-X', '5X', '4X'], ['LSE-Y', '5Y', '4Y'],\
    ['LEH-X', '7X', '5X'], ['LEH-Y', '7Y', '5Y'], ['RCS-X', '8X', '2X'], ['RCS-Y', '8Y', '2Y'],\
    ['RSE-X', '9X', '8X'], ['RSE-Y', '9Y', '8Y'], ['REH-X', '11X', '9X'], ['REH-Y', '11Y', '9Y'], ['CS-X', '1X', '2X'], ['CS-Y', '1Y', '2Y'],\
    ['SLH-X', '12X', '1X'], ['SLH-Y', '12Y', '1Y'], ['LHK-X', '13X', '12X'], ['LHK-Y', '13Y', '12Y'], ['LKA-X', '14X', '13X'], ['LKA-Y', '14Y', '13Y'],\
    ['SRH-X', '16X', '1X'], ['SRH-Y', '16Y', '1Y'], ['RHK-X', '17X', '16X'], ['RHK-Y', '17Y', '16Y'], ['RKA-X', '18X', '17X'], ['RKA-Y', '18Y', '17Y']]

    for data in data_list:
        with open('./data/' + data) as f:
            # 各関節をディクショナリに格納
            for line in f:
                coo_data[line.split(',')[0] + 'X'] = (line.split(',')[1]).replace("\n", '')
                coo_data[line.split(',')[0] + 'Y'] = (line.split(',')[2]).replace("\n", '')
        # ラベルを格納
        if count == 0:
            coo_vec['alpha'] = [data[0]]
        else:
            coo_vec['alpha'].append(data[0])

        # 各関節に対するベクトルを求める
        for vec in vec_list:
            if count == 0:
                coo_vec[vec[0]] = [int(coo_data[vec[1]]) - int(coo_data[vec[2]])]
            else:
                coo_vec[vec[0]].append(int(coo_data[vec[1]]) - int(coo_data[vec[2]]))

        count += 1

    #  各ベクトルをcsvファイルに出力
    df = pd.DataFrame({
    'LCS-X' : coo_vec['LCS-X'],  'LCS-Y' : coo_vec['LCS-Y'], 'LSE-X' : coo_vec['LSE-X'], 'LSE-Y' : coo_vec['LSE-Y'], 'LEH-X' : coo_vec['LEH-X'],\
    'LEH-Y' : coo_vec['LEH-Y'], 'RCS-X' : coo_vec['RCS-X'], 'RCS-Y' : coo_vec['RCS-Y'], 'RSE-X' : coo_vec['RSE-X'], 'RSE-Y' : coo_vec['RSE-Y'],\
    'REH-X' : coo_vec['REH-X'], 'REH-Y' : coo_vec['REH-Y'], 'CS-X' : coo_vec['CS-X'], 'CS-Y' : coo_vec['CS-Y'], 'SLH-X' : coo_vec['SLH-X'],\
    'SLH-Y' : coo_vec['SLH-Y'], 'LHK-X' : coo_vec['LHK-X'], 'LHK-Y' : coo_vec['LHK-Y'], 'LKA-X' : coo_vec['LKA-X'], 'LKA-Y' : coo_vec['LKA-Y'],\
    'SRH-X' : coo_vec['SRH-X'], 'SRH-Y' : coo_vec['SRH-Y'], 'RHK-X' : coo_vec['RHK-X'], 'RHK-Y' : coo_vec['RHK-Y'], \
    'RKA-X' : coo_vec['RKA-X'], 'RKA-Y' : coo_vec['RKA-Y'], 'label' : coo_vec['alpha']
    }
    ,columns = ['LCS-X', 'LCS-Y', 'LSE-X', 'LSE-Y', 'LEH-X', 'LEH-Y', 'RCS-X', 'RCS-Y', 'RSE-X', 'RSE-Y', 'REH-X', 'REH-Y',\
    'CS-X', 'CS-Y', 'SLH-X', 'SLH-Y', 'LHK-X', 'LHK-Y', 'LKA-X', 'LKA-Y','SRH-X', 'SRH-Y', 'RHK-X', 'RHK-Y', 'RKA-X', 'RKA-Y', 'label'])
    df.to_csv('./dataset.csv')


def alpha_int(Y):
    re_Y = np.array([], dtype='int64')
    for y in Y.values:
        if y == 'A':
            y = 0
        elif y == 'B':
            y = 1
        elif y == 'C':
            y = 2
        elif y == 'D':
            y = 3
        elif y == 'E':
            y = 4
        elif y == 'F':
            y = 5
        elif y == 'G':
            y = 6
        elif y == 'H':
            y = 7
        elif y == 'I':
            y = 8
        elif y == 'J':
            y = 9
        elif y == 'K':
            y = 10
        elif y == 'L':
            y = 11
        elif y == 'M':
            y = 12
        elif y == 'N':
            y = 13
        elif y == 'O':
            y = 14
        elif y == 'P':
            y = 15
        elif y == 'Q':
            y = 16
        elif y == 'R':
            y = 17
        elif y == 'S':
            y = 18
        elif y == 'T':
            y = 19
        elif y == 'U':
            y = 20
        elif y == 'V':
            y = 21
        elif y == 'W':
            y = 22
        elif y == 'X':
            y = 23
        elif y == 'Y':
            y = 24
        elif y == 'Z':
            y = 25

        re_Y = np.append(re_Y, y)
    return re_Y


# X = pd.read_csv('dataset.csv', usecols = lambda x : x not in 'label')
X = pd.read_csv('dataset.csv', usecols = ['LCS-X', 'LCS-Y', 'LSE-X', 'LSE-Y','LEH-X','LEH-Y', 'RCS-X', 'RCS-Y', 'RSE-X', 'RSE-Y', 'REH-X', 'REH-Y', 'CS-X', 'CS-Y',\
'SLH-X', 'SLH-Y', 'LHK-X', 'LHK-Y', 'LKA-X', 'LKA-Y', 'SRH-X', 'SRH-Y', 'RHK-X', 'RHK-Y', 'RKA-X', 'RKA-Y'])
Y = pd.read_csv('dataset.csv', usecols = ['label'])
y = alpha_int(Y)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=None )
# データの標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# 線形SVMのインスタンスを生成
model = SVC(kernel='linear', random_state=None)
# モデルの学習。fit関数で行う。
model.fit(X_train_std, y_train)
# トレーニングデータに対する精度
pred_train = model.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print("accurary rate for train : {}".format(accuracy_train))

# テストデータに対する精度
pred_test = model.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print("accurary rate for test : {}".format(accuracy_test))

# モデルを保存
with open('model.pickle', mode='wb') as fp:
    pickle.dump(model, fp)

# 作成したモデルに対して説明変数を代入すると目的変数を返す
print(model.predict([[-41, 6, -13, -70, 50, -89, 42, 7, 9, -78, -48, -81, 2, 71, -19, 69, -23, 118, -35, 101, 22, 69, 35, 119, 25, 99]]))
