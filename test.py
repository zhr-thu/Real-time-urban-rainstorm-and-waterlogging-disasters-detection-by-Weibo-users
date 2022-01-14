import jieba.posseg
import csv
import sklearn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import re
import numpy as np
import scipy.sparse as ss
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from sklearn import svm
import time
import gensim
import pickle

def poscut(string,stopwords):
    pair = jieba.posseg.cut(string.replace(" ",""))
    pos = []
    for i in pair:
        if i.word not in stopwords:
            if i.word not in pos:
                pos.append(i.word)
    return pos
# stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
t = time.time()
stopwords = []
idf_dic = {}
# remove_list = ["#北京暴雨#","#北京深夜大暴雨#","北京暴雨预警",'#北京的暴雨是下丢了吗#','#你以为的北京暴雨#','#北京的雨有多大#','#京津冀区域性暴雨#','#北京雨来了#']
content = csv.reader(open("beijing202008_6.csv",'r',encoding='utf-8'))
whole = []
text = []
# model = gensim.models.Word2Vec.load('word2')
for row in content:
    text.append(row[3])
    whole.append((row[3], row[2]))
se = []
word_list = []
for i in whole:
    words = poscut(i[0], stopwords)
    if len(i[0]) > 0:
        se.append(i)
        for word in words:
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
            if word not in word_list:
                word_list.append(word)
print(len(word_list))
print(time.time()-t)
t = time.time()
# r = 0
# x = np.zeros(shape=(len(se), 200))
# for i in se:
#     words = poscut(i[0], stopwords)
#     value = np.zeros(200)
#     for word in words:
#         if word in model.wv:
#             value += model.wv[word]
#     x[r] = value
#     r = r+1
for k, v in idf_dic.items():
    w = k
    p = '%.10f' % (math.log(len(whole) / (1.0 + v)))
    idf_dic[w] = float(p) / max(idf_dic.values())
se.remove(se[0])
x = np.zeros(shape=(len(se), len(word_list)))
fx = np.zeros(shape=(len(se), len(word_list)))
r = 0
for i in se:
    words = poscut(i[0], stopwords)
    row = []
    col = []
    data = []
    data2 = []
    for word in words:
        if word in word_list:
            j = word_list.index(word)
            col.append(j)
            row.append(0)
            k = float(idf_dic.get(word))
            data.append(k)
            data2.append(1)
    # print(row,'\n',col,'\n',data)
    # print('='*50)
    c = ss.coo_matrix((data, (row, col)), shape=(1, len(word_list)))
    d = ss.coo_matrix((data2, (row, col)), shape=(1, len(word_list)))
    x[r] = c.toarray()
    fx[r] = d.toarray()
    r += 1
content2 = csv.reader(open("beijing202107_4.csv",'r',encoding='utf-8'))
whole2 = []
for row in content2:
    whole2.append((row[4], row[3]))
se2 = []
for i in whole2:
    if len(i[0]) > 0:
        se2.append(i)
se2.remove(se2[0])
w = np.zeros(shape=(len(se2), len(word_list)))
fw = np.zeros(shape=(len(se2), len(word_list)))
r = 0
for i in se2:
    words = poscut(i[0], stopwords)
    row = []
    col = []
    data = []
    data2 = []
    for word in words:
        if word in word_list:
            j = word_list.index(word)
            col.append(j)
            row.append(0)
            k = float(idf_dic.get(word))
            data.append(k)
            data2.append(1)
    c = ss.coo_matrix((data, (row, col)), shape=(1, len(word_list)))
    d = ss.coo_matrix((data2, (row, col)), shape=(1, len(word_list)))
    w[r] = c.toarray()
    fw[r] = d.toarray()
    r += 1

y = []
for i in se:
    if i[1] == '2':
        y.append(1)
    else:
        y.append(i[1])
print(y)
y = np.array(y)
z = []
for i in se2:
    if i[1] == '2':
        z.append(1)
    else:
        z.append(i[1])
print(z)
z = np.array(z)
print(time.time()-t)
t = time.time()
# train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(fx, y, random_state=2, train_size=0.8)
list = [20]
for i in list:
    clf = svm.SVC(C=i, gamma=1, kernel='linear')
    clf.fit(fx, y)
    # print('训练集精确度：', (clf.score(fx, y)))
    # print('测试集精确度：', (clf.score(fw, z)))
    d = clf.predict(fw)
    b = np.array(z)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for k in range(len(d)):
        if int(d[k]) == 0:
            if int(b[k]) == 0:
                TN += 1
            else:
                FN += 1
        else:
            if int(b[k]) == 0:
                FP += 1
            else:
                TP += 1
    print(i, TP, FN, FP, TN)
    print(time.time() - t)
    t = time.time()
    l = []
    for index in range(len(z)):
        # a = clf.predict(np.array(fx[index]).reshape(1, -1))
        a = clf.predict(np.array(fw[index]).reshape(1, -1))
        if a == z[index]:
            continue
        else:
            l.append(index)
    with open('2.txt','w',encoding='utf-8') as f:
        for i in l:
            f.write(str(i))
            f.write('\n')
    print('训练完成，下面开始测试')
    while True:
        weibo = input()
        if weibo == '0':
            break
        words = poscut(weibo, stopwords)
        row = []
        col = []
        data = []
        data2 = []
        for word in words:
            if word in word_list:
                j = word_list.index(word)
                col.append(j)
                row.append(0)
                data2.append(1)
        d = ss.coo_matrix((data2, (row, col)), shape=(1, len(word_list))).todense()
        print(clf.predict(d))
    # print('训练集精确度：', (clf.score(train_x, train_y)))
    # print('测试集精确度：', (clf.score(test_x, test_y)))
    # d = clf.predict(test_x)
    # b = np.array(test_y)
    # TP = 0
    # FN = 0
    # FP = 0
    # TN = 0
    # for k in range(len(d)):
    #     if int(d[k]) == 0:
    #         if int(b[k]) == 0:
    #             TN += 1
    #         else:
    #             FN += 1
    #     else:
    #         if int(b[k]) == 0:
    #             FP += 1
    #         else:
    #             TP += 1
    # print(i, TP, FN, FP, TN)
    # print(time.time() - t)
    # t = time.time()

# d = clf.predict(test_x).tolist()
# b = test_y.tolist()
# c = []
# for j in b:
#     c.append(j[0])
# with open('1.txt','w',encoding='utf-8') as f:
#     f.write(str(d))
#     f.write(str(c))
# list = [25]
# for i in list:
#     clf = svm.SVC(C=i, gamma=1, kernel='linear')
#     clf.fit(fx, y)
#     print('训练集精确度：', (clf.score(fx, y)))
#     print('测试集精确度：', (clf.score(fw, z)))
#     d = clf.predict(fw)
#     b = np.array(z)
#     TP = 0
#     FN = 0
#     FP = 0
#     TN = 0
#     for k in range(len(d)):
#         if int(d[k]) == 0:
#             if int(b[k]) == 0:
#                 TN += 1
#             else:
#                 FN += 1
#         else:
#             if int(b[k]) == 0:
#                 FP += 1
#             else:
#                 TP += 1
#     print(i, TP, FN, FP, TN)
#     print(time.time() - t)
#     t = time.time()
