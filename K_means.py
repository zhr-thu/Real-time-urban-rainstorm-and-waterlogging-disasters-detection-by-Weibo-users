import csv
import math
import json
import jieba
import re
import jieba.analyse as analyse
import scipy.sparse as ss
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

def mcut(string_content):
    jcut=jieba.lcut(string_content)
    stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
    mcut=[]
    for word in jcut:
        if word not in stopwords:
            if word != '\t':
                mcut.append(word)
    return mcut
remove_list = ["#北京暴雨#","#北京深夜大暴雨#","北京暴雨预警"]
content = csv.reader(open("beijing202008_4.csv",'r',encoding='utf-8'))
whole = []
text = []

for row in content:
    text.append(row[3])
    whole.append((row[3],row[2]))
with open('情感词典\BosonNLP_sentiment_score\BosonNLP_sentiment_score.json', 'r', encoding='utf-8') as f:
    va_words = json.load(f)
#构建idf语料库
# idf_dic = {}
# for i in whole:
#     words = mcut(i[0])
#     for word in words:
#         if len(word) > 1:
#             idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
# for k, v in idf_dic.items():
#     w = k
#     p = '%.10f' % (math.log(len(whole)/ (1.0 + v)))
#     # 判断w是否是中文，用utf码区间判断
#     if w > u'\u4e00' and w <= u'\u9fa5':
#         idf_dic[w] = p
# with open("idf.txt", "w", encoding='utf-8') as f:
#      for k in idf_dic:
#         if k != '\n':
#             f.write(str(k) + ' ' + str(idf_dic[k]) + '\n')
# for i in whole:
#     if len(i[0]) > 12:
#         for j in remove_list:
#             if re.search(j,i[0]):
#                 i[0].replace(j,"")
#     words = mcut(i[0])
#     jieba.analyse.set_idf_path("idf.txt")
#     s = ""
#     s = s.join(words)
#     top = jieba.analyse.extract_tags(s, topK=3, withWeight=False)
#     print(top)
se = []
for i in whole:
    # mat = re.search('【.+】',i[0])
    # if mat:
    #     pass
    # else:
    #     se.append(i)
    se.append(i)
idf_dic = {}
# with open('or.txt','w',encoding='utf-8') as f:
#     for i in se:
#         f.writelines(str(i[1])+'\n')
# print('*'*100)
word_list = []
for i in se:
    if len(i[0]) > 12:
        for j in remove_list:
            mat = re.search(j,i[0])
            if mat:
                i = (i[0].replace(j,""),) + i[1:]
    words = mcut(i[0])
    for word in words:
        if len(word) > 1 and word not in word_list:
            word_list.append(word)
        if len(word) > 1:
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
for k, v in idf_dic.items():
    w = k
    p = '%.10f' % (math.log(len(se)/ (1.0 + v)))
    # 判断w是否是中文，用utf码区间判断
    if w > u'\u4e00' and w <= u'\u9fa5':
        idf_dic[w] = float(p) / max(idf_dic.values())
# with open("idf.txt", "w", encoding='utf-8') as f:
#      for k in idf_dic:
#         if k != '\n':
#             f.write(str(k) + ' ' + str(idf_dic[k]) + '\n')
a = np.zeros(shape = (len(se),len(word_list)))
r = 0
for i in se:
    words = mcut(i[0])
    row = []
    col = []
    data = []
    for word in words:
        if word in word_list:
            j = word_list.index(word)
            col.append(j)
            row.append(0)
            k = float(idf_dic.get(word))
            data.append(k)
    # print(row,'\n',col,'\n',data)
    # print('='*50)
    c = ss.coo_matrix((data,(row,col)),shape=(1,len(word_list)))
    a[r] = c.toarray()
    r += 1

model = KMeans(n_clusters=2, max_iter=5000, random_state=1)
model.fit(a)
print(model.cluster_centers_)
res = pd.Series(model.labels_)
res0 = res[res.values == 0]
res1 = res[res.values == 1]
model.fit_predict(a)
print(model.labels_.tolist())
with open('kmeans.txt','w',encoding='utf-8') as f:
    for i in model.labels_.tolist():
        f.writelines(str(i)+'\n')


# 肘部法确定最佳分类
# iteration = 5000
# meandistortions = []
# for i in range(1,10):
#     e = 0
#     model = KMeans(n_clusters=i, max_iter=iteration)
#     model.fit(a)
#     meandistortions.append(sum(np.min(
#         cdist(a, model.cluster_centers_,
#               'euclidean'), axis=1)) / a.shape[0])
# plt.plot(range(1,10), meandistortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Mean distortion degree')
# plt.title('The elbow law')
# plt.show()

# # 轮廓系数确定最佳分类
# silhouette = []
# for i in range(2, 10):
#     model = KMeans(n_clusters=i,random_state=777)
#     k_means = model.fit(a)
#     # print(k_means.labels_)
#     t = metrics.silhouette_score(a, k_means.labels_,metric='euclidean')
#     silhouette.append(t)
# plt.plot(range(2, 10), silhouette, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Silhouette Coefficient')
# plt.title('Silhouette Coefficient Method')
# plt.show()
