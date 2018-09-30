# -*- coding:utf-8 -*-
import jieba
import codecs
import logging
import numpy
import os
import datetime
import pymysql
from gensim import models, corpora, similarities
from gensim.similarities.docsim import Similarity
import sys
"""https://blog.csdn.net/vs412237401/article/details/52238248"""

stop_words = './textCommend/data/hlt_stop_words.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf-8').readlines()
stopwords = [w.strip() for w in stopwords]
'''利用docsim来做文本相似'''
print("利用doc2sim做文本相似")


def get_corpus():
    connect = pymysql.Connect(host='10.10.50.155', port=3316, user='root',
                              passwd="123456", db="recommendtext", charset='utf8')
    # 获取游标
    cursor = connect.cursor()
    sql = "SELECT corpus,textCode FROM corpus"
    #print("sssss", sql)
    cursor.execute(sql)
    corpus_seg = cursor.fetchone()
    word_dict = {}
    while corpus_seg:
        # result.append(tokenrow(corpus_seg[0]))
        word_dict[corpus_seg[0]] = corpus_seg[1]

        corpus_seg = cursor.fetchone()

    return word_dict


def tokenrow(line):
    row = []
    words = jieba.cut(line)
    #print("words", words)
    for word in words:
        if word not in stopwords and word not in ['', ' ', '\n']:
            row.append(word)
    return row


corpus_documents = []
word_dict = get_corpus()
for key, value in word_dict.items():
    #print(key, value)
    corpus_documents.append(tokenrow(key))
# 生成字典语料
dictionary = corpora.Dictionary(corpus_documents)

corpus = [dictionary.doc2bow(text) for text in corpus_documents]
# num_features:为语料库的特征数（比如：字典大小，或者lsi的主题数等）
similarity = Similarity('-Similarity-index', corpus, num_features=400)

#test_data = "查找关于中国南方电网公司子系统下的移动终端版本管理从今天起的菜单数量统计"
test_data = "昨天北京交通情况？"
cut_data = tokenrow(test_data)
test_corpus = dictionary.doc2bow(cut_data)
print(test_corpus)
similarity.num_best = 5
sims = similarity[test_corpus]
#print('sim_list', sims_list)
# 相关度排序
#s_list = sorted(sims, reverse=True)
for i in range(5):
    index = sims[i][0]
    print("122", index)
    simtext = corpus_documents[index]
    print("".join(simtext), sims[i][1])
    # print(corpus_documents[index])
#     print(word_dict[txt])
    # 返回给定文档的相似度。docpos：要查询文档在index所在的位置。
    # print(similarity.similarity_by_id(index))
    # 在指定docpos位置处理返回所索引vector。
    # print(similarity.vector_by_id(index))


# #########
print("########################################")
# print("LSF短文本相似度")
