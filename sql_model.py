# -*- coding:utf-8 -*-
'''将语料放入待数据库中进行处理，并实现增量训练'''
#-*- coding:utf-8 -*-
'''利用word2vec来训练，做文本相似'''
import json
import jieba
import codecs
import logging
import numpy
import os
import datetime
import pymysql
import sys
import argparse
from flask import Flask, Response,  request
from flask import render_template
from werkzeug import secure_filename
import time
app = Flask(__name__)
from gensim.models.word2vec import LineSentence, Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 构建停用词表

stop_words = './data/hlt_stop_words.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf-8').readlines()
stopwords = [w.strip() for w in stopwords]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        # elif isinstance(obj, bytes):
            # return str(obj, encoding='gbk')
        else:
            return super(MyEncoder, self).default(obj)


def tokenrow(line):
    row = []
    words = jieba.cut(line)
    #print("words", words)
    for word in words:
        if word not in stopwords and word not in ['', ' ', '\n']:
            row.append(word)
    return row


# 将语料存库
def saveSQL(tableid, corpus):
    flag = True  # 代表操作成功
    # 链接数据库
    connect = pymysql.Connect(host='10.10.50.155', port=3316, user='root',
                              passwd="123456", db="recommendtext", charset='utf8')
    # 获取游标
    cursor = connect.cursor()
    # 语料表还是，语料分割表
    if tableid == 0:  # 表示语料表插入
        # 取到上次最大值
        maxsql = "SELECT MAX(textId) FROM corpus"
        index = cursor.execute(maxsql)
        rr = cursor.fetchall()  # 获取返回全部数据
        print("max", rr[0][0])
        start = time.time()
        # 创建临时表，并插入数据
        # sql=""
        #sql = "INSERT INTO corpus (textId,textCode,corpus,segtextId,createTime)VALUES('%d','%d','%s','%d','%s')"
        sql = "INSERT INTO corpus (textId,textCode,corpus,segtextId,createTime) SELECT %s,%s,%s,%s,%s FROM dual  WHERE NOT EXISTS (SELECT * FROM corpus WHERE corpus.corpus=%s) "
        i = 1
        sql1 = "SELECT COUNT(*) FROM corpus WHERE corpus.corpus='%s'"
        insertdData = []
        for item, value in corpus.items():
            item = int(item)
            theTime = datetime.datetime.now()
            value = value.replace('\n', '').replace(" ", "")
            #print(sql1 % value)
            count = cursor.execute(sql1 % value)
            cc = cursor.fetchall()  # 获取返回全部数据
            print("count", cc[0][0])
            if cc[0][0] > 0:
                continue
            else:
                if rr[0][0]:
                    data = (rr[0][0] + i, item, value, rr[0][0] + i, theTime, value)
                else:
                    data = (i, item, value,  i, theTime, value)
                insertdData.append(data)
                i = i + 1
        # 批量插入
        try:
            cursor.executemany(sql, insertdData)
            connect.commit()
            end = time.time()
            print("time", end - start)
        except Exception as e:
            flag = False
            connect.rollback()
            print("error", e)
        #raise Exception("read error %s" % Exception)
        #print('成功插入', cursor.rowcount, '条数据')
    else:
        maxsql = "SELECT MAX(segtextId) FROM segcorpus"
        index = cursor.execute(maxsql)
        rr = cursor.fetchall()  # 获取返回全部数据
        #print("max", rr[0][0])
        start = time.time()
        sql = "INSERT INTO segcorpus (segtextId,segtextCode,segcorpus,textId,createTime) SELECT %s,%s,%s,%s,%s FROM dual  WHERE NOT EXISTS (SELECT * FROM segcorpus WHERE segcorpus.segcorpus=%s) "
        #sql = "INSERT INTO segcorpus (segtextId,segtextCode,segcorpus,textId,createTime)VALUES('%d','%d','%s','%d','%s')"
        # 数据库中的已存在的分词语料
        slectsq2 = "SELECT count(*) FROM segcorpus where segcorpus.segcorpus='%s'"

        i = 1
        insertdData = []
        for item, value in corpus.items():
            # for i in range(1, len(corpus)):
            #print(' '.join(corpus[i]), type(corpus[i]))
            item = int(item)
            value = tokenrow(value)
            value = ' '.join(value)
            #print("value", value)
            count = cursor.execute(slectsq2 % value)
            cc = cursor.fetchall()
            theTime = datetime.datetime.now()
            if cc[0][0] > 0:
                continue
            else:

                if rr[0][0]:
                    data = (rr[0][0] + i, item, value, rr[0][0] + i, theTime, value)
                else:
                    data = (i, item, value,  i, theTime, value)
                insertdData.append(data)
                i = i + 1
        try:
            # 判断数据表中是否已经存在如果存在则不插入
            cursor.executemany(sql, insertdData)
            connect.commit()
            end = time.time()
            print(end - start)
        except Exception as e:
            flag = False
            connect.rollback()
            print("eror", e)
            # sys.stdout.write(e)
            #raise Exception("read error %s" % Exception)
        print('成功插入', cursor.rowcount, '条数据')

    cursor.close()
    connect.close()
    return flag


def trainmodel(trainlog, file):
    corpus = []
    corpus_seg = []
    # 第一步，制作语料（训练数据）
    #print("file  ", file, type(file))
    # bytes.decode(file)
    # file.decode('utf-8')
    #print("file  ", file, type(file))
    load_dict = json.loads(file)
    for item, value in load_dict.items():
        #print(item, value)
        corpus.append(value)

    flag = saveSQL(0, load_dict)
    # 利用列表来训练不要存储本地
    corpuslist = []
    line = []
    for i in range(len(corpus)):
        corpuslist.append(tokenrow(corpus[i]))

    #[['查找', ' ', '基建', '项目', '管理子系统', '下', ' ', '主网', '基建投资', '计划', '管理', '后天', '菜单', '数量', '统计', '\n']]
    # 将分割语料存入库中
    flag = saveSQL(1, load_dict)
    # step2 训练模型
    if trainlog:  # 全量训练,此时需要从数据库读出
        if os.path.exists(os.path.join(SAVE_MODEL, 'w2v.mod')):
            # 取出数据库中所有的语料
            model, corpuslist = trainall()
            model.save(os.path.join(SAVE_MODEL, 'w2v.mod'))
            corpuslib = corpuslist
        else:  # 此时没有模型，比如第一次
            model = Word2Vec(corpuslist, size=150, min_count=1, iter=400)
            # print(model.wv.vocab)
            model.save(os.path.join(SAVE_MODEL, 'w2v.mod'))
            corpuslib = corpuslist
    else:  # 增量
        # 取出模型地址
        if os.path.exists(os.path.join(SAVE_MODEL, 'w2v.mod')):
            # 存在该模型
            model_path = os.path.join(SAVE_MODEL, 'w2v.mod')
            model = retrain(corpuslist, model_path, model_path)
            # 此时语料应该更新
            corpus_seg = []
            word_dict = fetcorpus()  # 数据库已有的语料
            for key, value in word_dict.items():
                #         print(key, value)
                corpus_seg.append(tokenrow(key))
            for i in range(len(corpuslist)):
                if corpuslist[i] in corpus_seg:
                    continue
                else:
                    corpus_seg.append(corpuslist[i])

            corpuslib = corpus_seg
            model.save(os.path.join(SAVE_MODEL, 'w2v.mod'))
        else:  # 此时还没有模型
            model = Word2Vec(corpuslist, size=150, min_count=1, iter=400)
            model.save(os.path.join(SAVE_MODEL, 'w2v.mod'))
            corpuslib = corpuslist
    return model, corpuslib
# 增量训练


def retrain(corpus, old_model_file, new_model_file):
    #sents = LineSentence("corpus.txt")
    model = Word2Vec.load(old_model_file)
    model.build_vocab(corpus, update=True)
    model.train(corpus, total_examples=model.corpus_count + len(corpus), epochs=model.iter)
    model.save(new_model_file)
    return model


def fetcorpus():
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

#     for key, value in word_dict.items():
#         print(key, value)
#         re.append(tokenrow(key))

    return word_dict
# 全量训练


def trainall():
    # 链接数据库
    result = []
    word_dict = fetcorpus()
    for key, value in word_dict.items():
        result.append(tokenrow(key))
    new_model = Word2Vec(result, size=150, min_count=1, iter=400)
    # print(new_model.wv.vocab)
    new_model.save(os.path.join(SAVE_MODEL, 'w2v.mod'))
    return new_model, result


# retrain(corpus, "./model/w2v.mod", "./model/w2v.mod")
# 本应用的配置项，设置允许上传的文件类型
ALLOWED_EXTENSIONS = set(['json'])
# Flask配置项，设置请求内容的大小限制，即限制了上传文件的大小
# MAX_CONTENT_LENGTH = 5 * 1024 * 1024
# 本应用的配置项，设置上传文件存放的目录
UPLOAD_FOLDER = './uploads'

# 模型保存
SAVE_MODEL = './model'
# 检查文件是否合法


def allowed_file(filename):
    # 判断文件的拓展名是否在 ALLOWED_EXTENSIONS中
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']  # 获取文件对象
        fread = file.read().decode('utf-8')
        #print("d", file.read(), type(fread))
        # 转成字符串
        #print("e", json.loads(fread))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(UPLOAD_FOLDER, filename))
            # 判断是否进行增量训练还是全量训练 True为全量，False为增量训练
            #model, corpus = trainmodel(False, os.path.join(UPLOAD_FOLDER, filename))
            model, corpus = trainmodel(args.retrain, fread)
            print('Upload Successfully')
            return 'training is over!'
        else:
            return 'Upload Failed'
    else:
        return render_template('upload.html')
# 部署接口


@app.route('/run', methods=['GET', 'POST'])
def rum_model():
    if request.method == 'POST':
        # 转成字符串
        #print("e", json.loads(fread))
        text = request.values.get('text')
        print(text)
        result = runmodel(text)
        print(result)
        return json.dumps(result, ensure_ascii=False, cls=MyEncoder)
    else:
        return render_template('upload.html')


def runmodel(text):
    # 获取新的语料
    #candidates = []
    word_dict = fetcorpus()
#     for key, value in word_dict.items():
#         print(key, value)
#         candidates.append(tokenrow(key))
    if os.path.exists(os.path.join(SAVE_MODEL, 'w2v.mod')):
        print(os.path.join(SAVE_MODEL, 'w2v.mod'))
        model_w2v = Word2Vec.load(os.path.join(SAVE_MODEL, 'w2v.mod'))
    else:
        print("该路径不存在")
    words = list(jieba.cut(text.strip()))
    flag = False
    word = []
    for w in words:
        if w not in model_w2v.wv.vocab:
            print("input word %s not in dict. skip this turn" % w)
        else:
            word.append(w)
    # 相似
    res = []
    result = []

    #print("word", word)
    if word:
        for key, value in word_dict.items():
            candidate = tokenrow(key)
        # for candidate in candidates:
            #print("candidate", candidate)
            for c in candidate:
                if c not in model_w2v.wv.vocab:
                    #print("candidate word %s not in dict. skip this turn" % c)
                    flag = True
            if flag:
                break
            if candidate:
                score = model_w2v.n_similarity(word, candidate)
                resultInfo = {'id': value, "score": score, "text": " ".join(candidate)}
                res.append(resultInfo)

        res.sort(key=lambda x: x['score'], reverse=True)

        k = 5  # top3

        for i in range(k):
            # if res[i]['score'] > 0.60:  # 认为相似
            if i < len(res):
                dict_temp = {"textCode": res[i]['id'], "text": res[i]['text'].replace(" ", ""), "score": res[i]['score']}

                result.append(dict_temp)

    else:
        print("没有相似文本,可能你寻找的表为：")
        result = []
    return result


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--retrain', default=True, type=bool)
    args = parser.parse_args()

    app.debug = args.debug
    app.run('0.0.0.0', args.port, args)


if __name__ == '__main__':

    main()
    #     text = "查找关于中国南方电网公司子系统下的移动终端版本管理从今天起的菜单数量统计"
    #     model, corpus = trainall()
    #     result = runmodel(model, corpus, text)
    #     print(result)
    #     tecp = {0: "昨天中国南网有多少人在线？"}
    #     saveSQL(1, tecp)
    #     with open("./data/newdatasets.json", 'r', encoding='utf-8') as f:
    #         load_f = json.load(f)
    #     saveSQL(0, load_f)
#     result = fetcorpus()
#     print(result)
