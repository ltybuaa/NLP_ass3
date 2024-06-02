import math
import jieba
import os  # 用于处理文件路径
import re
import sys
import random
import multiprocessing
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
def read_novel(path_in, path_out):  # 读取语料内容
    content = []
    names = os.listdir(path_in)
    for name in names:
        novel_name = path_in + '\\' + name
        fenci_name = path_out + '\\' + name
        for line in open(novel_name, 'r', encoding='ANSI'):
            line.strip('\n')
            line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
            line = content_deal(line)
            con = jieba.cut(line, cut_all=False) # 结巴分词
            # content.append(con)
            content.append(" ".join(con))
        with open(fenci_name, "w", encoding='utf-8') as f:
            f.writelines(content)
    return content, names


def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '她', '他', '你', '我', '它', '这'] #去掉其中的一些无意义的词语
    for a in ad:
        content = content.replace(a, '')
    return content


if __name__ == '__main__':   ##
    [data_txt, files] = read_novel("datasets", "output")
    #[data_txt, files] = read_novel("倚天屠龙记", "output")
    #model = Word2Vec(data_txt, vector_size=400, window=5, min_count=5, epochs=200, workers=multiprocessing.cpu_count())
    test_name = ['张无忌', '乔峰', '郭靖', '杨过', '令狐冲', '韦小宝']
    #test_name = ['张无忌']
    test_menpai = ['明教', '逍遥派', '少林', '全真教', '华山派', '少林']
    for i in range(0, 5):
        name = "output/" + files[i]
        print(name)
        model = Word2Vec(sentences=LineSentence(name), hs=1, min_count=10, window=5, vector_size=200, sg=0, epochs=200)
        for result in model.wv.similar_by_word(test_name[i], topn=10):
            print(result[0], result[1])
        for result in model.wv.similar_by_word(test_menpai[i], topn=10):
            print(result[0], result[1])


