#读取zip文件写入pandas
import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

#去数字以及标点符号
import re
def remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

#预处理Amazon Clothing_Shoes_and_Jewelry数据集 代入LDA主题模型检验分类效果
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
df = getDF('/Users/xiaoqiulu/Desktop/coling/reviews_Clothing_Shoes_and_Jewelry_5.json.gz')
raw_text = df["reviewText"].tolist() #total texts：278677
file = open("amazon_clothing.txt","w")#所有原始review写入txt文件
for x in raw_text:
    file.write(x+"\n")
raw_text = [x.lower() for x in raw_text]#英文单词全部小写化
raw_text = [remove(x) for x in raw_text]#去除字符串中的数字及标点符号
tokenized_text = [sentence.split(" ") for sentence in raw_text]#按照空格分词
stopwords = [line.strip() for line in open('/Users/xiaoqiulu/Desktop/coling/stopwords.txt',encoding='UTF-8').readlines()]#停用词
processed_text = []
for sentence in tokenized_text:
    sentence = [word for word in sentence if word not in stopwords]
    sentence = sentence[0:-1]
    processed_text.append(sentence)
id2word = corpora.Dictionary(processed_text)
corpus = [id2word.doc2bow(text) for text in processed_text]
lda_clothing = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=15)
for topic in lda_clothing.print_topics(num_words=10):
    print(topic)
bow = corpus[0]
print(lda_clothing.get_document_topics(bow))

import numpy as np
import logging
import pyLDAvis.gensim
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
lda_max = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=9)
data = pyLDAvis.gensim.prepare(lda_max, corpus, id2word)
pyLDAvis.show(data)
