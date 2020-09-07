#Preparation
#_________________Make LDA model_________________
from gensim import corpora, models, similarities
def ldaTest(file):
    raw_text = open(file).readlines()
    tokenized_text = [sentence.split(" ") for sentence in raw_text]
    stopwords = [line.strip() for line in open('stopwords_English.rtf',encoding='UTF-8').readlines()]
    processed_text = []
    for sentence in tokenized_text:
        sentence = [word for word in sentence if word not in stopwords]
        sentence = sentence[0:-1]
        processed_text.append(sentence)
    id2word = corpora.Dictionary(processed_text)
    corpus = [id2word.doc2bow(text) for text in processed_text]
    lda_model = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=8)
    return lda_model

#_______________Function : Process the original textData into Isometric sequence_______________
def wordProcess(text):
    data_index=[]#word-index
    data_temp=[]
    data_processed=[]
    data=[]#sentence after processing
    data_out=[]
    for each in text:
        data_processed.append(each.split(" "))

    stopwords = [line.strip() for line in open('stopwords_English.rtf',encoding='UTF-8').readlines()]
    for row in data_processed:
        for word in row:
            if word not in stopwords:
                data_out.append(word)
        data.append(data_out)
        data_out=[]
    
    word2index = np.load("word2index.npy",allow_pickle=True).item()#Dictionary Form
    for each in data:
        for word in each:
            if word in word2index.keys():
                data_temp.append(word2index[word])
            else:
                word = 0
        data_index.append(weibo_temp)
        data_temp=[]
    #Isometric sequence 90_length MAY be better
    for i in range(len(data_index)):
        if len(data_index[i])>90:
            while (len(data_index[i])-90>0):
                data_index[i].pop()
        if len(data_index[i])==90:
            data_index[i]=data_index[i]
        else:
            m=90-len(data_index[i])
            for j in range(m):
                data_index[i].append(0)
    for i in range(len(data_index)):
        data_index[i] = list(map(int,data_index[i]))
    data_index=  torch.LongTensor(data_index)
    return data_index

#_____________Function for sampling with Gumbel-softmax_________
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)

#____________Function:removal of digits & punctions______________
import re
def remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

#________________Preprocess: Lowercase + removing digits & punctions & stopwords____________
def wordProcess(raw_text):
    raw_text = [x.lower() for x in raw_text]#Lowercase
    raw_text = [remove(x) for x in raw_text]#Remove digits & punctions
    tokenized_text = [sentence.split(" ") for sentence in raw_text]#Split by blank space
    stopwords = [line.strip() for line in open('/Users/Desktop/coling/stopwords.txt',encoding='UTF-8').readlines()]#停用词
    processed_text = []
    for sentence in tokenized_text:
        sentence = [word for word in sentence if word not in stopwords]
        sentence = sentence[0:-1]
        processed_text.append(sentence)
    return processed_text

#________________Word2Vec Model______________
from gensim.models import Word2Vec
raw_data = open("amazon_clothing.txt","r").readlines()#original dataset
processed_data = wordProcess(raw_data)
embed_dim=32
w2vModel = Word2Vec(list(processed_data), min_count=1, size=embed_dim)
print(len(w2vModel.wv.vocab))#153128

#_______________Make Embedding_Matrix for each text______________
"""
while mapping the word to vector,
the padding of 0 would be mapping to np.array(0,embedding_dim)
"""
import numpy as np
embed_dim = 32
words = list(w2vModel.wv.vocab)#All key words in word2vecModel
vocab_length=len(words)
embedding_matrix = []
embedding_matrix.append(np.zeros(embed_dim))
for word in words:
    embedding_matrix.append(model[word])
embedding_matrix = np.array(embedding_matrix)
#________________Formalization for Embedding Matrix_____________
max_val=np.max(embedding_matrix)
min_val=np.min(embedding_matrix)
embedding_matrix = (embedding_matrix - min_val) / (max_val - min_val)
for i in range(embed_dim):
    embedding_matrix[0][i] = 0
word2index = dict(zip(words,range(1,len(words)+1)))
np.save("word2index",word2index)
np.save("embedding_matrix",embedding_matrix)

#__________Function: Transform text data into BOW_matrix_________
def bOW(text):
    data = wordProcess(text)
    BOW=[]#record BOW_Matrix
    for sentence in data:
        BOW_matrix = np.zeros(vocab_length+1)#record BOW_Matrix for each text
        for word in sentence:
            if word in word2index.keys():
                index = word2index[word]
                BOW_matrix[index]+=1
        BOW.append(BOW_matrix)
    return np.array(BOW)

#_____________change words to the form of index, for preparation of embedding_matrix_________
from collections import Counter
from gensim.models import Word2Vec
def vocab2index(text):
    stopwords = [line.strip() for line in open('stop_words.txt',encoding='UTF-8').readlines()]
    with open(text,'r') as f:
    content = f.read()
    words = content.split("")#words flow without processing
    words_stream=[]#processed words flow
    for word in words:
        if word not in stopwords:
            words_stream.append(word)
    counts = Counter(words_stream)
    vocab = sorted(counts, key=counts.get, reverse=True)
    #dictionary
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)
    return vocab_to_int

#_______________Make txt，where origin_text and fake_text are split by "/////""_______________
import random
file = open("InfomaxData.txt","w")#form: origin_text/////fake_text
origin_data = open("amazon_clothing.txt","r").readlines()#original text
fake_data = open("amazon_clothing.txt","r").readlines()#preparation for fake data
random.shuffle(fake_data)#shuffle the original text for fake data
for i in range(len(origin_data)):
    origin_data[i]=origin_data[i].strip("\n")#delete newline break
    fake_data[i]=fake_data[i].strip("\n")
    file.write(origin_data[i]+"/////"+fake_data[i]+"\n")

#__________Make the dataset for dataloading module________
from torch.utils.data import Dataset
class InfomaxDataSet(Dataset):
    def __init__(self,txt_path):
        file = open(txt_path, 'r')
        data=[]#form: original_text fake_text
        for line in file:
            line = line.rstrip()#delete blank space
            words = line.split("/////")#split words by the specific characters
            data.append((words[0],words[1]))
        self.data = data
    def __getitem__(self, index):
        origin_text,fake_text = self.data[index]
        return origin_text,fake_text
    def __len__(self):
        return len(self.data)
