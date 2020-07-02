import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import string

data=pd.read_csv("C:/Users/Mohit K/Desktop/labeledTrainData.tsv",delimiter = '\t')
data.head()

data = data.drop(columns="id")
data=data.iloc[0:10,:]
sentences=[]
y=[]
for i in range(len(data)):
    sentences.append(data['review'][i])

for i in range(len(data)):
    y.append([data['review'][i],data['sentiment'][i]])



def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = preProcess(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def preProcess(sentence):

    stop_words = set(stopwords.words('english'))
    words = re.sub("[^\w]", " ",  sentence).split()

    table = str.maketrans('', '', string.punctuation)
    cleaned_text=[w.translate(table) for w in words]
    cleaned_text = [word for word in cleaned_text if word.isalpha()]

    cleaned_text = [w.lower() for w in words if w not in stop_words ]
    cleaned_text = [word for word in cleaned_text if len(word) > 1]

    return cleaned_text

z=tokenize(sentences)  #vocabulary


def separate(y):

        sept=dict()                              #initialize dictionary
        for i in range(len(y)):
            row=y[i]                   #take the ith row vector
            label=row[-1]                          #take last element of row vector which is a binary label(0,1)
            if(label not in sept):
                sept[label]=list()                 #store labels in the dictionary
            sept[label].append(row)                #store row vectors corresponding to labels in the dictionary

        return sept

q=separate(y)        #dictionary


def bag(q):

    bagg=dict()
    for key,value in q.items():
        bagg[key]=vectorbag(value)

    return bagg

def vectorbag(value):

    words=[]
    for sent in value:

        words.append(preProcess(str(sent)))
    return words

def new_bag(q,z):
    words=bag(q)
    bog=dict()
    for key,value in words.items():

        bog[key]=newVector(value,z)

    return bog

def newVector(value,z):

    f=[]
    for i in range(len(value)):
        row=value[i]
        zr=np.zeros(len(z))
        for j in range(len(row)):

            for k,word in enumerate(z):
                if word==row[j]:
                    zr[k]+=1
        f.append(zr)

    return f

m=new_bag(q,z)
print(m)
