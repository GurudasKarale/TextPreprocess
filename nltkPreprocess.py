import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag,ne_chunk
import spacy
nlp=spacy.load('en_core_web_sm')

strr = "There are 3 balls in this bag, and 12 in the other one."

#convert into lower case
lower=strr.lower()
removNumber=re.sub(r'\d+', '', strr)

#split
splt = re.split("\s", strr, 4)
print(splt)
#remove html  tags
text='<a href="foo.com" class="bar">I Want This <b>text!</b></a>'
reg = re.compile(r'<[^>]+>')
removeHTML=reg.sub('', text)
print(removeHTML)

#remove punctuation
punct = str.maketrans('', '', string.punctuation)
removPunct=strr.translate(punct)

#remove stopwords

stop_words = set(stopwords.words("english"))
tokenize = word_tokenize(strr)

removeStop=[]
for word in tokenize:
    if word not in stop_words:
        removeStop.append(word)
#print(removeStop)

#Stemmming
stemmer = PorterStemmer()
stm=[]
for word in tokenize:
    stm.append(stemmer.stem(word))
#print(stm)

#lemmatization
lemmatizer = WordNetLemmatizer()
lematize=[]
for word in tokenize:
    lematize.append(lemmatizer.lemmatize(word, pos ='v'))
#print(lematize)

#pos tagging
tagg=pos_tag(tokenize)
print(tagg)

#chunking
grammar = "NP: {<DT>?<JJ>*<NN>}"
Parser = nltk.RegexpParser(grammar)
tree = Parser.parse(tagg)
for parsetree in tree.subtrees():
    print(parsetree)

#Named entity recognition
chunkk=ne_chunk(tagg)

#dependency parsing
text="There are 3 balls in this bag, and 12 in the other one."

for token in nlp(text):
    print(token.text,'=>',token.dep_,'=>',token.head.text)
