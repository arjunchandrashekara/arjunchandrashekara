import os
import nltk
import nltk.corpus
import numpy as np
import pandas as pd
nltk.download("punkt")
with open("E://excelr data//textmining//modi.txt","r+") as file:
    data= file.read().replace("\n","")

words= data.split()
print(words[:100])
len(words)

import re
words=re.split(r"\W+",data)

import string
print(string.punctuation)
table= str.maketrans("","",string.punctuation)

stripped=[w.translate(table) for w in words]

words_lower= [word.lower() for word in words]

words_upper= [word.upper() for word in words]

from nltk import sent_tokenize
sentences= sent_tokenize(data)

from nltk import word_tokenize
token=word_tokenize(data)

words=[word for word in token if word.isalpha()]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
from nltk.stem.porter import PorterStemmer


porter = PorterStemmer()
words=[word for word in words if not word in stop_words]
stemmed = [porter.stem(word) for word in words]

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(stemmed))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
str1 = ''.join(stemmed)
type(str1)


#bigram and trigram
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
finder= BigramCollocationFinder.from_words(data)

from textblob import TextBlob
def senti(x):
    return TextBlob(x).sentiment

df= senti(str1)

pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

df=pd.DataFrame(sentences)
str=sentiment_analyzer_scores(df)


from nltk import ngrams

n = 2
bigrams = ngrams(data, n)

bigrams

for grams in bigrams:
  print(grams)
  