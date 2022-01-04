import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

from pandas import DataFrame
data = pd.read_csv("Test.csv", low_memory=False)
print(type(data['reviewContent']))

#print(data)
print("-------------------------------------------------------------\n")
#print(data['reviewContent'])
print("-------------------------------------------------------------\n")


list_of_tokens =[]
for column in data['reviewContent']:
   text_tokens = word_tokenize(column)
  # print(text_tokens)
   list_of_tokens.append(text_tokens)
len(list_of_tokens)
print("DDDDDDDDd")
print(list_of_tokens)


def stopWordRemove(list_of_tokens):
  tokens_without_sw=[]
  sw = set(stopwords.words('english'))
  for i in range(0,len(list_of_tokens)):
    words_list = []
    for word in list_of_tokens[i]:
      # print(word)
      if not (word in sw):
        words_list.append( word)
    tokens_without_sw.append(words_list)
  return tokens_without_sw

tokens_without_sw = stopWordRemove(list_of_tokens)
print("--------------------------------------,-----------------------\n")

print(tokens_without_sw)

def lemmatization(tokens_without_sw):
  lmtz_list=[]
  lemmatizer = WordNetLemmatizer()
  for words_list in tokens_without_sw:
    lmtz_words_list = []
    for word in words_list:
      lmtz_words_list.append(lemmatizer.lemmatize(word))
    lmtz_list.append(lmtz_words_list)
  return  lmtz_list
lmtz_list=lemmatization(tokens_without_sw)
lmtz_list #without sw and lemmatized

corpus = []
#join the words to use it in vectorizer

print(lmtz_list)
for list_of_words in lmtz_list:
  corpus.append((" ").join(list_of_words))
print(corpus)

vectorizer = CountVectorizer(max_df=0.90, min_df=0.10)

X = vectorizer.fit_transform(corpus)
print(X)
print("-------------------------------------------------------------\n")
print(vectorizer.get_feature_names())
print("-------------------------------------------------------------\n")
X = X.toarray()
print(X)
