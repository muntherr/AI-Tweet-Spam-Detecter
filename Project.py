from nltk.corpus.reader.chasen import test
import pandas as pd
import re
from pandas import DataFrame
import sklearn
import nltk
import urllib3
import urllib.parse
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import opinion_lexicon
from sklearn import tree
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from wordcloud  import WordCloud
from spam.spamhaus import SpamHausChecker
from urllib.parse import urlparse
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from stemming.porter2 import stem




def NanValueHandling(data):
    #Handiling NAN Values
    #We Used mean and mode techniques for handling null Values
    #Id  Tweet  rating  usefulCount  date  reviewerID.1  firstCount  reviewCount  Filtered
    data['following'] = data['following'].replace(np.NAN, data['following'].mean())
    data['followers'] = data['followers'].replace(np.NAN, data['followers'].mean())
    data["actions"] = data["actions"].replace(np.NAN, data['actions'].mean())
    mode = data[data["location"]!="?"]["location"].mode()[0]
    data['location'] = data['location'].replace(np.NAN, mode)

def noOfMentions(data):
    data['Mentions'] = ""
    data['Mentions'].fillna(0)
    regex = r"@"
    for i in range(0, len(data)):
        Mentions = re.findall(regex, data.Tweet.iloc[i])
        #print(Mentions,"Len", len(Mentions))
        data.loc[i, 'Mentions'] = len(Mentions)
        #data.to_csv("ProcessedData.csv", index=False)

def noOfChar(data):
    data['Number of characters']=""
    data['Number of characters'].fillna(0)
    for i in range(0, len(data)):
        charcount = len(data.Tweet.iloc[i])
        white_spacecout=(data.Tweet.iloc[i].count(' '))
        NumOfChar=charcount -white_spacecout
        data.loc[i, 'Number of characters'] = NumOfChar
        #print(NumOfChar,"Len", len(data.Tweet.iloc[i]))
       # data.to_csv("ProcessedData.csv", index=False)

def noOfURLs(data):
    data['URLs'] = ""
    data['URLs'].fillna(0)
    # data.to_csv("ProcessedData.csv", index= False)
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    for i in range(0,len(data)):
        url = re.findall(regex, data.Tweet.iloc[i])
        # print(url,"Len", len(url))
        data.loc[i, 'URLs'] = len(url)

def noOfHashtags(data):
    data['Hashtags'] = ""
    data['Hashtags'].fillna(0)
    regex = r"#"
    for i in range(0, len(data)):
        hashtags = re.findall(regex, data.Tweet.iloc[i])
        #print(hashtags,"Len", len(hashtags))
        data.loc[i, 'Hashtags'] = len(hashtags)

def noOfWords(data):
    data['Words'] = ""
    data['Words'].fillna(0)
    for i in range(0, len(data)):
        result = len(data.Tweet.iloc[i].split())
        #print(result)
        data.loc[i, 'Words'] = result

def hashtags_words_ratio(data):
    df = pd.read_csv('ProcessedData.csv')
    data['Hashtags_to_Words_Ratio']= ""
    data['Hashtags_to_Words_Ratio'].fillna(0)
    for i in range(0, len(df)):
        ratio= (np.double)(df.Hashtags.iloc[i])/(np.double)(df.Words.iloc[i])
        data.loc[i, 'Hashtags_to_Words_Ratio'] = ratio

# Remove Stop Words From tweets and convert them to lower case
def RemoveStopWords(data):
    d = [line.strip() for line in data['Tweet']] 
    texts = [[word.lower() for word in text.split()] for text in d]
    tokens_without_sw=[]
    sw = set(stopwords.words('english'))
    for i in range(0,len(texts)):#
        words_list = []
        for word in texts[i]:
        # print(word)
            if not (word in sw):
                words_list.append(word)
        tokens_without_sw.append(words_list)
    texts = tokens_without_sw
    s = pd.Series(texts)
    for i in range(0,len(s)):
        data.loc[i, 'Tweet'] = " ".join((s)[i])
    return data
#Remove Special Chars from the tweets
def RemoveSpecialChar(data):
    out_list = [re.sub(r'[^a-zA-Z0-9]', ' ', word) for word in data['Tweet']]
    print(out_list)
    s = pd.Series(out_list)
    for i in range(0,len(s)):
        data.loc[i, 'Tweet'] = "".join((s)[i])

def Stemming(data):
    # We need to split each words in each sentences to do stimming on each words
    lancaster=LancasterStemmer()
    d = [line.strip() for line in data['Tweet']] 
    texts = [[stem(word) for word in text.split()] for text in d]
    print(len(data))
    s = pd.Series(texts)
    for i in range(0,len(s)):
        data.loc[i, 'Tweet'] = " ".join((s)[i])

  
# pip install spam-blocklists 
def CheckURL(data):
    list =[]
    data['IsSpamUrl'] = ""
    data['IsSpamUrl'].fillna(0)
    checker = SpamHausChecker()
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    for i in range(0,len(data)):
        url =re.findall(regex, data.Tweet.iloc[i])
        list.append(url)
        #
   
   # print("column[0] = {}".format(list[0]))
    print(list)  

# In this Function we need to normalize a numerical values from data set
def Normalization(data):
    x_data = data[['following','followers','actions']]
    x_data = x_data.apply(lambda x: (x - x.min(axis=0) ) /x.max(axis=0)-x.min(axis=0))
    print(data['following'] )
    for i in range(0,len(data)):
        data.loc[i, 'following'] = x_data['following'][i]
        data.loc[i, 'followers'] = x_data['followers'][i]
        data.loc[i, 'actions'] = x_data['actions'][i]


    #data['following'] = ([x_data.following])
    print(x_data['following'] )

    # for i in range(0, len(data)):
    #     #print(result)
    #     data.loc[i, 'following'] = x_data['following']
    #     # data.loc[i, 'followers'] = x_data.followers
    #     # data.loc[i, 'actions'] = x_data.actions

   
   



#Import training file and preparing data
data = pd.read_csv('train.csv',low_memory=False)
#print(data)
# data = RemoveStopWords(data)

# RemoveSpecialChar(data)
#Stemming(data)
#data.to_csv('ProcessedData.csv',index=False)



NanValueHandling(data)
Normalization(data)
# noOfURLs(data)
# noOfMentions(data)
# noOfChar(data)
# noOfWords(data)
# noOfHashtags(data)
# hashtags_words_ratio(data)

#CheckURL(data)

#print(data['location'].mode())
data.to_csv("ProcessedData.csv", index=False)




#noOfURLs(data)
# Import training file and preparing data
# df = pd.read_csv('train.csv')
# df.to_csv('ProcessedData.csv',index=False)
# data = pd.read_csv('ProcessedData.csv')
# print(data)








# 