import pickle
import re
import time
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from stemming.porter2 import stem

data = pd.read_csv("train.csv")
# Import training file and start preparing data

def NanValueHandling(data):
    # Handiling NAN Values
    # We Used mean and mode techniques for handling null Values
    data['following'] = data['following'].replace(np.NAN, data['following'].mean())
    data['followers'] = data['followers'].replace(np.NAN, data['followers'].mean())
    data["actions"] = data["actions"].replace(np.NAN, data['actions'].mean())
    mode = data[data["location"] != "?"]["location"].mode()[0]
    data['location'] = data['location'].replace(np.NAN, mode)

def noOfURLs(data):
    data['URLs'] = ""
    data['URLs'].fillna(0)
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    for i in range(0, len(data)):
        url = re.findall(regex, data.Tweet.iloc[i])
        data.loc[i, 'URLs'] = len(url)

def noOfHashtags(data):
    data['Hashtags'] = ""
    data['Hashtags'].fillna(0)
    regex = r"#"
    for i in range(0, len(data)):
        hashtags = re.findall(regex, data.Tweet.iloc[i])
        data.loc[i, 'Hashtags'] = len(hashtags)

def noOfMentions(data):
    data['Mentions'] = ""
    data['Mentions'].fillna(0)
    regex = r"@"
    for i in range(0, len(data)):
        Mentions = re.findall(regex, data.Tweet.iloc[i])
        data.loc[i, 'Mentions'] = len(Mentions)

def noOfWords(data):
    data['Words'] = ""
    data['Words'].fillna(0)
    for i in range(0, len(data)):
        result = len(data.Tweet.iloc[i].split())
        data.loc[i, 'Words'] = result

def hashtags_words_ratio(data):
    df = pd.read_csv('ProcessedData.csv')
    data['Hashtags_to_Words_Ratio'] = ""
    data['Hashtags_to_Words_Ratio'].fillna(0)
    for i in range(0, len(df)):
        ratio = df.Hashtags.iloc[i] / df.Words.iloc[i]
        data.loc[i, 'Hashtags_to_Words_Ratio'] = ratio

def links_words_ratio(data):
    df = pd.read_csv('ProcessedData.csv')
    data['Links_to_Words_Ratio'] = ""
    data['Links_to_Words_Ratio'].fillna(0)
    for i in range(0, len(df)):
        ratio = df.URLs.iloc[i] / df.Words.iloc[i]
        data.loc[i, 'Links_to_Words_Ratio'] = ratio

def noOfChars(data):
    data['Characters'] = ""
    data['Characters'].fillna(0)
    for i in range(0, len(data)):
        charcount = len(data.Tweet.iloc[i])
        white_spacecout = (data.Tweet.iloc[i].count(' '))
        NumOfChar = charcount - white_spacecout
        data.loc[i, 'Characters'] = NumOfChar

def caps_words_ratio(data):
    df = pd.read_csv('ProcessedData.csv')
    data['Caps_to_Words_Ratio'] = ""
    data['Caps_to_Words_Ratio'].fillna(0)
    for i in range(0, len(data)):
        token = data.Tweet.iloc[i].split(" ")
        count = 0
        for j in token:
            if (len(j) > 1 and j.isupper() == True):
                count += 1
        ratio = count / df.Words.iloc[i]
        data.loc[i, 'Caps_to_Words_Ratio'] = ratio

def RemoveStopWords(data):
    d = [line.strip() for line in data['Tweet']]
    texts = [[word.lower() for word in text.split()] for text in d]
    tokens_without_sw = []
    sw = set(stopwords.words('english'))
    for i in range(0, len(texts)):  #
        words_list = []
        for word in texts[i]:
            if not (word in sw):
                words_list.append(word)
        tokens_without_sw.append(words_list)
    texts = tokens_without_sw
    s = pd.Series(texts)
    for i in range(0, len(s)):
        data.loc[i, 'Tweet'] = " ".join((s)[i])
    return data

def RemoveSpecialChar(data):
    out_list = [re.sub(r'[^a-zA-Z0-9]', ' ', word) for word in data['Tweet']]
    s = pd.Series(out_list)
    for i in range(0, len(s)):
        data.loc[i, 'Tweet'] = "".join((s)[i])

def Stemming(data):
    # We need to split each words in each sentences to do stimming on each words
    lancaster = LancasterStemmer()
    d = [line.strip() for line in data['Tweet']]
    texts = [[stem(word) for word in text.split()] for text in d]
    s = pd.Series(texts)
    for i in range(0, len(s)):
        data.loc[i, 'Tweet'] = " ".join((s)[i])

def Normalization(data):
    scalar = MinMaxScaler()
    x_data = data[
        ['following', 'followers', 'actions', 'URLs', 'location', 'Hashtags', 'Mentions', 'Words', 'Characters',
         'Caps_to_Words_Ratio', 'Links_to_Words_Ratio', 'Hashtags_to_Words_Ratio']]
    data[['following', 'followers', 'actions', 'URLs', 'location', 'Hashtags', 'Mentions', 'Words', 'Characters',
          'Caps_to_Words_Ratio', 'Links_to_Words_Ratio', 'Hashtags_to_Words_Ratio']] = scalar.fit_transform(
        data[['following', 'followers', 'actions', 'URLs', 'location', 'Hashtags', 'Mentions', 'Words', 'Characters',
              'Caps_to_Words_Ratio', 'Links_to_Words_Ratio', 'Hashtags_to_Words_Ratio']])
    x_data = pd.DataFrame(x_data)
    for i in range(0, len(data)):
        data.loc[i, 'following'] = (data['following'][i])
        data.loc[i, 'followers'] = (data['followers'][i])
        data.loc[i, 'actions'] = data['actions'][i]
        data.loc[i, 'URLs'] = data['URLs'][i]
        data.loc[i, 'Hashtags'] = data['Hashtags'][i]
        data.loc[i, 'Mentions'] = data['Mentions'][i]
        data.loc[i, 'Words'] = data['Words'][i]
        data.loc[i, 'location'] = data['location'][i]
        data.loc[i, 'Caps_to_Words_Ratio'] = data['Caps_to_Words_Ratio'][i]
        data.loc[i, 'Links_to_Words_Ratio'] = data['Links_to_Words_Ratio'][i]
        data.loc[i, 'Hashtags_to_Words_Ratio'] = data['Hashtags_to_Words_Ratio'][i]
        data.loc[i, 'Characters'] = data['Characters'][i]

#feature selection using low variance filter method
def RemoveLowVarianceFeatures(X):
    threshold = X
    data = pd.read_csv("ProcessedData.csv")
    NanValueHandling(data)
    data['is_retweet'] = data['is_retweet'].replace(np.NAN, data['is_retweet'].mean())
    # saving results in data_scaled data frame and calculatinf variance using .var()
    data_scaled = pd.DataFrame(data)
    data_scaled.var()
    variance = data_scaled.var()
    columns = data.columns
    # saving names of features having various more than threshold value
    features = []
    for i in range(0, len(variance)):
        if variance[i] >= threshold:  # setting threshold as 9%
            features.append(columns[i])
    new_data = data[features]
    return new_data

def Converter(data):
    data.drop('Tweet', inplace=True, axis=1)
    for i in range(0, len(data)):
        if (data.Type.iloc[i] == 'Spam'):
            data.loc[i, 'Type'] = 0
        else:
            data.loc[i, 'Type'] = 1

def locationEncoding(data):
    location = LabelEncoder()
    data['location'] = location.fit_transform(data['location'])

def TweetEncoding(data):
    Tweet= LabelEncoder()
    data['Tweet'] = Tweet.fit_transform(data['Tweet'])

# Descion Tree Classifier
def createTree(data):
    print("Decision Tree")
    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    X_smt, y_smt = smt.fit_resample(X, Y)
    y_smt = np.nan_to_num(np.array(y_smt))
    X_smt = np.nan_to_num(np.array(X_smt))
    X = X_smt
    Y = y_smt
    # # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)  # 80% training and 20% test
    # # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # # Train Decision Tree Classifer
    start = time.time()
    clf = clf.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "tree_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    return X_test, X_train

def loadTree(data):
    # Load from file
    with open("tree_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    score = np.nan_to_num(np.array(score))

    print("Test score: {0:.2f} %".format(100 * score))

    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))

    return Ypredict, Y

# Neural Netowork Classifier
def createNeuralNetwork(data):
    print("Neural network")
    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    X_smt, y_smt = smt.fit_resample(X, Y)
    y_smt = np.nan_to_num(np.array(y_smt))
    X_smt = np.nan_to_num(np.array(X_smt))
    X = X_smt
    Y = y_smt
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), activation='tanh', solver='sgd',
                        max_iter=11000)
    start = time.time()
    mlp.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")

    # saving tree model
    pkl_filename = "NeuralNetwork_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp, file)
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)
    print(metrics.classification_report(y_test, predict_test))

def loadNeuralNetwork(data):
    # Load from file
    with open("NeuralNetwork_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    score = np.nan_to_num(np.array(score))
    # Calculate the accuracy score and predict target values
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))

# Naive Bayes Classifier
def createNaiveBayes(data):
    print("Naive Bayes")
    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))

    X_smt, y_smt = smt.fit_resample(X, Y)
    y_smt = np.nan_to_num(np.array(y_smt))
    X_smt = np.nan_to_num(np.array(X_smt))
    X = X_smt
    Y = y_smt

    # Split dataset into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    model = GaussianNB()
    start = time.time()
    model.fit(X_train, Y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    GaussianNB(priors=None, var_smoothing=1e-09)
    # saving tree model
    pkl_filename = "NaiveBayes_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    Y_pred = model.predict(X_test)
    print(metrics.classification_report(Y_test, Y_pred))

def loadNaiveBayes(data):
    # Load from file
    with open("NaiveBayes_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    score = np.nan_to_num(np.array(score))
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))

def createXGBoost(data):
    print("XGBoost")
    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    seed = 7
    test_size = 0.33
    smt = SMOTETomek(random_state=seed)
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))

    X_smt, y_smt = smt.fit_resample(X, Y)
    y_smt = np.nan_to_num(np.array(y_smt))
    X_smt = np.nan_to_num(np.array(X_smt))
    X = X_smt
    Y = y_smt

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, use_label_encoder=False)
    start = time.time()
    model.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "XGBoost_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))


def loadXGBoost(data):
    print("XGBoost")

    # Load from file
    with open("XGBoost_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))


def createRandomForest(data):
    print("Random Forest")
    df = data
    # To get a high-level view of what the dataset looks like, execute the following command:
    X = df.drop('Type', axis=1)
    Y = df['Type']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)

    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    X_smt, y_smt = smt.fit_resample(X, Y)
    y_smt = np.nan_to_num(np.array(y_smt))
    X_smt = np.nan_to_num(np.array(X_smt))
    X = X_smt
    Y = y_smt
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    start = time.time()
    clf = clf.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "RandomForest_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))


def loadRandomForest(data):
    print("Entered")
    # Load from file
    with open("RandomForest_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    df = data
    X = df.drop('Type', axis=1)
    Y = df['Type']
    Y = np.nan_to_num(np.array(Y))
    X = np.nan_to_num(np.array(X))
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    score = np.nan_to_num(np.array(score))
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))


# fill in a csv file with the final processed dataset and the extracted features
def ProcessedCSVfile(data):
    NanValueHandling(data)
    noOfURLs(data)
    noOfHashtags(data)
    noOfMentions(data)
    noOfWords(data)
    noOfChars(data)
    data.to_csv("ProcessedData.csv", index=False)
    hashtags_words_ratio(data)
    links_words_ratio(data)
    caps_words_ratio(data)
    RemoveStopWords(data)
    RemoveSpecialChar(data)
    Stemming(data)
    data.to_csv("ProcessedData.csv", index=False)
    locationEncoding(data)
    Converter(data)
    Normalization(data)
    data.to_csv("ProcessedData.csv", index=False, float_format='{:f}'.format, encoding='utf-8')

def LVF_File():
    data = RemoveLowVarianceFeatures(0.009)
    data = data.drop('Id', axis=1)
    data.to_csv("LVF.csv", index=False, float_format='{:f}'.format, encoding='utf-8')

def MyCombination():
    data = pd.read_csv("ProcessedData.csv")
    data = data.drop('Id', axis=1)
    data = data.drop('Words', axis=1)
    data = data.drop('Characters', axis=1)
    data = data.drop('following', axis=1)
    data = data.drop('is_retweet', axis=1)
    data = data.drop('actions', axis=1)
    data.to_csv("MyFeatures.csv", index=False, float_format='{:f}'.format, encoding='utf-8')

def TweetText():
    data = pd.read_csv("train.csv")
    Stemming(data)
    TweetEncoding(data)
    data = data.drop('Id', axis=1)
    data = data.drop('following', axis=1)
    data = data.drop('is_retweet', axis=1)
    data = data.drop('actions', axis=1)
    data = data.drop('followers', axis=1)
    data = data.drop('location', axis=1)
    for i in range(0, len(data)):
        if (data.Type.iloc[i] == 'Spam'):
            data.loc[i, 'Type'] = 0
        else:
            data.loc[i, 'Type'] = 1
    data.to_csv("TextOnly.csv", index=False, float_format='{:f}'.format, encoding='utf-8')

def combineFeatures():
    df = pd.read_csv("ProcessedData.csv")
    data = df.drop('Id', axis=1)
    data.to_csv("Combined.csv", index=False)
    t = pd.read_csv("train.csv")
    Stemming(t)
    TweetEncoding(t)
    t.to_csv("Combined.csv", index=False, float_format='{:f}'.format, encoding='utf-8')


def ApplyClassifiers(inFile):
    data = pd.read_csv(inFile)

    createTree(data)
    loadTree(data)

    createNeuralNetwork(data)
    loadNeuralNetwork(data)

    createXGBoost(data)
    loadXGBoost(data)

    createRandomForest(data)
    loadRandomForest(data)

    createNaiveBayes(data)
    loadNaiveBayes(data)

# ApplyClassifiers("TextOnly.csv")

def printMenu():
    print("----------------------------------SPAM TWEET DETECTOR----------------------------------\n")
    print("  This program detects spam tweets among the data-set provided in the file (train.csv)  \n",
        "---------------------------------------------------------------------------------------")
    print("Please specify the features selection method to be used in models training and evaluation:")
    print("1- Low Variance Filter Method")
    print("2- A Random Combination of Features")
    print("3- Tweet Content Only")
    print("4- All Features")
    print("5- EXIT")
    print("\n---------------------------------------------------------------------------------------")

def userInput():
    print("Please choose a number from 1-5 --> ")
    X= int(input())
    while(X!=5):
        if X==1:
            ApplyClassifiers("LVF.csv")
        elif X==2:
            ApplyClassifiers("MyFeatures.csv")
        elif X==3:
            ApplyClassifiers("TextOnly.csv")
        elif X==4:
            ApplyClassifiers("ProcessedData.csv")
        elif X==5:
            print("PROGRAM TERMINATED!")
            exit()
        else:
            print("Please specify the features selection method to be used in models training and evaluation,",
                  "OR ENTER 5 TO EXIT THE PROGRAM")
            print("1- Low Variance Filter Method")
            print("2- A Random Combination of Features")
            print("3- Tweet Content Only")
            print("4- All Features")
            print("5- EXIT")
            print("\n---------------------------------------------------------------------------------------")
            print("Please choose a number from 1-5 -->")
            X= int(input())


printMenu()
userInput()
