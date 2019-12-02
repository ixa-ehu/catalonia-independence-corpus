# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:56:44 2019

@author: EZOTOVA
"""

import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from gensim.models import FastText

#Loading pretrained word2vec model. 
model = FastText.load_fasttext_format('c:/Users/ezotova/Desktop/Python/Dataset_Español/fasttext/cc.es.300.bin')
w2v_vectors = model.wv.vectors 
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab} 
num_features = model.vector_size
vocab = model.wv.vocab.keys()

print("==========================================")

#load datasets 
tweets_t = pd.read_csv('train.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_t = tweets_t.fillna('')

tweets_val = pd.read_csv('val.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_val = tweets_val.fillna('')

tweets_test = pd.read_csv('test.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_test = tweets_test.fillna('')

#concatenate train and validation sets for further cross-validation
frames = [tweets_t, tweets_val]
tweets_train = pd.concat(frames)

#list of tweets cleaned of punctuation, usernames, and normalized (see text_preprocessing.py)
tweets_train_text = list(tweets_train.CLEAN_FULL.values)
tweets_test_text = list(tweets_test.CLEAN_FULL.values)

#converting strings to tokens
text_data_train = []
for line in tweets_train_text: 
    tokens = line.split()
    text_data_train.append(tokens)
    
text_data_test = []
for line in tweets_test_text: 
    tokens = line.split()
    text_data_test.append(tokens)
    
#functions for calculating an average vector per tweet
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    #append a vector to each word
    for word in  words:
        try: 
            nwords = nwords + 1
            v = model[word]
            
            #print(v)
            if np.isnan(v).any():
                print(word, v)
        except KeyError:
            continue
        featureVec = featureVec + model[word]
        #print(word, featureVec)
      
    # dividing the result by number of words to get average
    if nwords != 0:
        featureVec = featureVec/nwords
    return featureVec

def getAvgFeatureVecsOOV(tweets, model, num_features):
    counter = 0
    tweetFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
#    all_tweets = len(tweets)
    for i, tweet in enumerate(tweets):
#       # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(tweets)))
            
        tweetFeatureVecs[counter] = featureVecMethod(tweet, model, num_features)
        counter = counter+1 
        
    return tweetFeatureVecs
    
#calculating the average vector
X_train = getAvgFeatureVecsOOV(text_data_train, model, num_features)
X_test = getAvgFeatureVecsOOV(text_data_test, model, num_features)
print(len(X_train))
print(len(X_test))

#check if there are null vectors 
for i, v in enumerate(X_train):
    if v.any() == 0.0:
        print(i)


for i, v in enumerate(X_test):
    if v.any() == 0.0:
        print(i)


print("=============================================")

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF

#labels from the training set
labels_train = list(tweets_train.LABEL.values)
#Encode the labes to integers
le = preprocessing.LabelEncoder()
le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
labels_int_train = le.transform(labels_train)

#labels from test set
labels_test = list(tweets_test.LABEL.values)
#Encode the labes to integers
le = preprocessing.LabelEncoder()
le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
labels_int_test = le.transform(labels_test)

#x_train, x_test, y_train, y_test = train_test_split(X, labels_int, test_size=0.1, random_state=0)
x_train = X_train
y_train = labels_int_train

x_test = X_test
y_test = labels_int_test

#making a numpy array
x_train = np.array(x_train, dtype=np.float64)
x_test = np.array(x_test, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)
y_test = np.array(y_test, dtype=np.float64)

#number of train and test examples
n_train = len(x_train) 
n_test = len(x_test)
#final matrix for training and testing
x_train = x_train[:n_train]
y_train = y_train[:n_train]
x_test = x_test[:n_test]
y_test = y_test[:n_test]

print("===============================================")
print("Grid search")

#gridsearch for SVM model
from sklearn import svm
from sklearn.model_selection import GridSearchCV

svc = svm.SVC(kernel="rbf")

#define the grids of parameters
parameters = {'C':[1, 10, 100, 300, 500, 700, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.5, 0.75, 1]}

svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1, verbose=True)
clf.fit(x_train, y_train)
#saving the result of grid search
grid_result = pd.DataFrame(clf.cv_results_)
file_name = "Fasttext_Not_Lemma/esp_FT_grid_result.csv"
grid_result.to_csv(file_name, encoding='utf-8', index=False)
#select the best parameters 
df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1] 
C = list(df_grid_first.param_C.values)
gamma = list(df_grid_first.param_gamma.values)

print("===============================================")
print("Cross Validation")

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import io


clf = svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0])

cv_results = cross_validate(clf, X_train, y_train, 
                            cv=10, n_jobs=-1, 
                            scoring='f1_macro')

print('CV RESULTS ', cv_results['test_score'])

y_pred = cross_val_predict(clf, X_train, y_train, 
                           cv=10, n_jobs=-1)

target_names=["AGAINST", "FAVOR", "NEUTRAL"]
cl_report = classification_report(y_train, y_pred, target_names=target_names, digits=4)
cm = confusion_matrix(y_train, y_pred)
print(cm)
print("CROSS VALIDATION")
print(cl_report)

#saving classification reports
report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
file_name = 'CV_report.csv'
report_df.to_csv(file_name, encoding='utf-8', index=False)

#saving predicted and wrong predicted examples
tweets_train['predicted'] = y_pred
tweets_train['true'] = y_train
tweets_train.to_csv('predicted_cross_validation.csv', encoding='utf-8', index=False)
df1 = tweets_train.loc[tweets_train['predicted'] != tweets_train['true']]
wrong_prediction = df1[['TWEET', 'LEMMA_CLEAN', 'LABEL', 'id_str', 'predicted', 'true']]
wrong_prediction.to_csv('wrong_prediction_cross_validation.csv', encoding='utf-8', index=False)


print("================================================")
print("Training and Testing")

clf_train = svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0])
clf_train.fit(X_train, y_train)
y_pred_test = clf_train.predict(X_test)


f_score_macro = f1_score(y_test, y_pred_test, average='macro')
print('F1 macro ', f_score_macro)
f_score_micro = f1_score(y_test, y_pred_test, average='micro')
print('F1 micro ', f_score_micro)
precision = precision_score(y_test, y_pred_test, average='macro') 
print('PRECISION ', precision)
recall = recall_score(y_test, y_pred_test, average='macro')
print('RECALL ', recall)
cm = confusion_matrix(y_test, y_pred_test)
print('CONFUSION MATRIX')
print(cm)
target_names=["AGAINST", "FAVOR", "NEUTRAL"]

#save the classification reports
cl_report_test = classification_report(y_test, y_pred_test, target_names=target_names, digits=4)
report_df_test = pd.read_fwf(io.StringIO(cl_report_test), sep="\s+")
print('Classification report ')
print(report_df_test)
file_name = "test_report.csv"
report_df_test.to_csv(file_name, encoding='utf-8', sep='\t', index=False)

#save the predicted data and wrong predictions
tweets_test['predicted'] = y_pred_test
tweets_test['true'] = y_test
tweets_test.to_csv("predicted_test.csv", sep='\t', encoding='utf-8', index=False)
df2 = tweets_test.loc[tweets_test['predicted'] != tweets_test['true']]
wrong_prediction_test = df2[['TWEET', 'LEMMA_CLEAN', 'LABEL', 'id_str', 'predicted', 'true']]
wrong_prediction_test.to_csv("wrong_predictions_test.csv", encoding='utf-8', index=False)

