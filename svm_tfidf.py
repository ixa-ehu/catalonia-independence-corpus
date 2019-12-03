# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:20 2019

@author: EZOTOVA
"""

from __future__ import print_function, division

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

import pandas as pd 
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#load datasets 
tweets_t = pd.read_csv('train_hSVM_castellano_dataset.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_t = tweets_t.fillna('')

tweets_val = pd.read_csv('val_hSVM_castellano_dataset.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_val = tweets_val.fillna('')

tweets_test = pd.read_csv('test_hSVM_castellano_dataset.csv', dtype={'id_str': 'str'}, sep='\t', encoding="utf-8")
tweets_test = tweets_test.fillna('')

frames = [tweets_t, tweets_val]
tweets_train = pd.concat(frames)

#list of wteets
tweets_train_text = list(tweets_train.LEMMA_CLEAN.values)
tweets_test_text = list(tweets_test.LEMMA_CLEAN.values)

labels_train = list(tweets_train.LABEL.values)
labels_test = list(tweets_test.LABEL.values)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(tweets_train_text)
X_test = vectorizer.transform(tweets_test_text)
print('X TRAIN ', X_train.shape)
print('X TEST ', X_test.shape)

le = preprocessing.LabelEncoder()

le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
y_train = le.transform(labels_train)
print('Y TRAIN ', y_train.shape)

#Encode the labes to integers
le = preprocessing.LabelEncoder()
le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
y_test = le.transform(labels_test)
print('Y TEST', y_test.shape)

print('=====================================================')
print('grid search')

num_features = 12000

pipeline1 = Pipeline(
    [  ("filter", SelectKBest(mutual_info_classif, k=num_features)),
    ]
)
grid_parameters_tune = [{"classification__C": [1, 10, 100, 300, 500, 700, 1000], 
                         'classification__gamma': [0, 0.1, 0.01, 0.001, 0.25, 0.5, 0.75, 1]}]
model = GridSearchCV(pipeline1, grid_parameters_tune, cv=5, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.cv_results_)

grid_result = pd.DataFrame(model.cv_results_)
grid_best = pd.DataFrame(model.best_params_, index=[0])
file_name = 'SVM_lemma/'+str(num_features)+'esp_grid_result.csv'
grid_result.to_csv(file_name, encoding='utf-8', index=False)
#select the best parameters 
df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1] 
       ("classification", svm.SVC(kernel="rbf")),
C = list(df_grid_first.param_classification__C.values)
gamma = list(df_grid_first.param_classification__gamma.values)

print('C ', C)
print('gamma ', gamma)
df_grid_first.to_csv('SVM_lemma/es_best_param_'+str(num_features)+'.csv', encoding='utf-8', index=False)


features = dict(zip(vectorizer.get_feature_names(),
               mutual_info_classif(X_tr, y_train, discrete_features=True)
               ))
#print(features)

print("===============================================")
print("Cross Validation")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

clf = Pipeline(
    [  ("filter", SelectKBest(mutual_info_classif, k=num_features)),
       ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0])),
    ]
)

cv_results = cross_validate(clf, X_train, y_train, 
                            cv=10, n_jobs=4, 
                            scoring='f1_macro')

print('CV RESULTS ', cv_results['test_score'])


y_pred = cross_val_predict(clf, X_train, y_train, 
                           cv=10, n_jobs=4)

target_names=["AGAINST", "FAVOR", "NEUTRAL"]
cl_report = classification_report(y_train, y_pred, target_names=target_names, digits=4)
cm = confusion_matrix(y_train, y_pred)
print(cm)
print("CROSS VALIDATION")
print(cl_report)

#saving classification reports
import io
report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
file_name = 'SVM_CV_report.csv'
report_df.to_csv(file_name, encoding='utf-8', index=False)
#saving predicted and wrong predicted examples
tweets_train['predicted'] = y_pred
tweets_train['true'] = y_train
tweets_train.to_csv('SVM_CV_predicted.csv', encoding='utf-8', index=False)
df1 = tweets_train.loc[tweets_train['predicted'] != tweets_train['true']]
wrong_prediction = df1[['TWEET', 'LEMMA_CLEAN', 'LABEL', 'id_str', 'predicted', 'true']]
wrong_prediction.to_csv('SVM_CV_wrong_predicrion.csv', encoding='utf-8', index=False)

print("================================================")
print("Training and Testing")

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score

clf_train = Pipeline(
    [  ("filter", SelectKBest(mutual_info_classif, k=num_features)),
       ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0])),
    ]
)

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
file_name = "SVM_test_report.csv"
report_df_test.to_csv(file_name, encoding='utf-8', sep='\t', index=False)
#save the predicted data and wrong predictions
tweets_test['predicted'] = y_pred_test
tweets_test['true'] = y_test
tweets_test.to_csv("SVM_test_predicted.csv", sep='\t', encoding='utf-8', index=False)
df2 = tweets_test.loc[tweets_test['predicted'] != tweets_test['true']]
wrong_prediction_test = df2[['TWEET', 'LEMMA_CLEAN', 'LABEL', 'id_str', 'predicted', 'true']]
wrong_prediction_test.to_csv("SVM_test_wrong_prediction.csv", encoding='utf-8', index=False)
